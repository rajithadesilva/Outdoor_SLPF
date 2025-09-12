import os
import cv2
import numpy as np
import csv
import argparse
from ultralytics import YOLO
from tqdm import tqdm
import torch
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timezone, timedelta
from pyproj import Transformer


# ---------- CONFIG ----------
FRAME_STRIDE = 1
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
yolo = YOLO("./models/yolo.pt").to(device)
CLASS_IDS = [2, 4]
CLASS_NAMES = {2: 'poles', 4: 'trunks'}
CLASS_WEIGHTS = {
    0: 0.5,  # Background (class 0) contribute 50%  
    2: 1.0,  # Poles (class ID 2) contribute 100%
    4: 1.0   # Trunks (class ID 4) contribute 50%
}
BEV_SIZE = (1000, 1000)
BEV_SCALE = 100
PARTICLE_COUNT = 100
PARTICLE_STD = 2.0
SEMANTIC_SIGMA = 0.05
GPS_SIGMA = 1.1
ANGLE_STD = np.deg2rad(10)
INIT_HEADING = np.deg2rad(110)
SENSOR_RANGE = 5.0
HORIZONTAL_FOV = np.deg2rad(87)
CAMERA_BASE_TF = 0.55

# Lidar Params
LIDAR_RANGE = 4.0
SEMANTIC_RADIUS = 0.5
LIDAR_TO_CAMERA_DX = 0.0  # meters (forward)
LIDAR_TO_CAMERA_DY = 0.0  # meters (left)

geojson_path = "data/riseholme_poles_trunk.geojson"
# Paths for folder-based processing
DATA_PATH = "data/2025/ICRA2/"
CSV_DATA_PATH = DATA_PATH + "data.csv"

# Camera Intrinsics
class Intrinsics:
    def __init__(self):
        self.width_depth = 848
        self.height_depth = 480
        self.width_color = 1280
        self.height_color = 720
        self.x_scale = self.width_color / self.width_depth
        self.y_scale = self.height_color / self.height_depth
        self.ppx = 426.27 * self.x_scale   # principal point x
        self.ppy = 241.27 * self.y_scale   # principal point y
        self.fx = 419.92 * self.x_scale    # focal length x
        self.fy = 419.92 * self.y_scale    # focal length y

intr = Intrinsics()


# ---------- UTILS ----------
def quaternion_to_yaw(x, y, z, w):
    """
    Convert a quaternion into a yaw angle (rotation around the z-axis).
    """
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = np.arctan2(t3, t4)
    return yaw_z

def yaw_to_quaternion(yaw):
    """Converts a yaw angle to a quaternion (qx, qy, qz, qw)."""
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    # Assuming pitch (p) and roll (r) are zero
    cp = 1.0 # cos(0 * 0.5)
    sp = 0.0 # sin(0 * 0.5)
    cr = 1.0 # cos(0 * 0.5)
    sr = 0.0 # sin(0 * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    return qx, qy, qz, qw

def save_tum_trajectory(trajectory_data, output_path):
    """
    Saves the trajectory data in TUM format.
    Args:
        trajectory_data (list of tuples): Each tuple contains (timestamp, x, y, theta).
        output_path (str): The path to save the .tum file.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        # TUM format: timestamp tx ty tz qx qy qz qw
        f.write("# timestamp tx ty tz qx qy qz qw\n")
        for timestamp, x, y, theta in trajectory_data:
            qx, qy, qz, qw = yaw_to_quaternion(theta)
            f.write(f"{timestamp} {x} {y} 0.0 {qx} {qy} {qz} {qw}\n")
    print(f"[INFO] Trajectory saved to {output_path}")

def load_landmarks_as_lines(path):
    """
    Loads landmarks and groups them by row, preserving individual points.
    The points within each row are sorted to ensure correct adjacency.
    """
    gdf = gpd.read_file(path)

    # Create a unified 'row_id' column
    def extract_row_id(row):
        if row['feature_type'] == 'vine' and 'vine_vine_row_id' in row and row['vine_vine_row_id']:
            return row['vine_vine_row_id']
        elif row['feature_type'] == 'row_post' and 'feature_name' in row and row['feature_name']:
            parts = row['feature_name'].split('_')
            return f"{parts[0]}_{parts[1]}" if len(parts) >= 2 else None
        return None

    gdf['row_id'] = gdf.apply(extract_row_id, axis=1)
    gdf.dropna(subset=['row_id'], inplace=True)

    # Transform coordinates and classify
    if not gdf.crs or not gdf.crs.is_projected:
        gdf = gdf.to_crs(gdf.estimate_utm_crs())

    center = gdf.geometry.union_all().centroid
    gdf['x_centered'] = gdf.geometry.x - center.x
    gdf['y_centered'] = gdf.geometry.y - center.y

    def classify(row):
        return 2 if row['feature_type'] == 'row_post' else 4
    gdf["class_id"] = gdf.apply(classify, axis=1)

    # Group points into a dictionary, with points in each row sorted
    grouped_points = {}
    for row_id, group in gdf.groupby('row_id'):
        points_data = group[['x_centered', 'y_centered']].values

        # Sort points within the row to ensure they are in order
        if np.ptp(points_data[:, 0]) < np.ptp(points_data[:, 1]):
            sorted_indices = np.argsort(points_data[:, 1]) # Sort by Y for vertical rows
        else:
            sorted_indices = np.argsort(points_data[:, 0]) # Sort by X for horizontal rows

        sorted_group = group.iloc[sorted_indices]

        grouped_points[row_id] = []
        for _, point_row in sorted_group.iterrows():
            grouped_points[row_id].append({
                'coords': np.array([point_row['x_centered'], point_row['y_centered']]),
                'class': point_row['class_id']
            })

    return grouped_points, np.array([center.x, center.y])

def load_lidar_frame_from_csv(
    lidar_csv_path: str,
    max_range: float = None,
    fov_radians: float = None,
    angle_center_rad: float = 0.0,
    drop_out_of_range: bool = True,
    cap_to_max: bool = False
):
    """
    Read a single LiDAR scan CSV into a structured lidar_frame dict.

    Parameters
    ----------
    lidar_csv_path : str
        Path to CSV with header lines starting with '#', then rows:
        beam_index,angle_rad,range_m,intensity
    max_range : float, optional
        If provided, beams with range > max_range are either set to NaN
        (drop_out_of_range=True) or clamped to max_range (cap_to_max=True).
    fov_radians : float, optional
        If provided, keep only beams with angle in
        [angle_center - fov/2, angle_center + fov/2].
        (Angles are in the LaserScan convention: 0 rad forward, CCW positive.)
    angle_center_rad : float, default 0.0
        Center angle for FOV cropping.
    drop_out_of_range : bool, default True
        If True and max_range is set, mark ranges > max_range as NaN.
        If False and cap_to_max is False, keep original values.
    cap_to_max : bool, default False
        If True and max_range is set, clamp ranges to max_range.

    Returns
    -------
    lidar_frame : dict
        {
            'meta': {
                'angle_min': float,
                'angle_max': float,
                'angle_increment': float,
                'time_increment': float,
                'scan_time': float,
                'range_min': float,
                'range_max': float,
            },
            'angles': np.ndarray shape (N,),
            'ranges': np.ndarray shape (N,),
            'intensity': np.ndarray shape (N,) or None,
            'xy': np.ndarray shape (N,2),  # x forward, y left
            'mask_valid': np.ndarray bool shape (N,), # True where range is finite
        }
    """

    # ---- 1) Parse header lines beginning with '#' ----
    meta = {
        'angle_min': None, 'angle_max': None, 'angle_increment': None,
        'time_increment': None, 'scan_time': None,
        'range_min': None, 'range_max': None
    }
    rows = []
    with open(lidar_csv_path, 'r') as f:
        # First, read all lines to handle mixed header/data easily
        raw_lines = f.readlines()

    # Extract meta and find where table starts
    start_idx = 0
    for i, line in enumerate(raw_lines):
        s = line.strip()
        if s.startswith('#'):
            # header in form "# key,value"
            parts = s[1:].split(',', 1)
            if len(parts) == 2:
                key = parts[0].strip()
                val = parts[1].strip()
                try:
                    val_f = float(val)
                except ValueError:
                    val_f = None
                # map known keys
                keymap = {
                    'angle_min_rad': 'angle_min',
                    'angle_max_rad': 'angle_max',
                    'angle_increment_rad': 'angle_increment',
                    'time_increment_s': 'time_increment',
                    'scan_time_s': 'scan_time',
                    'range_min_m': 'range_min',
                    'range_max_m': 'range_max',
                }
                if key in keymap:
                    meta[keymap[key]] = val_f
            continue
        else:
            # first non-# line should be CSV header
            start_idx = i
            break

    # ---- 2) Parse the beam table ----
    # Expect header like: beam_index,angle_rad,range_m,intensity
    table_lines = raw_lines[start_idx:]
    reader = csv.DictReader(table_lines)
    angles = []
    ranges = []
    intens = []

    for r in reader:
        try:
            angle = float(r.get('angle_rad', 'nan'))
        except ValueError:
            angle = np.nan
        try:
            rng = float(r.get('range_m', 'nan'))
        except ValueError:
            rng = np.nan

        angles.append(angle)
        ranges.append(rng)

        # intensity may be empty
        inten_raw = r.get('intensity', '')
        try:
            inten_val = float(inten_raw) if inten_raw.strip() != '' else np.nan
        except ValueError:
            inten_val = np.nan
        intens.append(inten_val)

    angles = np.asarray(angles, dtype=float)
    ranges = np.asarray(ranges, dtype=float)
    intensity = np.asarray(intens, dtype=float) if len(intens) == len(angles) else None

    # ---- 3) Optional FOV crop ----
    if fov_radians is not None and np.isfinite(fov_radians):
        half = float(fov_radians) * 0.5
        lo = angle_center_rad - half
        hi = angle_center_rad + half
        fov_mask = (angles >= lo) & (angles <= hi)
        angles = angles[fov_mask]
        ranges = ranges[fov_mask]
        if intensity is not None:
            intensity = intensity[fov_mask]

    # ---- 4) Apply max range rule if requested ----
    if max_range is not None and np.isfinite(max_range):
        if drop_out_of_range:
            ranges = np.where(ranges > max_range, np.nan, ranges)
        elif cap_to_max:
            ranges = np.minimum(ranges, max_range)
        # else keep original values

    # Also respect sensor's own range limits from meta if available
    if meta['range_min'] is not None:
        ranges = np.where(ranges < meta['range_min'], np.nan, ranges)
    if meta['range_max'] is not None:
        # Only mark as NaN if we didn't explicitly clamp above
        if not cap_to_max:
            ranges = np.where(ranges > meta['range_max'], np.nan, ranges)

    # ---- 5) Convert to Cartesian (ROS LaserScan convention) ----
    # x = r*cos(theta) (forward), y = r*sin(theta) (left)
    x = ranges * np.cos(angles)
    y = ranges * np.sin(angles)
    xy = np.stack([x, y], axis=1)

    mask_valid = np.isfinite(ranges)

    lidar_frame = {
        'meta': meta,
        'angles': angles,
        'ranges': ranges,
        'intensity': intensity,
        'xy': xy,
        'mask_valid': mask_valid,
    }
    return lidar_frame

def load_csv_with_utm(csv_path, noise_std=PARTICLE_STD):
    """
    Loads CSV data, converts lat/lon to UTM coordinates, and adds optional noise.
    """
    df = pd.read_csv(csv_path)
    # Assumes WGS 84 (EPSG:4326) and UTM zone 30N (EPSG:32630)
    transformer = Transformer.from_crs("epsg:4326", "epsg:32630", always_xy=True)
    df["utm_easting"], df["utm_northing"] = transformer.transform(df["longitude"].values, df["latitude"].values)
    df["utm_easting_noisy"] = df["utm_easting"] + np.random.normal(0, noise_std, len(df))
    df["utm_northing_noisy"] = df["utm_northing"] + np.random.normal(0, noise_std, len(df))
    return df

def initialize_particles(n, extent):
    low = np.array(extent[0])
    high = np.array(extent[1])
    return np.random.uniform(low=low, high=high, size=(n, 3))

def initialize_particles_around_pose(center_pose, std_dev=(PARTICLE_STD, PARTICLE_STD, ANGLE_STD), count=PARTICLE_COUNT):
    """
    Initialize particles around a given center pose with Gaussian noise.

    Args:
        center_pose: tuple (x, y, theta) - center pose to sample around.
        std_dev: tuple of standard deviations (x_std, y_std, theta_std)
        count: number of particles

    Returns:
        particles: np.ndarray of shape (count, 3)
    """
    x0, y0, theta0 = center_pose
    x_samples = np.random.normal(x0, std_dev[0], count)
    y_samples = np.random.normal(y0, std_dev[1], count)
    theta_samples = np.random.normal(theta0, std_dev[2], count)
    return np.stack([x_samples, y_samples, theta_samples], axis=-1)

def motion_update(particles, delta_distance, delta_theta, noise_std=(0.1, 0.1, ANGLE_STD)):
    N = len(particles)
    noise = np.random.normal(0, 0.2, size=(N, 2))
    noise_angle = np.random.normal(0, ANGLE_STD, size=(N, 1))

    for i in range(N):
        theta = particles[i, 2]
        if delta_distance < 0.1:
            delta_distance = 0
            delta_theta = 0
        # Forward motion in local frame projected to global map frame
        dx = delta_distance * np.cos(theta)
        dy = delta_distance * np.sin(theta)

        # Apply motion + noise
        particles[i, 0] += dx + noise[i, 0]
        particles[i, 1] += dy + noise[i, 1]
        particles[i, 2] += delta_theta + noise_angle[i, 0]

    return particles

def effective_sample_size(weights):
    return 1.0 / np.sum(weights ** 2)

def adaptive_resample(particles, weights, ess_threshold=0.99):
    ess = effective_sample_size(weights)
    if ess < ess_threshold * len(particles):
        idx = np.random.choice(len(particles), size=len(particles), p=weights)
        return particles[idx]
    return particles

def get_ray_segment_intersection(ray_origin, ray_dir, p1, p2):
    """
    Finds the intersection point of a ray and a line segment.

    Args:
        ray_origin (np.array): The (x, y) starting point of the ray.
        ray_dir (np.array): The (x, y) direction vector of the ray.
        p1 (np.array): The (x, y) start point of the line segment.
        p2 (np.array): The (x, y) end point of the line segment.

    Returns:
        np.array: The (x, y) intersection point, or None if they don't intersect.
    """
    v1 = ray_origin - p1
    v2 = p2 - p1
    v3 = np.array([-ray_dir[1], ray_dir[0]])  # Vector perpendicular to the ray direction

    dot_v2_v3 = np.dot(v2, v3)
    if np.abs(dot_v2_v3) < 1e-6:  # Avoid division by zero if lines are parallel
        return None

    # Calculate the 2D cross product manually to avoid NumPy 2.0 deprecation warnings
    cross_product_mag = v2[0] * v1[1] - v2[1] * v1[0]

    t1 = cross_product_mag / dot_v2_v3
    t2 = np.dot(v1, v3) / dot_v2_v3

    # Check if the intersection is along the ray's forward direction (t1 >= 0)
    # and within the bounds of the line segment (0 <= t2 <= 1)
    if t1 >= 0.0 and 0.0 <= t2 <= 1.0:
        return ray_origin + t1 * ray_dir

    return None

def measurement_likelihood(grouped_map_points,
                           bev_poles_obs,
                           bev_trunks_obs,
                           bev_background_obs,
                           particles,
                           miss_penalty,
                           wrong_hit_penalty,
                           gps_weight,
                           gps_xy=None,
                           gps_sigma=GPS_SIGMA):
    """
    Calculates particle weights from semantic (poles/trunks) and background LiDAR points.
    Background rays (class 0) are free-space checks:
      - If a ray intersects any map segment, we treat it like a 'correct hit' and
        use the range error to reward proximity (no class mismatch notion).
      - If it doesn't intersect any segment, we apply a miss penalty.

    Returns
    -------
    weights: (N,) normalized particle weights
    stats_to_return: dict with diagnostics for the best particle
    """
    num_particles = len(particles)
    weights = np.zeros(num_particles)
    stats_per_particle = []

    # --- Collect observations ---
    obs_all_local = []
    if bev_poles_obs.size > 0:
        for obs in bev_poles_obs:
            obs_all_local.append({'coords': obs, 'class': 2})
    if bev_trunks_obs.size > 0:
        for obs in bev_trunks_obs:
            obs_all_local.append({'coords': obs, 'class': 4})
    if bev_background_obs is not None and bev_background_obs.size > 0:
        for obs in bev_background_obs:
            obs_all_local.append({'coords': obs, 'class': 0})  # 0 = background/free-space check

    # If no observations, fall back to GPS only
    if not obs_all_local:
        for i, (px, py, _) in enumerate(particles):
            px = px - CAMERA_BASE_TF  # your existing camera base offset
            if gps_xy is not None:
                gps_dist = np.linalg.norm(np.array([px, py]) - gps_xy)
                weights[i] = np.exp(-(gps_dist**2) / (2 * gps_sigma**2))
            else:
                weights[i] = 1.0

        total_weight = np.sum(weights)
        if total_weight > 0:
            weights /= total_weight
        else:
            weights = np.ones(num_particles) / num_particles

        # Simple stats (no semantic)
        gps_dist = float(gps_dist) if gps_xy is not None else 0.0
        empty_stats = {
            'gps_dist': gps_dist,
            'log_gps': (-(gps_dist**2) / (2 * gps_sigma**2)) if gps_xy is not None else 0.0,
            'log_semantic': 0.0,
            'correct_hits': 0,
            'incorrect_hits': 0,
            'no_hits': 0,
            'weight': float(np.max(weights))
        }
        return weights, empty_stats

    # --- Main scoring ---
    for i, (px, py, p_theta) in enumerate(particles):
        log_semantic = 0.0
        correct_hits, incorrect_hits, no_hits = 0, 0, 0
        particle_origin = np.array([px, py], dtype=float)

        for obs_data in obs_all_local:
            obs_local_coords = np.asarray(obs_data['coords'], dtype=float)
            obs_class = int(obs_data['class'])

            # Observation in local BEV (left=x, forward=z)
            obs_range = float(np.linalg.norm(obs_local_coords))
            # angle wrt particle's forward (+x world) -> arctan2(left, forward)
            obs_angle_local = float(np.arctan2(obs_local_coords[0], obs_local_coords[1]))
            ray_angle_world = float(p_theta + obs_angle_local)
            ray_dir_world = np.array([np.cos(ray_angle_world), np.sin(ray_angle_world)], dtype=float)

            closest_hit_range, closest_hit_class = SENSOR_RANGE, -1

            # Intersect against map segments
            for row_id, points_in_row in grouped_map_points.items():
                for j in range(len(points_in_row) - 1):
                    p1_data = points_in_row[j]
                    p2_data = points_in_row[j + 1]
                    p1_coords = p1_data['coords']
                    p2_coords = p2_data['coords']

                    # Segment class by pole dominance rule
                    segment_class = 2 if (p1_data['class'] == 2 or p2_data['class'] == 2) else 4

                    intersection = get_ray_segment_intersection(
                        particle_origin, ray_dir_world, p1_coords, p2_coords
                    )
                    if intersection is not None:
                        dist = np.linalg.norm(intersection - particle_origin)
                        if dist < closest_hit_range:
                            closest_hit_range, closest_hit_class = dist, segment_class

            class_weight = CLASS_WEIGHTS.get(obs_class, 1.0)

            if obs_class in (2, 4):
                # Semantic classes: poles/trunks with mismatch handling
                if closest_hit_class != -1:
                    if closest_hit_class == obs_class:
                        range_error = abs(obs_range - closest_hit_range)
                        reward = -(range_error**2) / (2 * SEMANTIC_SIGMA**2)
                        log_semantic += class_weight * reward
                        correct_hits += 1
                    else:
                        penalty = -(wrong_hit_penalty**2) / (2 * SEMANTIC_SIGMA**2)
                        log_semantic += class_weight * penalty
                        incorrect_hits += 1
                else:
                    penalty = -(miss_penalty**2) / (2 * SEMANTIC_SIGMA**2)
                    log_semantic += class_weight * penalty
                    no_hits += 1

            else:
                # Background (class 0): free-space check
                if closest_hit_class != -1:
                    # There is a map surface along the ray -> compare ranges (treat like 'correct')
                    range_error = abs(obs_range - closest_hit_range)
                    reward = -(range_error**2) / (2 * SEMANTIC_SIGMA**2)
                    log_semantic += class_weight * reward
                    correct_hits += 1
                else:
                    # No map surface along this ray within SENSOR_RANGE -> miss penalty
                    penalty = -(miss_penalty**2) / (2 * SEMANTIC_SIGMA**2)
                    log_semantic += class_weight * penalty
                    no_hits += 1

        # Average per observation to keep scale stable vs. obs count
        log_semantic /= len(obs_all_local)

        # GPS term
        gps_dist = 0.0
        log_gps = 0.0
        if gps_xy is not None:
            gps_dist = float(np.linalg.norm(np.array([px, py]) - gps_xy))
            log_gps = -(gps_dist**2) / (2 * gps_sigma**2)

        # Fuse GPS and semantic
        log_likelihood = gps_weight * log_gps + (1.0 - gps_weight) * log_semantic
        weights[i] = np.exp(log_likelihood)

        stats_per_particle.append({
            'gps_dist': gps_dist,
            'log_gps': log_gps,
            'log_semantic': log_semantic,
            'correct_hits': correct_hits,
            'incorrect_hits': incorrect_hits,
            'no_hits': no_hits,
            'weight': weights[i]
        })

    # Normalize & pick best
    total_weight = float(np.sum(weights))
    if total_weight > 0:
        weights /= total_weight
    else:
        weights[:] = 1.0 / num_particles

    best_particle_idx = int(np.argmax(weights))
    stats_to_return = stats_per_particle[best_particle_idx]
    return weights, stats_to_return

def build_segment_tensors(grouped_map_points, device='cuda'):
    """Convert grouped_map_points to flat torch tensors on the target device."""
    p1_list, p2_list, seg_cls_list = [], [], []
    for _, points_in_row in grouped_map_points.items():
        if len(points_in_row) < 2:
            continue
        for j in range(len(points_in_row) - 1):
            p1 = points_in_row[j]
            p2 = points_in_row[j + 1]
            # pole-dominance rule
            seg_class = 2 if (p1['class'] == 2 or p2['class'] == 2) else 4
            p1_list.append(p1['coords'])
            p2_list.append(p2['coords'])
            seg_cls_list.append(seg_class)

    p1 = torch.as_tensor(np.array(p1_list, dtype=np.float32), device=device)  # (M,2)
    p2 = torch.as_tensor(np.array(p2_list, dtype=np.float32), device=device)  # (M,2)
    seg_cls = torch.as_tensor(np.array(seg_cls_list, dtype=np.int64), device=device)  # (M,)
    v2 = p2 - p1  # (M,2)

    return p1, p2, v2, seg_cls

def measurement_likelihood_gpu(grouped_map_points_unused,
                               bev_poles_obs,
                               bev_trunks_obs,
                               bev_background_obs,
                               particles,
                               miss_penalty,
                               wrong_hit_penalty,
                               gps_weight,
                               gps_xy=None,
                               gps_sigma=GPS_SIGMA,
                               *,
                               seg_p1=None, seg_p2=None, seg_v2=None, seg_cls=None,
                               sensor_range=SENSOR_RANGE,
                               class_weights=CLASS_WEIGHTS,
                               device='cuda',
                               segment_chunk=4096,
                               # --- New knobs ---
                               normalize='robust',     # {'robust','zscore',None}
                               clamp_norm=6.0,         # clamp normalized terms to [-clamp_norm, +clamp_norm]
                               softmax_temp=1.0):      # temperature for softmax (>=1.0 smooths)
    """
    GPU vectorized measurement likelihood with per-frame log-term normalization.

    Normalization (per-frame across particles):
      - 'robust': x' = (x - median(x)) / (1.4826 * MAD(x) + eps)
      - 'zscore': x' = (x - mean(x)) / (std(x) + eps)
      - None:     no normalization (original scaling)

    Then fuse: log_like' = gps_weight * log_gps' + (1 - gps_weight) * log_semantic'
    Finally: weights = softmax(log_like'/softmax_temp)
    """
    assert seg_p1 is not None and seg_p2 is not None and seg_v2 is not None and seg_cls is not None, \
        "Provide precomputed segment tensors via build_segment_tensors()."
    torch_device = torch.device(device)
    eps = 1e-12

    # ---- Pack observations (to torch, on device) ----
    obs_list, obs_classes = [], []
    if bev_poles_obs is not None and bev_poles_obs.size > 0:
        obs_list.append(torch.as_tensor(bev_poles_obs, dtype=torch.float32, device=torch_device))
        obs_classes.append(torch.full((bev_poles_obs.shape[0],), 2, dtype=torch.int64, device=torch_device))
    if bev_trunks_obs is not None and bev_trunks_obs.size > 0:
        obs_list.append(torch.as_tensor(bev_trunks_obs, dtype=torch.float32, device=torch_device))
        obs_classes.append(torch.full((bev_trunks_obs.shape[0],), 4, dtype=torch.int64, device=torch_device))
    if bev_background_obs is not None and bev_background_obs.size > 0:
        obs_list.append(torch.as_tensor(bev_background_obs, dtype=torch.float32, device=torch_device))
        obs_classes.append(torch.full((bev_background_obs.shape[0],), 0, dtype=torch.int64, device=torch_device))

    if len(obs_list) == 0:
        # GPS-only fallback (vectorized)
        parts = torch.as_tensor(particles[:, :2], dtype=torch.float32, device=torch_device)  # (N,2)
        if gps_xy is not None:
            gps_t = torch.as_tensor(gps_xy, dtype=torch.float32, device=torch_device)
            d = torch.linalg.norm(parts - gps_t, dim=1)
            log_gps = -(d**2) / (2.0 * (gps_sigma**2))
            # normalize GPS-only too (helps avoid peaky exp)
            if normalize is not None:
                if normalize == 'robust':
                    med = torch.median(log_gps)
                    mad = torch.median((log_gps - med).abs())
                    scale = 1.4826 * mad + eps
                    log_gps_n = torch.clamp((log_gps - med) / scale, -clamp_norm, clamp_norm)
                elif normalize == 'zscore':
                    mean = torch.mean(log_gps)
                    std = torch.std(log_gps) + eps
                    log_gps_n = torch.clamp((log_gps - mean) / std, -clamp_norm, clamp_norm)
                else:
                    log_gps_n = log_gps
            else:
                log_gps_n = log_gps
            logits = log_gps_n / max(softmax_temp, 1e-6)
            logits = logits - logits.max()  # stable softmax
            weights = torch.softmax(logits, dim=0)
        else:
            weights = torch.full((particles.shape[0],), 1.0 / particles.shape[0],
                                 dtype=torch.float32, device=torch_device)

        best_idx = int(torch.argmax(weights).item())
        stats = {
            'gps_dist': float(d.max().item()) if gps_xy is not None else 0.0,
            'log_gps': float(log_gps.max().item()) if gps_xy is not None else 0.0,
            'log_semantic': 0.0,
            'correct_hits': 0,
            'incorrect_hits': 0,
            'no_hits': 0,
            'weight': float(weights[best_idx].item())
        }
        return weights.detach().cpu().numpy(), stats

    obs_all = torch.cat(obs_list, dim=0)      # (J,2) [left,forward]
    obs_cls = torch.cat(obs_classes, dim=0)   # (J,)
    J = obs_all.shape[0]
    N = particles.shape[0]

    # ---- Precompute per-observation constants on GPU ----
    obs_range = torch.linalg.norm(obs_all, dim=1)                  # (J,)
    obs_ang_local = torch.atan2(obs_all[:, 0], obs_all[:, 1])      # (J,)

    # ---- Particles on GPU ----
    parts_xy = torch.as_tensor(particles[:, :2], dtype=torch.float32, device=torch_device)  # (N,2)
    parts_th = torch.as_tensor(particles[:, 2], dtype=torch.float32, device=torch_device)   # (N,)

    # ---- Build rays for all (N,J) ----
    ray_angle_world = parts_th[:, None] + obs_ang_local[None, :]  # (N,J)
    ray_dir = torch.stack([torch.cos(ray_angle_world), torch.sin(ray_angle_world)], dim=-1)  # (N,J,2)
    O = parts_xy[:, None, :].expand(-1, J, -1)  # (N,J,2)
    v3 = torch.stack([-ray_dir[..., 1], ray_dir[..., 0]], dim=-1)  # (N,J,2)

    # ---- Intersections in chunks ----
    closest_hit_range = torch.full((N, J), sensor_range, dtype=torch.float32, device=torch_device)
    closest_hit_class = torch.full((N, J), -1, dtype=torch.int64, device=torch_device)

    def cross2d(a, b):  # (...,2) x (...,2)
        return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]

    Nseg = seg_p1.shape[0]
    for start in range(0, Nseg, segment_chunk):
        end = min(start + segment_chunk, Nseg)
        p1_chunk = seg_p1[start:end]   # (M,2)
        v2_chunk = seg_v2[start:end]   # (M,2)
        cls_chunk = seg_cls[start:end] # (M,)

        v1 = O[:, :, None, :] - p1_chunk[None, None, :, :]                 # (N,J,M,2)
        denom = (v2_chunk[None, None, :, :] * v3[:, :, None, :]).sum(-1)   # (N,J,M)
        parallel = denom.abs() < 1e-8

        t1 = cross2d(v2_chunk[None, None, :, :], v1) / (denom + 1e-12)
        t2 = (v1 * v3[:, :, None, :]).sum(-1) / (denom + 1e-12)

        valid = (~parallel) & (t1 >= 0.0) & (t2 >= 0.0) & (t2 <= 1.0)
        if not valid.any():
            continue

        dist = torch.where(valid, t1, torch.full_like(t1, float('inf')))
        min_dist, min_idx = dist.min(dim=-1)  # (N,J)
        improved = min_dist < closest_hit_range

        closest_hit_range = torch.where(improved, min_dist, closest_hit_range)
        new_cls = cls_chunk[min_idx.clamp_min(0)]
        closest_hit_class = torch.where(improved, new_cls, closest_hit_class)

    # ---- Per-ray contributions ----
    cw = torch.ones(J, dtype=torch.float32, device=torch_device)
    if class_weights:
        for k, v in class_weights.items():
            cw = torch.where(obs_cls == int(k), torch.tensor(float(v), device=torch_device), cw)
    cw = cw[None, :]  # (1,J)

    obs_range_b = obs_range[None, :]  # (1,J)
    any_hit = closest_hit_class >= 0
    range_err = (obs_range_b - closest_hit_range).abs()
    reward = -(range_err**2) / (2.0 * (SEMANTIC_SIGMA**2))
    miss_pen = - (miss_penalty**2) / (2.0 * (SEMANTIC_SIGMA**2))
    wrong_pen = - (wrong_hit_penalty**2) / (2.0 * (SEMANTIC_SIGMA**2))

    is_sem = (obs_cls[None, :] == 2) | (obs_cls[None, :] == 4)
    hit_and_match = any_hit & is_sem & (closest_hit_class == obs_cls[None, :])
    hit_and_mismatch = any_hit & is_sem & (closest_hit_class != obs_cls[None, :])
    miss_sem = (~any_hit) & is_sem

    contrib_sem = torch.zeros((N, J), dtype=torch.float32, device=torch_device)
    contrib_sem = torch.where(hit_and_match, reward, contrib_sem)
    contrib_sem = torch.where(hit_and_mismatch, torch.full_like(contrib_sem, wrong_pen), contrib_sem)
    contrib_sem = torch.where(miss_sem, torch.full_like(contrib_sem, miss_pen), contrib_sem)

    is_bg = (obs_cls[None, :] == 0)
    hit_bg = any_hit & is_bg
    miss_bg = (~any_hit) & is_bg
    contrib_bg = torch.zeros((N, J), dtype=torch.float32, device=torch_device)
    contrib_bg = torch.where(hit_bg, reward, contrib_bg)
    contrib_bg = torch.where(miss_bg, torch.full_like(contrib_bg, miss_pen), contrib_bg)

    contrib = torch.where(is_sem, contrib_sem, contrib_bg)
    contrib = contrib * cw
    log_semantic = contrib.mean(dim=1)  # (N,)

    # ---- GPS term ----
    if gps_xy is not None:
        gps_t = torch.as_tensor(gps_xy, dtype=torch.float32, device=torch_device)
        d_gps = torch.linalg.norm(parts_xy - gps_t, dim=1)  # (N,)
        log_gps = -(d_gps**2) / (2.0 * (gps_sigma**2))      # (N,)
    else:
        d_gps = torch.zeros(N, dtype=torch.float32, device=torch_device)
        log_gps = torch.zeros(N, dtype=torch.float32, device=torch_device)

    # ---- NEW: per-frame normalization of log terms (across particles) ----
    def robust_norm(x: torch.Tensor) -> torch.Tensor:
        med = torch.median(x)
        mad = torch.median((x - med).abs())
        scale = 1.4826 * mad + eps
        xn = (x - med) / scale
        if clamp_norm is not None:
            xn = torch.clamp(xn, -clamp_norm, clamp_norm)
        return xn

    def zscore_norm(x: torch.Tensor) -> torch.Tensor:
        mean = torch.mean(x)
        std = torch.std(x) + eps
        xn = (x - mean) / std
        if clamp_norm is not None:
            xn = torch.clamp(xn, -clamp_norm, clamp_norm)
        return xn

    if normalize == 'robust':
        log_sem_n = robust_norm(log_semantic)
        log_gps_n = robust_norm(log_gps)
    elif normalize == 'zscore':
        log_sem_n = zscore_norm(log_semantic)
        log_gps_n = zscore_norm(log_gps)
    else:
        log_sem_n = log_semantic
        log_gps_n = log_gps

    # ---- Fuse & softmax with temperature ----
    fused = gps_weight * log_gps_n + (1.0 - gps_weight) * log_sem_n  # (N,)
    logits = fused / max(softmax_temp, 1e-6)
    logits = logits - logits.max()  # stable softmax
    weights = torch.softmax(logits, dim=0)

    # ---- Diagnostics for best particle ----
    best_idx = int(torch.argmax(weights).item())
    correct_hits = (hit_and_match[best_idx].sum() + hit_bg[best_idx].sum()).item()
    incorrect_hits = hit_and_mismatch[best_idx].sum().item()
    no_hits = (miss_sem[best_idx].sum() + miss_bg[best_idx].sum()).item()

    stats = {
        'gps_dist': float(d_gps[best_idx].item()),
        'log_gps': float(log_gps[best_idx].item()),
        'log_semantic': float(log_semantic[best_idx].item()),
        'weight': float(weights[best_idx].item()),
        'correct_hits': int(correct_hits),
        'incorrect_hits': int(incorrect_hits),
        'no_hits': int(no_hits),
    }

    return weights.detach().cpu().numpy(), stats

def _plot_segmented_map(ax, grouped_map_points, linestyle='-', label_prefix=""):
    """
    MODIFIED: Helper function to draw the map. A segment is blue if at least 
    one endpoint is a pole. It's green only if both endpoints are trunks.
    """
    # Create dummy plots to ensure legend entries are created
    ax.plot([], [], color='blue', linestyle=linestyle, label=f'{label_prefix} Pole Segments')
    ax.plot([], [], color='green', linestyle=linestyle, label=f'{label_prefix} Trunk Segments')
    ax.scatter([], [], c='blue', marker='x', label=f'{label_prefix} Pole Points')
    ax.scatter([], [], c='green', marker='x', label=f'{label_prefix} Trunk Points')

    for row_id, points_in_row in grouped_map_points.items():
        # First, plot all the 'x' markers for the individual landmark points
        for point_data in points_in_row:
            coords = point_data['coords']
            color = 'blue' if point_data['class'] == 2 else 'green'
            ax.scatter(coords[0], coords[1], c=color, marker='x', s=40, zorder=5)

        # Then, plot the connecting segments based on the pole dominance rule
        for i in range(len(points_in_row) - 1):
            p1_data = points_in_row[i]
            p2_data = points_in_row[i+1]

            p1_coords = p1_data['coords']
            p2_coords = p2_data['coords']

            # Pole dominance rule: if either point is a pole (class 2), the segment is a pole segment.
            if p1_data['class'] == 2 or p2_data['class'] == 2:
                segment_color = 'blue'
            else:
                segment_color = 'green' # Only if both are trunks

            # Draw the full segment with the determined color
            ax.plot([p1_coords[0], p2_coords[0]], [p1_coords[1], p2_coords[1]], color=segment_color, linestyle=linestyle, linewidth=1.5)

def visualize_particle_overlap(frame_idx, overlay, particle,
                               bev_poles_obs, bev_trunks_obs, bev_background_obs,
                               grouped_map_points, output_dir,
                               sensor_range=5.0):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_title(f"Particle Observation Overlap - Frame {frame_idx:04d}")
    px, py, theta = particle
    particle_origin = np.array([px, py])

    # 1) Map (segmented)
    _plot_segmented_map(ax, grouped_map_points)

    # 2) Gather observations
    obs_all_local = []
    if bev_poles_obs.size > 0:
        for obs in bev_poles_obs:
            obs_all_local.append({'coords': np.asarray(obs, float), 'class': 2})
    if bev_trunks_obs.size > 0:
        for obs in bev_trunks_obs:
            obs_all_local.append({'coords': np.asarray(obs, float), 'class': 4})
    if bev_background_obs is not None and getattr(bev_background_obs, "size", 0) > 0:
        for obs in bev_background_obs:
            obs_all_local.append({'coords': np.asarray(obs, float), 'class': 0})

    # 3) Plot observation points (in world frame)
    if obs_all_local:
        obs_coords_local = np.array([o['coords'] for o in obs_all_local])
        obs_classes = np.array([o['class'] for o in obs_all_local])

        # local [left, forward] -> world
        x_left, z_fwd = obs_coords_local[:, 0], obs_coords_local[:, 1]
        x_world = px + np.cos(theta) * z_fwd - np.sin(theta) * x_left
        y_world = py + np.sin(theta) * z_fwd + np.cos(theta) * x_left
        obs_world = np.vstack([x_world, y_world]).T

        ax.scatter(obs_world[obs_classes == 2, 0], obs_world[obs_classes == 2, 1],
                   edgecolor='blue', facecolor='none', s=100, linewidth=2.0, label='Pole Obs')
        ax.scatter(obs_world[obs_classes == 4, 0], obs_world[obs_classes == 4, 1],
                   edgecolor='green', facecolor='none', s=100, linewidth=2.0, label='Trunk Obs')
        ax.scatter(obs_world[obs_classes == 0, 0], obs_world[obs_classes == 0, 1],
                   edgecolor='brown', facecolor='none', s=90, linewidth=2.0, label='Background Obs')

    # Legend dummies for ray styles
    ax.plot([], [], color='green', linestyle='--', label='Ray Hit')
    ax.plot([], [], color='gray', linestyle='--', label='Ray Miss')

    # 4) Rays: HIT vs MISS only
    for obs in obs_all_local:
        obs_local = obs['coords']
        obs_class = int(obs['class'])

        # Build ray in world
        obs_angle_local = np.arctan2(obs_local[0], obs_local[1])  # atan2(left, forward)
        ray_angle_world = theta + obs_angle_local
        ray_dir_world = np.array([np.cos(ray_angle_world), np.sin(ray_angle_world)], dtype=float)

        # Find nearest segment intersection
        closest_hit_range = sensor_range
        hit_found = False
        for _, points_in_row in grouped_map_points.items():
            for j in range(len(points_in_row) - 1):
                p1 = points_in_row[j]['coords']
                p2 = points_in_row[j + 1]['coords']
                intersection = get_ray_segment_intersection(particle_origin, ray_dir_world, p1, p2)
                if intersection is not None:
                    dist = np.linalg.norm(intersection - particle_origin)
                    if dist < closest_hit_range:
                        closest_hit_range = dist
                        hit_found = True

        # For background points, we consider HIT if a segment is intersected
        # before the background point's range; otherwise it's a MISS.
        if obs_class == 0:
            obs_range = float(np.linalg.norm(obs_local))
            if hit_found and closest_hit_range < obs_range:
                # HIT (segment appears before background distance)
                ray_end = particle_origin + ray_dir_world * closest_hit_range
                ray_color = 'green'
            else:
                # MISS (no segment up to background distance)
                ray_end = particle_origin + ray_dir_world * min(obs_range, sensor_range)
                ray_color = 'gray'
        else:
            # Poles/trunks: HIT if any segment is intersected (ignore class)
            if hit_found:
                ray_end = particle_origin + ray_dir_world * closest_hit_range
                ray_color = 'green'
            else:
                ray_end = particle_origin + ray_dir_world * sensor_range
                ray_color = 'gray'

        ax.plot([particle_origin[0], ray_end[0]], [particle_origin[1], ray_end[1]],
                color=ray_color, linestyle='--', linewidth=1.3)

    # 5) Pose
    arrow_length = 0.5
    ax.quiver(px, py, np.cos(theta) * arrow_length, np.sin(theta) * arrow_length,
              angles='xy', scale_units='xy', scale=1, color='red', width=0.005, zorder=10,
              label='Particle Pose')

    ax.set_xlim(px - sensor_range, px + sensor_range)
    ax.set_ylim(py - sensor_range, py + sensor_range)
    ax.set_aspect('equal')
    ax.grid(True)
    ax.legend(fontsize='small')

    # Save composite with RGB overlay on the left
    particle_dir = os.path.join(output_dir, "particle_overlap")
    os.makedirs(particle_dir, exist_ok=True)
    right_img_path = os.path.join(particle_dir, f"frame_{frame_idx:04d}_right.jpg")
    plt.savefig(right_img_path, bbox_inches='tight')
    plt.close()

    right_img = cv2.imread(right_img_path)
    if right_img is not None:
        left_img = cv2.resize(overlay, (right_img.shape[1], right_img.shape[0]))
        combined = np.hstack((left_img, right_img))
        final_img_path = os.path.join(particle_dir, f"frame_{frame_idx:04d}.jpg")
        cv2.imwrite(final_img_path, combined)
        os.remove(right_img_path)

def visualize_particles(grouped_map_points, particles, frame_idx, output_dir, trajectory, gps_trajectory, rgb_overlay=None):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_title(f"Particles - Frame {frame_idx:04d}")

    # Plot the full map using the new segmented style
    _plot_segmented_map(ax, grouped_map_points, label_prefix="Map")

    # Determine plot bounds from map points
    all_map_points = [p['coords'] for row in grouped_map_points.values() for p in row]
    if all_map_points:
        all_map_points = np.array(all_map_points)
        x_min_lmk, x_max_lmk = np.min(all_map_points[:, 0]), np.max(all_map_points[:, 0])
        y_min_lmk, y_max_lmk = np.min(all_map_points[:, 1]), np.max(all_map_points[:, 1])

        all_x = np.concatenate([particles[:, 0], np.array(trajectory)[:, 0] if trajectory else [], np.array(gps_trajectory)[:, 0] if gps_trajectory else []])
        all_y = np.concatenate([particles[:, 1], np.array(trajectory)[:, 1] if trajectory else [], np.array(gps_trajectory)[:, 1] if gps_trajectory else []])

        padding = 5.0
        x_min = min(x_min_lmk, all_x.min()) - padding
        x_max = max(x_max_lmk, all_x.max()) + padding
        y_min = min(y_min_lmk, all_y.min()) - padding
        y_max = max(y_max_lmk, all_y.max()) + padding
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

    # Plot trajectories and particles
    if len(gps_trajectory) > 1:
        ax.plot(np.array(gps_trajectory)[:, 0], np.array(gps_trajectory)[:, 1], 'k--', linewidth=1.5, label="GPS Path")
    if len(trajectory) > 1:
        ax.plot(np.array(trajectory)[:, 0], np.array(trajectory)[:, 1], 'r-', linewidth=2, label="Estimated Path")
    ax.quiver(particles[:, 0], particles[:, 1], np.cos(particles[:, 2])*0.5, np.sin(particles[:, 2])*0.5, angles='xy', scale_units='xy', scale=1, color='red', width=0.005, alpha=0.7, label='Particles')

    ax.legend()
    ax.grid(True)
    ax.set_aspect('equal')

    # Save visualization
    vis_path = os.path.join(output_dir, f"particles/frame_{frame_idx:04d}.jpg")
    os.makedirs(os.path.dirname(vis_path), exist_ok=True)
    plot_path = vis_path.replace('.jpg', '_plot.jpg')
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()

    plot_img = cv2.imread(plot_path)
    if rgb_overlay is not None and plot_img is not None:
        rgb_resized = cv2.resize(rgb_overlay, (plot_img.shape[1], plot_img.shape[0]))
        combo = np.hstack((rgb_resized, plot_img))
        cv2.imwrite(vis_path, combo)
        os.remove(plot_path)
    elif plot_img is not None:
        os.rename(plot_path, vis_path)

# ---------- MAIN ----------
def process_data_with_localization(csv_data_path, rgb_dir, depth_dir, lidar_dir, miss_penalty, wrong_hit_penalty, gps_weight,
                                     output_folder="amcl_output"):
    os.makedirs(os.path.join(output_folder, "particles"), exist_ok=True)
    df_data = load_csv_with_utm(csv_data_path)
    grouped_map_points, center = load_landmarks_as_lines(geojson_path)

    """ Initialize particles based on landmarks extent
    all_coords = np.vstack([poles_coords, trunks_coords])
    extent = [(all_coords[:, 0].min(), all_coords[:, 1].min(), -np.pi),
              (all_coords[:, 0].max()+5, all_coords[:, 1].max()+5, np.pi)]
    particles = initialize_particles(PARTICLE_COUNT, extent=extent)
    #"""

    #""" Initial GPS-based pose (centered coordinates, heading assumed 0)
    first_row = df_data.iloc[0]
    init_x = first_row["utm_easting"] - center[0]
    init_y = first_row["utm_northing"] - center[1]
    #init_theta = INIT_HEADING
    init_theta = quaternion_to_yaw(first_row['odom_orient_x'],
                               first_row['odom_orient_y'],
                               first_row['odom_orient_z'],
                               first_row['odom_orient_w'])
    init_theta = init_theta + np.deg2rad(180)

    particles = initialize_particles_around_pose(
        center_pose=(init_x, init_y, init_theta),
        std_dev=(PARTICLE_STD, PARTICLE_STD, np.deg2rad(360)),
        count=PARTICLE_COUNT
    )
    #"""

    full_trajectory_data = [] # Will store (timestamp, x, y, theta)
    gps_trajectory = []
    noisy_gps_trajectory = []
    # Initialize odometry state variables
    prev_odom_pos_x, prev_odom_pos_y, prev_odom_yaw = None, None, None

    stats_fieldnames = ['frame_idx', 'gps_dist', 'log_gps', 'log_semantic', 'correct_hits', 'incorrect_hits', 'no_hits', 'weight']
    CSV_OUTPUT_PATH = os.path.join(output_folder, "stats.csv")
    
    # Write the header to the CSV file once at the beginning
    with open(CSV_OUTPUT_PATH, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=stats_fieldnames)
        writer.writeheader()

    # Main loop over the CSV file rows
    pbar = tqdm(df_data.iterrows(), total=df_data.shape[0], desc="Processing frames")
    for frame_idx, row in pbar:
        if frame_idx % FRAME_STRIDE != 0:
            continue
        
        # --- Load images ---
        rgb_path = os.path.join(rgb_dir, row['rgb_image'])
        depth_path = os.path.join(depth_dir, row['depth_image'])
        lidar_path = os.path.join(lidar_dir, row['lidar_csv'])


        if not (os.path.exists(rgb_path) and os.path.exists(depth_path)):
            print(f"Warning: Data missing for index {frame_idx}. RGB: {rgb_path}, Depth: {depth_path}. Skipping.")
            continue

        color_img = cv2.imread(rgb_path)
        depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED) # Assumes 16-bit PNG/TIFF
        lidar_frame = load_lidar_frame_from_csv(lidar_path, LIDAR_RANGE)

        if color_img is None or depth_img is None:
            print(f"Warning: Failed to load images for index {frame_idx}. Skipping.")
            continue

        # --- YOLO Semantic Detection ---
        results = yolo.predict(color_img, conf=0.2, classes=CLASS_IDS, verbose=False)[0]
        bev_poles_obs, bev_trunks_obs = [], []
        overlay = color_img.copy()
        semantic_centers = []  # list of tuples: (cx_rel_x, cz_rel_z, class_id)

        if results.masks is not None:
            masks = results.masks.data.cpu().numpy()
            for i, mask in enumerate(masks):
                class_id = int(results.boxes.cls[i].item())
                if class_id not in (2, 4):
                    continue

                # Resize mask to color & depth sizes
                mask_resized_color = cv2.resize(mask, (color_img.shape[1], color_img.shape[0]),
                                                interpolation=cv2.INTER_NEAREST)
                mask_indices_color = np.argwhere(mask_resized_color > 0.5)
                if mask_indices_color.size == 0:
                    continue

                mask_resized_depth = cv2.resize(mask, (depth_img.shape[1], depth_img.shape[0]),
                                                interpolation=cv2.INTER_NEAREST)
                mask_indices_depth = np.argwhere(mask_resized_depth > 0.5)
                if mask_indices_depth.size == 0:
                    continue

                # Depth sampling (min depth in mask area, ignoring zeros)
                depth_values_mm = depth_img[mask_indices_depth[:, 0], mask_indices_depth[:, 1]]
                valid_depths_mm = depth_values_mm[depth_values_mm > 0]
                if valid_depths_mm.size == 0:
                    continue
                min_depth_m = float(np.min(valid_depths_mm)) * 0.001
                if min_depth_m == 0 or min_depth_m > 10.0:
                    continue

                # Mask centroid in color image (u=row, v=col)
                u, v = np.mean(mask_indices_color, axis=0).astype(int)

                # Back-project to camera coordinates (OpenNI-like):
                # x_cam = (v - cx)/fx * z, y_cam = (u - cy)/fy * z, z_cam = z
                x_cam = (v - intr.ppx) / intr.fx * min_depth_m   # right (+)
                y_cam = (u - intr.ppy) / intr.fy * min_depth_m   # down (+)
                z_cam = min_depth_m                               # forward (+)

                # BEV convention used in your code: rel_x (left +), rel_z (forward +)
                # Convert camera coords: left = -x_cam, forward = z_cam
                rel_x, rel_z = -x_cam, z_cam

                # Save as BEV observations from camera
                if class_id == 2:
                    bev_poles_obs.append([rel_x, rel_z])
                else:
                    bev_trunks_obs.append([rel_x, rel_z])

                # Keep center for semantic circles (used to classify LiDAR beams)
                semantic_centers.append((rel_x, rel_z, class_id))

                # Draw mask contour on overlay for debug
                overlay_color = (255, 0, 0) if class_id == 2 else (0, 255, 0)
                mask_vis = (mask_resized_color > 0.5).astype(np.uint8) * 255
                contours, _ = cv2.findContours(mask_vis, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(overlay, contours, -1, overlay_color, 2)
        else:
            # No masks found: reset overlay to raw image
            overlay = color_img.copy()

        # --- Use semantic circles to label LiDAR beams ---
        bev_poles_obs_lidar, bev_trunks_obs_lidar, background_lidar = [], [], []

        if semantic_centers and lidar_frame is not None and 'xy' in lidar_frame:
            # LiDAR points in LiDAR frame: x_fwd, y_left
            xy_lidar = lidar_frame['xy']              # shape (N,2)
            mask_valid = lidar_frame.get('mask_valid', np.isfinite(lidar_frame['ranges']))
            xy_lidar = xy_lidar[mask_valid]

            if xy_lidar.size > 0:
                # Transform LiDAR points into camera-origin BEV coordinates:
                # shift by LiDAR->Camera (forward/left), then map to (rel_x, rel_z) = (left, forward)
                x_fwd = xy_lidar[:, 0] - LIDAR_TO_CAMERA_DX
                y_left = xy_lidar[:, 1] - LIDAR_TO_CAMERA_DY

                lidar_rel_x = y_left
                lidar_rel_z = x_fwd
                lidar_bev = np.stack([lidar_rel_x, lidar_rel_z], axis=1)  # (N,2)

                # Prepare semantic circle centers
                centers = np.array([[cx, cz] for (cx, cz, _) in semantic_centers], dtype=float)  # (M,2)
                classes = np.array([cid for (_, _, cid) in semantic_centers], dtype=int)         # (M,)

                # Compute nearest center for each LiDAR point (vectorized)
                # distances^2 to all centers: (N,M)
                diffs = lidar_bev[:, None, :] - centers[None, :, :]
                d2 = np.sum(diffs * diffs, axis=2)
                nn_idx = np.argmin(d2, axis=1)
                nn_d = np.sqrt(d2[np.arange(d2.shape[0]), nn_idx])

                # Classify by radius threshold
                inside = nn_d <= SEMANTIC_RADIUS
                nn_classes = classes[nn_idx]

                # Split into semantic vs background
                sem_points = lidar_bev[inside]
                sem_classes = nn_classes[inside]
                bg_points = lidar_bev[~inside]

                if sem_points.size > 0:
                    poles_mask = (sem_classes == 2)
                    trunks_mask = (sem_classes == 4)

                    if np.any(poles_mask):
                        bev_poles_obs_lidar = sem_points[poles_mask].tolist()
                    if np.any(trunks_mask):
                        bev_trunks_obs_lidar = sem_points[trunks_mask].tolist()

                if bg_points.size > 0:
                    background_lidar = bg_points.tolist()
        """
        if len(bev_poles_obs) + len(bev_trunks_obs) > 0:
            # Prepare BEV visualization image
            debug_bev = np.zeros(BEV_SIZE, dtype=np.uint8)
            origin = (BEV_SIZE[1] // 2, BEV_SIZE[0])  # center-bottom

            for px, pz in bev_poles_obs:
                mx = int(origin[0] - px * BEV_SCALE)
                mz = int(origin[1] - pz * BEV_SCALE)
                if 0 <= mx < BEV_SIZE[1] and 0 <= mz < BEV_SIZE[0]:
                    cv2.circle(debug_bev, (mx, mz), 4, 180, -1)

            for tx, tz in bev_trunks_obs:
                mx = int(origin[0] - tx * BEV_SCALE)
                mz = int(origin[1] - tz * BEV_SCALE)
                if 0 <= mx < BEV_SIZE[1] and 0 <= mz < BEV_SIZE[0]:
                    cv2.circle(debug_bev, (mx, mz), 4, 255, -1)

            # Resize and convert to color
            debug_bev_color = cv2.cvtColor(debug_bev, cv2.COLOR_GRAY2BGR)
            debug_bev_resized = cv2.resize(debug_bev_color, (overlay.shape[1], overlay.shape[0]))

            # Concatenate RGB overlay and BEV projection side by side
            combined_debug = np.hstack((overlay, debug_bev_resized))

            # Save to file
            vis_dir = os.path.join(output_folder, "debug_bev")
            os.makedirs(vis_dir, exist_ok=True)
            vis_path = os.path.join(vis_dir, f"frame_{frame_idx:04d}.jpg")
            cv2.imwrite(vis_path, combined_debug)
        #"""
        ##################################################################
        # Convert to numpy arrays
        bev_poles_obs = np.array(bev_poles_obs_lidar)
        bev_trunks_obs = np.array(bev_trunks_obs_lidar)
        bev_background_obs = np.array(background_lidar)

        # Get GPS data for current frame (used for measurement update)
        gps_x = row["utm_easting"] - center[0]
        gps_y = row["utm_northing"] - center[1]
        gps_x_noisy = row["utm_easting_noisy"] - center[0]
        gps_y_noisy = row["utm_northing_noisy"] - center[1]

        # Get odometry data for the current frame
        current_odom_pos_x = row['odom_pos_x'] 
        current_odom_pos_y = row['odom_pos_y']
        current_odom_yaw = quaternion_to_yaw(
            row['odom_orient_x'],
            row['odom_orient_y'],
            row['odom_orient_z'],
            row['odom_orient_w']
        )
        # If we have a previous state, calculate the change and update particles
        if prev_odom_pos_x is not None:
            dx_odom = (current_odom_pos_x - prev_odom_pos_x)
            dy_odom = current_odom_pos_y - prev_odom_pos_y
            delta_distance = np.sqrt(dx_odom ** 2 + dy_odom ** 2)

            filtered_odom_yaw = 0.9*current_odom_yaw + 0.1*prev_odom_yaw
            delta_theta = filtered_odom_yaw - prev_odom_yaw
            delta_theta = (delta_theta + np.pi) % (2 * np.pi) - np.pi # Normalize angle

            particles = motion_update(particles, delta_distance, delta_theta)

        # Update the previous odometry state for the next iteration
        prev_odom_pos_x = current_odom_pos_x
        prev_odom_pos_y = current_odom_pos_y
        prev_odom_yaw = current_odom_yaw

        seg_p1, seg_p2, seg_v2, seg_cls = build_segment_tensors(grouped_map_points, device=device)
        weights, frame_stats = measurement_likelihood_gpu(
            grouped_map_points,   # not used (kept for signature parity)
            bev_poles_obs,
            bev_trunks_obs,
            bev_background_obs,
            particles,            # np.ndarray (N,3)
            miss_penalty=miss_penalty,
            wrong_hit_penalty=wrong_hit_penalty,
            gps_weight=gps_weight,
            gps_xy=(gps_x_noisy, gps_y_noisy),
            gps_sigma=GPS_SIGMA,
            seg_p1=seg_p1, seg_p2=seg_p2, seg_v2=seg_v2, seg_cls=seg_cls,
            sensor_range=SENSOR_RANGE,
            class_weights=CLASS_WEIGHTS,
            device=device,
            segment_chunk=4096,   # adjust if you have many segments / limited VRAM
        )
        """ --- Measurement Update ---
        weights, frame_stats = measurement_likelihood(
            grouped_map_points, bev_poles_obs, bev_trunks_obs, bev_background_obs, particles,
            miss_penalty=miss_penalty, wrong_hit_penalty=wrong_hit_penalty, gps_weight=gps_weight,
            gps_xy=(gps_x_noisy, gps_y_noisy)
        )
        frame_stats['frame_idx'] = frame_idx
        with open(CSV_OUTPUT_PATH, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=stats_fieldnames)
            writer.writerow(frame_stats)
        """
        if np.sum(weights) > 0:
            weights /= np.sum(weights)
        else:
            # Handle case of zero weights, e.g., re-initialize or assign uniform weights
            weights = np.ones(PARTICLE_COUNT) / PARTICLE_COUNT

        #""" Visualize overlap for best particle
        highest_weight_index = np.argmax(weights)
        best_particle = particles[highest_weight_index]
        if bev_poles_obs.size > 0 or bev_trunks_obs.size > 0:
            visualize_particle_overlap(
                frame_idx, overlay, best_particle,
                bev_poles_obs, bev_trunks_obs, bev_background_obs,
                grouped_map_points,
                output_folder,
                sensor_range=SENSOR_RANGE
            )
        #"""

        est_pose = np.average(particles, axis=0, weights=weights)
        
        # Store full pose data for TUM export, using frame_idx as the timestamp
        full_trajectory_data.append((frame_idx, est_pose[0], est_pose[1], est_pose[2]))
        gps_trajectory.append([gps_x, gps_y])
        noisy_gps_trajectory.append((frame_idx, gps_x_noisy, gps_y_noisy, 0))
        
        particles = adaptive_resample(particles, weights)

        # Create a simple list of (x, y) for the visualization function
        trajectory_xy = [(t[1], t[2]) for t in full_trajectory_data]

        visualize_particles(
            grouped_map_points,
            particles, frame_idx,
            output_folder, trajectory_xy, gps_trajectory,
            rgb_overlay=overlay
        )

    # --- After loop, save trajectory to TUM file ---
    tum_output_dir = os.path.join(output_folder)
    tum_filename = f"trajectory_{gps_weight}.tum"
    tum_output_path = os.path.join(tum_output_dir, tum_filename)
    save_tum_trajectory(full_trajectory_data, tum_output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run AMCL with configurable penalties and weights.")
    parser.add_argument('--miss-penalty', type=float, default=4.0,
                        help='Penalty value for a ray not hitting any map feature.')
    parser.add_argument('--wrong-hit-penalty', type=float, default=4.0,
                        help='Penalty value for a ray hitting a map feature of the wrong class.')
    parser.add_argument('--gps-weight', type=float, default=0.5,
                        help='A complementary weight coefficient for the GPS error in the likelihood estimation.')
    args = parser.parse_args()

    print(f"[INFO] Running with Miss Penalty: {args.miss_penalty}, Wrong Hit Penalty: {args.wrong_hit_penalty}, GPS Weight: {args.gps_weight}")

    process_data_with_localization(
        csv_data_path=CSV_DATA_PATH,
        rgb_dir=DATA_PATH,
        depth_dir=DATA_PATH,
        lidar_dir=DATA_PATH,
        miss_penalty=args.miss_penalty,
        wrong_hit_penalty=args.wrong_hit_penalty,
        gps_weight=args.gps_weight,
        output_folder=f"amcl_output/ICRA/spf_lidar/{args.gps_weight}/"
    )
    print("[INFO] Finished processing all frames from the CSV file.")