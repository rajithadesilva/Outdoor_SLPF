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
    2: 1.0,  # Poles (class ID 2) contribute 100%
    4: 0.5   # Trunks (class ID 4) contribute 50%
}
BEV_SIZE = (1000, 1000)
BEV_SCALE = 100
PARTICLE_COUNT = 100
PARTICLE_STD = 2.0
ANGLE_STD = np.deg2rad(10)
INIT_HEADING = np.deg2rad(110)
SENSOR_RANGE = 5.0
HORIZONTAL_FOV = np.deg2rad(87)
TF = 0.55
geojson_path = "data/riseholme_poles_trunk.geojson"
# Paths for folder-based processing
DATA_PATH = "data/2025/ICRA/"
CSV_DATA_PATH = DATA_PATH + "data.csv"

# Camera Intrinsics - replace with your camera's actual values
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

def adaptive_resample(particles, weights, ess_threshold=0.95):
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

def measurement_likelihood(grouped_map_points, bev_poles_obs, bev_trunks_obs, particles,
                           miss_penalty, wrong_hit_penalty, gps_weight,
                           gps_xy=None, gps_sigma=PARTICLE_STD):
    """
    MODIFIED: Calculates particle weights using provided penalties and also returns
    a dictionary of diagnostic statistics for the best-performing particle of the frame.
    """
    num_particles = len(particles)
    weights = np.zeros(num_particles)
    # This list will hold the diagnostic stats for each particle
    stats_per_particle = []

    # --- Pre-computation for observations ---
    obs_all_local = []
    if bev_poles_obs.size > 0:
        for obs in bev_poles_obs: obs_all_local.append({'coords': obs, 'class': 2})
    if bev_trunks_obs.size > 0:
        for obs in bev_trunks_obs: obs_all_local.append({'coords': obs, 'class': 4})

    # If no observations, we can return early.
    if not obs_all_local:
        for i, (px, py, _) in enumerate(particles):
            px = px - TF ###############
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
        
        # Return empty stats when no semantic observations are made
        empty_stats = {'gps_dist': gps_dist, 'log_gps': (-(gps_dist**2) / (2 * gps_sigma**2)), 'log_semantic': 0, 'correct_hits': 0, 'incorrect_hits': 0, 'no_hits': 0}
        return weights, empty_stats

    # --- Main loop through each particle ---
    for i, (px, py, p_theta) in enumerate(particles):
        log_semantic = 0.0
        # Initialize hit counters for this particle
        correct_hits, incorrect_hits, no_hits = 0, 0, 0
        particle_origin = np.array([px, py])

        # --- Loop through each actual observation to cast a ray ---
        for obs_data in obs_all_local:
            # ... (ray casting setup code is the same) ...
            obs_local_coords, obs_class = obs_data['coords'], obs_data['class']
            obs_range = np.linalg.norm(obs_local_coords)
            obs_angle_local = np.arctan2(obs_local_coords[0], obs_local_coords[1])
            ray_angle_world = p_theta + obs_angle_local
            ray_dir_world = np.array([np.cos(ray_angle_world), np.sin(ray_angle_world)])
            closest_hit_range, closest_hit_class = SENSOR_RANGE, -1

            for row_id, points_in_row in grouped_map_points.items():
                for j in range(len(points_in_row) - 1):
                    p1_data, p2_data = points_in_row[j], points_in_row[j+1]
                    p1_coords, p2_coords = p1_data['coords'], p2_data['coords']
                    if p1_data['class'] == 2 or p2_data['class'] == 2: segment_class = 2
                    else: segment_class = 4
                    intersection = get_ray_segment_intersection(particle_origin, ray_dir_world, p1_coords, p2_coords)
                    if intersection is not None:
                        dist = np.linalg.norm(intersection - particle_origin)
                        if dist < closest_hit_range:
                            closest_hit_range, closest_hit_class = dist, segment_class

            class_weight = CLASS_WEIGHTS.get(obs_class, 1.0)
            if closest_hit_class != -1:
                if closest_hit_class == obs_class:
                    range_error = np.abs(obs_range - closest_hit_range)
                    reward = -(range_error**2) / (2 * PARTICLE_STD**2)
                    log_semantic += class_weight * reward
                    correct_hits += 1  # Increment counter
                else:
                    reward_panelty = -(wrong_hit_penalty**2) / (2 * PARTICLE_STD**2)
                    log_semantic += class_weight * reward_panelty
                    incorrect_hits += 1 # Increment counter
            else:
                reward_panelty = -(miss_penalty**2) / (2 * PARTICLE_STD**2)
                log_semantic += class_weight * reward_panelty
                no_hits += 1 # Increment counter
        log_semantic = log_semantic / len(obs_all_local)
        # Calculate GPS score for this particle
        log_gps = 0.0
        gps_dist = 0.0
        if gps_xy is not None:
            gps_dist = np.linalg.norm(np.array([px, py]) - gps_xy)
            log_gps = -(gps_dist**2) / (2 * gps_sigma**2)

        # Calculate final log likelihood and weight
        log_likelihood = log_gps#gps_weight * log_gps + (1.0-gps_weight)*log_semantic
        weights[i] = np.exp(log_likelihood)

        # Append this particle's stats to our list
        stats_per_particle.append({
            'gps_dist': gps_dist,
            'log_gps': log_gps,
            'log_semantic': log_semantic,
            'correct_hits': correct_hits,
            'incorrect_hits': incorrect_hits,
            'no_hits': no_hits,
            'weight': weights[i]
        })

    # --- Normalize weights and select best particle's stats ---
    total_weight = np.sum(weights)
    if total_weight > 0:
        weights /= total_weight
    else:
        weights = np.ones(num_particles) / num_particles
    
    best_particle_idx = np.argmax(weights)
    stats_to_return = stats_per_particle[best_particle_idx]

    return weights, stats_to_return

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

def visualize_particle_overlap(frame_idx, overlay, particle, bev_poles_obs, bev_trunks_obs, grouped_map_points, output_dir, sensor_range=5.0):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_title(f"Particle Observation Overlap - Frame {frame_idx:04d}")
    px, py, theta = particle
    particle_origin = np.array([px, py])

    # === 1. Plot the segmented map representation (unchanged) ===
    _plot_segmented_map(ax, grouped_map_points)

    # === 2. Combine observations and plot them as circles ===
    obs_all_local = []
    if bev_poles_obs.size > 0:
        for obs in bev_poles_obs: obs_all_local.append({'coords': obs, 'class': 2})
    if bev_trunks_obs.size > 0:
        for obs in bev_trunks_obs: obs_all_local.append({'coords': obs, 'class': 4})

    if obs_all_local:
        obs_coords_local = np.array([o['coords'] for o in obs_all_local])
        obs_classes = np.array([o['class'] for o in obs_all_local])

        obs_x_local, obs_z_local = obs_coords_local[:, 0], obs_coords_local[:, 1]
        obs_x_world = px + np.cos(theta) * obs_z_local - np.sin(theta) * obs_x_local
        obs_y_world = py + np.sin(theta) * obs_z_local + np.cos(theta) * obs_x_local
        obs_all_world = np.vstack([obs_x_world, obs_y_world]).T

        ax.scatter(obs_all_world[obs_classes==2, 0], obs_all_world[obs_classes==2, 1],
                   edgecolor='blue', facecolor='none', s=100, linewidth=2.5, label='Observed Poles')
        ax.scatter(obs_all_world[obs_classes==4, 0], obs_all_world[obs_classes==4, 1],
                   edgecolor='green', facecolor='none', s=100, linewidth=2.5, label='Observed Trunks')

    # === 3. Cast and Visualize Rays from Particle to Observations ===
    # Add dummy plots for clean legend entries
    ax.plot([], [], color='green', linestyle=':', label='Correct Ray Hit')
    ax.plot([], [], color='red', linestyle=':', label='Incorrect Ray Hit')
    ax.plot([], [], color='gray', linestyle=':', label='Ray Miss')

    for obs_data in obs_all_local:
        obs_local_coords = obs_data['coords']
        obs_class = obs_data['class']

        # Define ray direction in world frame
        obs_angle_local = np.arctan2(obs_local_coords[0], obs_local_coords[1])
        ray_angle_world = theta + obs_angle_local
        ray_dir_world = np.array([np.cos(ray_angle_world), np.sin(ray_angle_world)])

        # Perform intersection check against all map segments
        closest_hit_range = sensor_range
        closest_hit_class = -1 # -1 signifies no hit

        for row_id, points_in_row in grouped_map_points.items():
            for j in range(len(points_in_row) - 1):
                p1_data = points_in_row[j]
                p2_data = points_in_row[j+1]

                # --- MODIFIED: Pole dominance rule for segment classification ---
                if p1_data['class'] == 2 or p2_data['class'] == 2:
                    segment_class = 2
                else:
                    segment_class = 4
                
                intersection = get_ray_segment_intersection(
                    particle_origin, ray_dir_world, p1_data['coords'], p2_data['coords']
                )

                if intersection is not None:
                    dist = np.linalg.norm(intersection - particle_origin)
                    if dist < closest_hit_range:
                        closest_hit_range = dist
                        # Note: The class of the segment is already determined above
                        closest_hit_class = segment_class

        # Determine ray color and endpoint based on the hit result
        if closest_hit_class != -1: # The ray hit a segment
            ray_end = particle_origin + ray_dir_world * closest_hit_range
            color = 'green' if closest_hit_class == obs_class else 'red'
        else: # The ray missed all segments
            ray_end = particle_origin + ray_dir_world * sensor_range
            color = 'gray'

        # Draw the ray
        ax.plot([particle_origin[0], ray_end[0]], [particle_origin[1], ray_end[1]], color=color, linestyle=':', linewidth=1.2)

    # === 4. Plot Particle Pose and Finalize Chart ===
    arrow_length = 0.5
    ax.quiver(px, py, np.cos(theta)*arrow_length, np.sin(theta)*arrow_length, angles='xy', scale_units='xy', scale=1, color='red', width=0.005, zorder=10, label='Particle Pose')

    ax.set_xlim(px - sensor_range, px + sensor_range)
    ax.set_ylim(py - sensor_range, py + sensor_range)
    ax.set_aspect('equal')
    ax.grid(True)
    ax.legend(fontsize='small')

    # --- Save image (code unchanged) ---
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
def process_data_with_localization(csv_data_path, rgb_dir, depth_dir, miss_penalty, wrong_hit_penalty, gps_weight,
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
    init_theta = INIT_HEADING
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

        if not (os.path.exists(rgb_path) and os.path.exists(depth_path)):
            print(f"Warning: Data missing for index {frame_idx}. RGB: {rgb_path}, Depth: {depth_path}. Skipping.")
            continue

        color_img = cv2.imread(rgb_path)
        depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED) # Assumes 16-bit PNG/TIFF

        if color_img is None or depth_img is None:
            print(f"Warning: Failed to load images for index {frame_idx}. Skipping.")
            continue

        # --- YOLO Semantic Detection ---
        results = yolo.predict(color_img, conf=0.2, classes=CLASS_IDS, verbose=False)[0]
        bev_poles_obs, bev_trunks_obs = [], []
        overlay = color_img.copy()
        if results.masks is not None:
            masks = results.masks.data.cpu().numpy()
            for i, mask in enumerate(masks):
                class_id = int(results.boxes.cls[i].item())

                mask_resized_color = cv2.resize(mask, (color_img.shape[1], color_img.shape[0]), interpolation=cv2.INTER_NEAREST)
                mask_indices_color = np.argwhere(mask_resized_color > 0.5)
                if mask_indices_color.size == 0:
                    continue
                mask_resized_depth = cv2.resize(mask, (depth_img.shape[1], depth_img.shape[0]), interpolation=cv2.INTER_NEAREST)
                mask_indices_depth = np.argwhere(mask_resized_depth > 0.5)
                if mask_indices_depth.size == 0:
                    continue

                depth_values_mm = depth_img[mask_indices_depth[:, 0], mask_indices_depth[:, 1]]
                valid_depths_mm = depth_values_mm[depth_values_mm > 0]
                if valid_depths_mm.size == 0:
                    continue
                min_depth_m = np.min(valid_depths_mm) * 0.001
                if min_depth_m == 0 or min_depth_m > 10.0:
                    continue

                u, v = np.mean(mask_indices_color, axis=0).astype(int)
                x = (v - intr.ppx) / intr.fx * min_depth_m
                y = (u - intr.ppy) / intr.fy * min_depth_m
                z = min_depth_m

                rel_x, rel_z = -x, z
                if class_id == 2:
                    bev_poles_obs.append([rel_x, rel_z])
                elif class_id == 4:
                    bev_trunks_obs.append([rel_x, rel_z])
        ##################################################################
                # Draw contours for mask
                overlay_color = (255, 0, 0) if class_id == 2 else (0, 255, 0)
                mask_vis = (mask_resized_color > 0.5).astype(np.uint8) * 255
                contours, _ = cv2.findContours(mask_vis, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(overlay, contours, -1, overlay_color, 2)
        else:
            # No masks found: reset overlay to raw image
            overlay = color_img.copy()

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
        bev_poles_obs = np.array(bev_poles_obs)
        bev_trunks_obs = np.array(bev_trunks_obs)

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
            dx_odom = current_odom_pos_x - prev_odom_pos_x
            dy_odom = current_odom_pos_y - prev_odom_pos_y
            delta_distance = np.sqrt(dx_odom ** 2 + dy_odom ** 2)

            filtered_odom_yaw = 0.8*current_odom_yaw + 0.2*prev_odom_yaw
            delta_theta = filtered_odom_yaw - prev_odom_yaw
            delta_theta = (delta_theta + np.pi) % (2 * np.pi) - np.pi # Normalize angle

            particles = motion_update(particles, delta_distance, delta_theta)

        # Update the previous odometry state for the next iteration
        prev_odom_pos_x = current_odom_pos_x
        prev_odom_pos_y = current_odom_pos_y
        prev_odom_yaw = current_odom_yaw

        # --- Measurement Update ---
        weights, frame_stats = measurement_likelihood(
            grouped_map_points, bev_poles_obs, bev_trunks_obs, particles,
            miss_penalty=miss_penalty, wrong_hit_penalty=wrong_hit_penalty, gps_weight=gps_weight,
            gps_xy=(gps_x_noisy, gps_y_noisy)
        )
        frame_stats['frame_idx'] = frame_idx
        with open(CSV_OUTPUT_PATH, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=stats_fieldnames)
            writer.writerow(frame_stats)
        
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
                bev_poles_obs, bev_trunks_obs,
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
    parser.add_argument('--gps-weight', type=float, default=0.8,
                        help='A complementary weight coefficient for the GPS error in the likelihood estimation.')
    args = parser.parse_args()

    print(f"[INFO] Running with Miss Penalty: {args.miss_penalty}, Wrong Hit Penalty: {args.wrong_hit_penalty}, GPS Weight: {args.gps_weight}")

    process_data_with_localization(
        csv_data_path=CSV_DATA_PATH,
        rgb_dir=DATA_PATH,
        depth_dir=DATA_PATH,
        miss_penalty=args.miss_penalty,
        wrong_hit_penalty=args.wrong_hit_penalty,
        gps_weight=args.gps_weight,
        output_folder=f"amcl_output/experiment/NOisy_GPS_{args.gps_weight}"
    )
    print("[INFO] Finished processing all frames from the CSV file.")