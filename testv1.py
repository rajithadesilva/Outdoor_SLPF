import os
import cv2
import numpy as np
import csv
# import pyrealsense2 as rs # No longer needed
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
FRAME_STRIDE = 5
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
yolo = YOLO("./models/yolo.pt").to(device)
CLASS_IDS = [2, 4]
CLASS_NAMES = {2: 'poles', 4: 'trunks'}
CLASS_WEIGHTS = {
    2: 0.8,  # Poles (class ID 2) contribute 80%
    4: 0.2   # Trunks (class ID 4) contribute 20%
}
BEV_SIZE = (1000, 1000)
BEV_SCALE = 100
PARTICLE_COUNT = 100
PARTICLE_STD = 2.0
ANGLE_STD = np.deg2rad(10)
INIT_HEADING = np.deg2rad(-80)
SENSOR_RANGE = 4.0
HORIZONTAL_FOV = np.deg2rad(87)
MAX_RAY_HALF_ANGLE = np.deg2rad(5)
RECOVERY_THRESHOLD = 4.0
LIDAR_FOV_DEG = 180.0                # Horizontal Field of View for the synthetic lidar
LIDAR_NUM_RAYS = 91                 # Number of rays (odd number is good to have a center ray)
LIDAR_RANGE_STD = 0.01               # (meters) Standard deviation for range error in likelihood
SEMANTIC_MATCH_BONUS = 2.0    
geojson_path = "data/riseholme_poles_trunk.geojson"
# Paths for folder-based processing
DATA_PATH = "data/2025/1/"
CSV_DATA_PATH = DATA_PATH + "gps.csv"
RGB_DIR = DATA_PATH
DEPTH_DIR = DATA_PATH
CSV_OUTPUT_PATH=f'amcl_output/csv/likelihood_stats_{PARTICLE_COUNT}_{PARTICLE_STD}.csv'

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
def load_landmarks_from_geojson(path):
    gdf = gpd.read_file(path)
    if not gdf.crs or not gdf.crs.is_projected:
        gdf = gdf.to_crs(gdf.estimate_utm_crs())
    gdf["x"] = gdf.geometry.x
    gdf["y"] = gdf.geometry.y
    coords = gdf[["x", "y"]].values
    center = coords.mean(axis=0)
    coords -= center
    gdf["x_centered"] = coords[:, 0]
    gdf["y_centered"] = coords[:, 1]

    def classify(row):
        name = f"{row.get('feature_type', '')} {row.get('feature_name', '')}".lower()
        if "post" in name or "pole" in name:
            return "pole"
        elif "trunk" in name or "vine" in name:
            return "trunk"
        else:
            return "other"

    gdf["semantic"] = gdf.apply(classify, axis=1)
    poles = gdf[gdf["semantic"] == "pole"]
    trunks = gdf[gdf["semantic"] == "trunk"]
    return poles[["x_centered", "y_centered"]].values, trunks[["x_centered", "y_centered"]].values, gdf, center

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
    noise = np.random.normal(0, noise_std, size=(N, 3))

    for i in range(N):
        theta = particles[i, 2]

        # Forward motion in local frame projected to global map frame
        dx = delta_distance * np.cos(theta)
        dy = delta_distance * np.sin(theta)

        # Apply motion + noise
        particles[i, 0] += dx + noise[i, 0]
        particles[i, 1] += dy + noise[i, 1]
        particles[i, 2] += delta_theta + noise[i, 2]

    return particles

def effective_sample_size(weights):
    return 1.0 / np.sum(weights ** 2)

def adaptive_resample(particles, weights, ess_threshold=0.95):
    ess = effective_sample_size(weights)
    if ess < ess_threshold * len(particles):
        idx = np.random.choice(len(particles), size=len(particles), p=weights)
        return particles[idx]
    return particles

def adaptive_resample2(particles, weights, gps_xy, ess_threshold=0.95, recovery_threshold=RECOVERY_THRESHOLD, recovery_fraction=1.0):
    """
    Resamples particles, with a recovery mechanism for when the filter diverges.
    """
    # 1. Check for divergence by measuring distance to GPS
    particle_positions = particles[:, :2]
    avg_dist_to_gps = np.mean(np.linalg.norm(particle_positions - gps_xy, axis=1))

    # 2. Trigger recovery if diverged
    if avg_dist_to_gps > recovery_threshold:
        print(f"INFO: Particle cloud diverged (avg_dist={avg_dist_to_gps:.2f}m > {recovery_threshold}m). Triggering recovery.")
        
        num_to_replace = int(len(particles) * recovery_fraction)
        
        # Find the indices of particles with the lowest weights
        worst_particle_indices = np.argsort(weights)[:num_to_replace]

        # Create new particles sampled around the current GPS pose
        # Note: We assume a wide heading uncertainty during recovery
        new_particles = initialize_particles_around_pose(
            center_pose=(gps_xy[0], gps_xy[1], np.random.uniform(-np.pi, np.pi)),
            std_dev=(PARTICLE_STD, PARTICLE_STD, np.deg2rad(360)),
            count=num_to_replace
        )

        # Replace the worst particles with the new ones
        particles[worst_particle_indices] = new_particles
        
        # Reset weights to be uniform after injection to give new particles a chance
        weights.fill(1.0 / len(particles))
        
        # No need to resample further, as we've just reset a portion of the cloud
        return particles, weights

    # 3. If not diverged, perform standard adaptive resampling
    else:
        ess = effective_sample_size(weights)
        if ess < ess_threshold * len(particles):
            idx = np.random.choice(len(particles), size=len(particles), p=weights)
            return particles[idx], np.full(len(particles), 1.0 / len(particles))
        return particles, weights

def measurement_likelihood2(map_poles, map_trunks, bev_poles_obs, bev_trunks_obs, particles,
    gps_xy=None, gps_sigma=PARTICLE_STD, csv_path=CSV_OUTPUT_PATH):
    sigma = PARTICLE_STD
    penalty_threshold = 0.5
    penalty_lambda = 1.0

    weights = np.zeros(len(particles))

    obs_all = []
    classes = []
    if bev_poles_obs.size > 0:
        obs_all.extend(bev_poles_obs.tolist())
        classes.extend([2] * len(bev_poles_obs))
    if bev_trunks_obs.size > 0:
        obs_all.extend(bev_trunks_obs.tolist())
        classes.extend([4] * len(bev_trunks_obs))

    if not obs_all:
        return np.ones(len(particles)) / len(particles)

    # To store per-particle likelihood components for stats
    per_particle_log_match = np.zeros(len(particles))
    per_particle_log_structure = np.zeros(len(particles))
    per_particle_log_penalty = np.zeros(len(particles))
    per_particle_log_gps = np.zeros(len(particles))
    per_particle_gps_dist = np.zeros(len(particles))  # store GPS distances

    for i, (px, py, theta) in enumerate(particles):
        obs_world = []
        for ox, oz in obs_all:
            mx = px + np.cos(theta) * oz - np.sin(theta) * ox
            my = py + np.sin(theta) * oz + np.cos(theta) * ox
            obs_world.append((mx, my))
        obs_world = np.array(obs_world)

        log_match = 0.0
        log_structure = 0.0
        log_penalty = 0.0
        log_gps = 0.0

        # Landmark Matching
        for obs, cls in zip(obs_world, classes):
            map_candidates = map_poles if cls == 2 else map_trunks
            if len(map_candidates) > 0:
                dists = np.linalg.norm(map_candidates - obs, axis=1)
                closest_dist = np.min(dists)
                if closest_dist < penalty_threshold:
                    log_match += - ((closest_dist ** 2) / (2 * sigma ** 2))*1000

        """ Spatial Layout Consistency
        for j in range(len(obs_world)):
            for k in range(j + 1, len(obs_world)):
                if classes[j] != classes[k]:
                    obs_dist = np.linalg.norm(obs_world[j] - obs_world[k])

                    if classes[j] == 2:
                        map_from = map_poles
                        map_to = map_trunks
                    else:
                        map_from = map_trunks
                        map_to = map_poles

                    best_pair_score = -np.inf
                    for m1 in map_from:
                        for m2 in map_to:
                            map_dist = np.linalg.norm(m1 - m2)
                            diff = obs_dist - map_dist
                            pair_score = - (diff ** 2) / (2 * sigma ** 2)
                            best_pair_score = max(best_pair_score, pair_score)

                    log_structure += best_pair_score*100
        """

        # Negative Evidence Penalty
        false_positives = 0
        for obs, cls in zip(obs_world, classes):
            map_candidates = map_poles if cls == 2 else map_trunks
            if len(map_candidates) == 0:
                false_positives += 1
                continue
            dists = np.linalg.norm(map_candidates - obs, axis=1)
            if np.min(dists) > penalty_threshold:
                false_positives += 1
        log_penalty = (-penalty_lambda * false_positives)*5

        # GPS Proximity Boost
        gps_dist = 0.0
        if gps_xy is not None:
            dx = px - gps_xy[0]
            dy = py - gps_xy[1]
            gps_dist = np.sqrt(dx**2 + dy**2)
            log_gps = - (gps_dist ** 2) / (2 * gps_sigma ** 2)

        # Final total log-likelihood (you can adjust this formula as needed)
        log_total = log_penalty + log_match

        weights[i] = np.exp(log_total) * np.exp(-gps_dist)

        # Store for stats
        per_particle_log_match[i] = log_match
        per_particle_log_structure[i] = log_structure
        per_particle_log_penalty[i] = log_penalty
        per_particle_log_gps[i] = log_gps
        per_particle_gps_dist[i] = gps_dist  # store GPS dist

    # Normalize weights
    total = np.sum(weights)
    if total == 0:
        weights[:] = 1.0 / len(particles)
    else:
        weights /= total

    # Write statistics to CSV if requested
    if csv_path is not None:
        median_x = np.median(particles[:, 0])
        median_y = np.median(particles[:, 1])
        std_x = np.std(particles[:, 0])
        std_y = np.std(particles[:, 1])

        mean_log_match = np.median(per_particle_log_match)
        mean_log_structure = np.median(per_particle_log_structure)
        mean_log_penalty = np.median(per_particle_log_penalty)
        mean_log_gps = np.median(per_particle_log_gps)
        mean_gps_dist = np.median(per_particle_gps_dist)

        header = [
            'median_x', 'median_y', 'std_x', 'std_y',
            'mean_log_match', 'mean_log_structure', 'mean_log_penalty', 'mean_log_gps',
            'mean_gps_dist'
        ]
        row = [
            median_x, median_y, std_x, std_y,
            mean_log_match, mean_log_structure, mean_log_penalty, mean_log_gps,
            mean_gps_dist
        ]

        file_exists = os.path.isfile(csv_path)
        with open(csv_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            if not file_exists:
                writer.writerow(header)
            writer.writerow(row)

    return weights

def get_ray_segment_intersection(ray_origin, ray_dir, p1, p2):
    """ Finds the intersection of a ray with a line segment without NumPy 2.0 warnings. """
    v1 = ray_origin - p1
    v2 = p2 - p1
    v3 = np.array([-ray_dir[1], ray_dir[0]]) # Perpendicular to ray direction

    dot_v2_v3 = np.dot(v2, v3)
    if np.abs(dot_v2_v3) < 1e-6: # Parallel lines
        return None

    # Calculate 2D cross product manually to avoid deprecation warning
    cross_product_mag = v2[0] * v1[1] - v2[1] * v1[0]
    
    t1 = cross_product_mag / dot_v2_v3
    t2 = np.dot(v1, v3) / dot_v2_v3

    if t1 >= 0.0 and 0.0 <= t2 <= 1.0:
        return ray_origin + t1 * ray_dir
    return None

def measurement_likelihood(map_poles, map_trunks, bev_poles_obs, bev_trunks_obs, particles,
                           gps_xy=None, gps_sigma=PARTICLE_STD):
    """
    Calculates particle weights by creating synthetic lidar scans from observations
    and the map, then comparing them.
    """
    # Combine all map landmarks for efficient ray-tracing
    map_all = np.vstack([map_poles, map_trunks])
    map_classes = np.array([2] * len(map_poles) + [4] * len(map_trunks))

    # Define the angles for our synthetic lidar scan
    lidar_fov_rad = np.deg2rad(LIDAR_FOV_DEG)
    ray_angles = np.linspace(-lidar_fov_rad / 2.0, lidar_fov_rad / 2.0, LIDAR_NUM_RAYS)
    
    # Combine actual sensor observations
    obs_all = []
    obs_classes = []
    if bev_poles_obs.size > 0:
        obs_all.extend(bev_poles_obs)
        obs_classes.extend([2] * len(bev_poles_obs))
    if bev_trunks_obs.size > 0:
        obs_all.extend(bev_trunks_obs)
        obs_classes.extend([4] * len(bev_trunks_obs))

    if not obs_all: # If no observations, fallback to GPS only
        weights = np.zeros(len(particles))
        for i, (px, py, _) in enumerate(particles):
            if gps_xy is not None:
                gps_dist = np.linalg.norm(np.array([px, py]) - gps_xy)
                weights[i] = np.exp(-(gps_dist**2) / (2 * gps_sigma**2))
            else:
                weights[i] = 1.0
        return weights / np.sum(weights)

    obs_all = np.array(obs_all)
    obs_classes = np.array(obs_classes)

    weights = np.zeros(len(particles))

    # --- Main loop through each particle ---
    for i, (px, py, p_theta) in enumerate(particles):
        log_likelihood = 0.0

        # === 1. Generate the EXPECTED scan from the MAP for the current particle pose ===
        expected_scan = []
        for angle in ray_angles:
            ray_dir_world = np.array([np.cos(p_theta + angle), np.sin(p_theta + angle)])
            
            # Find all map points within sensor range
            dists_from_particle = np.linalg.norm(map_all - (px, py), axis=1)
            nearby_map_indices = np.where(dists_from_particle < SENSOR_RANGE)[0]

            closest_hit_range = SENSOR_RANGE
            closest_hit_class = -1 # -1 means no hit

            # Simple ray-casting: check for closest landmark near the ray
            for idx in nearby_map_indices:
                p_to_lm = map_all[idx] - (px, py)
                dist_to_lm = np.linalg.norm(p_to_lm)
                # Project landmark onto ray to see if it's a candidate
                proj = np.dot(p_to_lm, ray_dir_world)
                if proj > 0: # Must be in front
                    perp_dist = np.linalg.norm(p_to_lm - proj * ray_dir_world)
                    if perp_dist < 0.5 and dist_to_lm < closest_hit_range: # 0.5m tolerance
                        closest_hit_range = dist_to_lm
                        closest_hit_class = map_classes[idx]
            
            expected_scan.append({'range': closest_hit_range, 'class': closest_hit_class})

        # === 2. Generate the OBSERVED scan from the SENSOR data ===
        # Sensor data is already in the particle's local frame
        # Split observations into left and right of the particle's facing direction
        obs_x, obs_z = obs_all[:, 0], obs_all[:, 1]
        
        # Note: In our setup, local +x is LEFT, -x is RIGHT
        left_mask = obs_x > 0
        right_mask = obs_x <= 0

        # Sort points on each side by forward distance (z) to create "walls"
        left_indices = np.argsort(obs_all[left_mask, 1])
        right_indices = np.argsort(obs_all[right_mask, 1])
        
        left_wall_pts = obs_all[left_mask][left_indices]
        left_wall_cls = obs_classes[left_mask][left_indices]
        right_wall_pts = obs_all[right_mask][right_indices]
        right_wall_cls = obs_classes[right_mask][right_indices]

        observed_scan = []
        for angle in ray_angles:
            ray_dir_local = np.array([np.sin(angle), np.cos(angle)]) # sin for x, cos for z
            
            wall_pts = left_wall_pts if angle > 0 else right_wall_pts
            wall_cls = left_wall_cls if angle > 0 else right_wall_cls
            
            hit_range = SENSOR_RANGE
            hit_class = -1

            if len(wall_pts) > 1:
                for j in range(len(wall_pts) - 1):
                    p1, p2 = wall_pts[j], wall_pts[j+1]
                    intersection = get_ray_segment_intersection(np.array([0,0]), ray_dir_local, p1, p2)
                    if intersection is not None:
                        dist = np.linalg.norm(intersection)
                        if dist < hit_range:
                            hit_range = dist
                            hit_class = wall_cls[j] # Class of the segment's start point
                            break # Found first intersection on this wall
            
            observed_scan.append({'range': hit_range, 'class': hit_class})

        # === 3. Compare the scans to calculate likelihood ===
        scan_log_prob = 0.0
        for j in range(LIDAR_NUM_RAYS):
            obs_r = observed_scan[j]['range']
            exp_r = expected_scan[j]['range']
            obs_c = observed_scan[j]['class']
            exp_c = expected_scan[j]['class']

            # A) Range likelihood (Gaussian error)
            range_diff = obs_r - exp_r
            prob_range = -(range_diff**2) / (2 * LIDAR_RANGE_STD**2)
            
            # B) Semantic likelihood
            prob_semantic = 0.0
            if obs_c != -1 and exp_c != -1: # Only compare if both rays hit something
                if obs_c == exp_c:
                    # Apply class-specific weight and semantic bonus
                    prob_semantic = CLASS_WEIGHTS.get(obs_c, 1.0) * SEMANTIC_MATCH_BONUS
                else:
                    # Penalty for mismatch
                    prob_semantic = -SEMANTIC_MATCH_BONUS

            scan_log_prob += (prob_range + prob_semantic)
        
        log_likelihood += scan_log_prob / LIDAR_NUM_RAYS # Average probability over all rays

        # === 4. Boost by GPS distance ===
        if gps_xy is not None:
            gps_dist = np.linalg.norm(np.array([px, py]) - gps_xy)
            log_likelihood -= (gps_dist**2) / (2 * gps_sigma**2)

        weights[i] = np.exp(log_likelihood)

    # --- Normalize all particle weights ---
    total_weight = np.sum(weights)
    if total_weight > 0:
        weights /= total_weight
    else:
        weights = np.ones(len(particles)) / len(particles)

    return weights

def visualize_particles(poles_coords, trunks_coords, particles, frame_idx, output_dir, trajectory, gps_trajectory, rgb_overlay=None):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_title(f"Particles - Frame {frame_idx:04d}")

    # Landmark bounds
    landmark_x = np.concatenate([poles_coords[:, 0], trunks_coords[:, 0]])
    landmark_y = np.concatenate([poles_coords[:, 1], trunks_coords[:, 1]])
    x_min_lmk, x_max_lmk = landmark_x.min(), landmark_x.max()
    y_min_lmk, y_max_lmk = landmark_y.min(), landmark_y.max()

    # Other visual element bounds
    all_x = np.concatenate([
        particles[:, 0],
        np.array(trajectory)[:, 0] if trajectory else [],
        np.array(gps_trajectory)[:, 0] if gps_trajectory else []
    ])
    all_y = np.concatenate([
        particles[:, 1],
        np.array(trajectory)[:, 1] if trajectory else [],
        np.array(gps_trajectory)[:, 1] if gps_trajectory else []
    ])
    x_min_vis, x_max_vis = all_x.min(), all_x.max()
    y_min_vis, y_max_vis = all_y.min(), all_y.max()

    padding = 1.0
    x_min = min(x_min_lmk, x_min_vis) - padding
    x_max = max(x_max_lmk, x_max_vis) + padding
    y_min = min(y_min_lmk, y_min_vis) - padding
    y_max = max(y_max_lmk, y_max_vis) + padding
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # Plot landmarks
    ax.scatter(poles_coords[:, 0], poles_coords[:, 1], c='blue', s=10, label='Poles')
    ax.scatter(trunks_coords[:, 0], trunks_coords[:, 1], c='green', s=10, label='Trunks')

    # Plot particles with heading arrows
    px, py, pt = particles[:, 0], particles[:, 1], particles[:, 2]
    arrow_length = 1.0
    dx = arrow_length * np.cos(pt)
    dy = arrow_length * np.sin(pt)
    ax.quiver(px, py, dx, dy, angles='xy', scale_units='xy', scale=1, color='red', width=0.005, label='Particles')

    # Plot ground truth trajectory
    if len(gps_trajectory) > 1:
        gps_traj = np.array(gps_trajectory)
        ax.plot(gps_traj[:, 0], gps_traj[:, 1], 'b-', linewidth=1.5, label="Ground Truth")

    ax.legend()
    ax.grid(True)
    ax.set_aspect('equal')

    # Save visualization
    vis_path = os.path.join(output_dir, f"particles/frame_{frame_idx:04d}.jpg")
    os.makedirs(os.path.dirname(vis_path), exist_ok=True)
    plt.savefig(vis_path.replace('.jpg', '_plot.jpg'), bbox_inches='tight')
    plt.close()

    plot_img = cv2.imread(vis_path.replace('.jpg', '_plot.jpg'))

    if rgb_overlay is not None:
        rgb_resized = cv2.resize(rgb_overlay, (plot_img.shape[1], plot_img.shape[0]))
        combo = np.hstack((rgb_resized, plot_img))
        cv2.imwrite(vis_path, combo)
        os.remove(vis_path.replace('.jpg', '_plot.jpg'))
    else:
        os.rename(vis_path.replace('.jpg', '_plot.jpg'), vis_path)

def visualize_particle_overlap2(frame_idx, overlay, particle, bev_poles_obs, bev_trunks_obs, poles_coords, trunks_coords, output_dir, sensor_range=5.0):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_title(f"Particle Observation Overlap - Frame {frame_idx:04d}")

    px, py, theta = particle

    # Transform observations to map frame
    obs_poles_world = []
    for ox, oz in bev_poles_obs:
        mx = px + np.cos(theta) * oz - np.sin(theta) * ox
        my = py + np.sin(theta) * oz + np.cos(theta) * ox
        obs_poles_world.append((mx, my))

    obs_trunks_world = []
    for ox, oz in bev_trunks_obs:
        mx = px + np.cos(theta) * oz - np.sin(theta) * ox
        my = py + np.sin(theta) * oz + np.cos(theta) * ox
        obs_trunks_world.append((mx, my))

    obs_poles_world = np.array(obs_poles_world)
    obs_trunks_world = np.array(obs_trunks_world)

    # Plot observations
    if len(obs_poles_world) > 0:
        ax.scatter(obs_poles_world[:, 0], obs_poles_world[:, 1], c='blue', s=50, label='Observed Poles')
    if len(obs_trunks_world) > 0:
        ax.scatter(obs_trunks_world[:, 0], obs_trunks_world[:, 1], c='green', s=50, label='Observed Trunks')

    # Plot nearby landmarks (within sensor range)
    def filter_nearby(landmarks):
        dists = np.linalg.norm(landmarks - np.array([px, py]), axis=1)
        return landmarks[dists < 20*sensor_range]

    nearby_poles = filter_nearby(poles_coords)
    nearby_trunks = filter_nearby(trunks_coords)

    if len(nearby_poles) > 0:
        ax.scatter(nearby_poles[:, 0], nearby_poles[:, 1], c='blue', s=30, marker='x', label='Map Poles')
    if len(nearby_trunks) > 0:
        ax.scatter(nearby_trunks[:, 0], nearby_trunks[:, 1], c='green', s=30, marker='x', label='Map Trunks')

    # Plot particle
    arrow_length = 1.0  # Same scale as in visualize_particles
    dx = arrow_length * np.cos(theta)
    dy = arrow_length * np.sin(theta)
    ax.quiver(px, py, dx, dy, angles='xy', scale_units='xy', scale=1,
              color='red', width=0.005, label='Particle Pose')

    ax.set_xlim(px - 2*sensor_range, px + 2*sensor_range)
    ax.set_ylim(py - 2*sensor_range, py + 2*sensor_range)
    ax.set_aspect('equal')
    ax.grid(True)
    ax.legend()

    # Save image
    particle_dir = os.path.join(output_dir, "particle_overlap")
    os.makedirs(particle_dir, exist_ok=True)
    right_img_path = os.path.join(particle_dir, f"frame_{frame_idx:04d}_right.jpg")
    left_img_path = os.path.join(particle_dir, f"frame_{frame_idx:04d}_left.jpg")
    final_img_path = os.path.join(particle_dir, f"frame_{frame_idx:04d}.jpg")
    plt.savefig(right_img_path, bbox_inches='tight')
    plt.close()

    # Combine with RGB overlay
    right_img = cv2.imread(right_img_path)
    left_img = cv2.resize(overlay, (right_img.shape[1], right_img.shape[0]))
    combined = np.hstack((left_img, right_img))
    cv2.imwrite(final_img_path, combined)
    os.remove(right_img_path)

def _plot_wall_segments(ax, points, classes, particle_pose, linestyle, label_prefix=""):
    """Helper to sort points relative to a pose and plot them as connected walls."""
    if len(points) == 0:
        return

    px, py, theta = particle_pose
    
    # --- Transform points to particle's local frame to determine left/right and sort ---
    # This is for sorting/grouping purposes only; plotting is done in world coordinates.
    dx = points[:, 0] - px
    dy = points[:, 1] - py
    local_x = -dx * np.sin(theta) + dy * np.cos(theta)
    local_z = dx * np.cos(theta) + dy * np.sin(theta)

    # Split into left and right walls
    # Note: local +x is LEFT, -x is RIGHT
    left_indices = np.where(local_x > 0)[0]
    right_indices = np.where(local_x <= 0)[0]

    # Sort each wall by forward distance (z)
    sorted_left_indices = left_indices[np.argsort(local_z[left_indices])]
    sorted_right_indices = right_indices[np.argsort(local_z[right_indices])]

    # --- Plot the walls ---
    # We'll use these to ensure we only label each line type once
    labeled = {'pole': False, 'trunk': False} 
    
    for wall_indices in [sorted_left_indices, sorted_right_indices]:
        if len(wall_indices) < 2:
            continue
        
        wall_points = points[wall_indices]
        wall_classes = classes[wall_indices]

        for i in range(len(wall_points) - 1):
            p1 = wall_points[i]
            p2 = wall_points[i+1]
            start_class = wall_classes[i]

            color = 'blue' if start_class == 2 else 'green'
            class_name = 'pole' if start_class == 2 else 'trunk'
            
            label = None
            if not labeled[class_name]:
                label = f"{label_prefix} {class_name.capitalize()} Wall"
                labeled[class_name] = True

            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, linestyle=linestyle, linewidth=2, label=label)

def visualize_particle_overlap(frame_idx, overlay, particle, bev_poles_obs, bev_trunks_obs, poles_coords, trunks_coords, output_dir, sensor_range=5.0):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_title(f"Particle Observation Overlap - Frame {frame_idx:04d}")

    px, py, theta = particle

    # === 1. Process and Plot OBSERVED Walls (Dashed Lines) ===
    obs_all_local = []
    obs_classes = []
    if bev_poles_obs.size > 0:
        obs_all_local.extend(bev_poles_obs)
        obs_classes.extend([2] * len(bev_poles_obs))
    if bev_trunks_obs.size > 0:
        obs_all_local.extend(bev_trunks_obs)
        obs_classes.extend([4] * len(bev_trunks_obs))

    if obs_all_local:
        obs_all_local = np.array(obs_all_local)
        obs_classes = np.array(obs_classes)

        # Transform local observations to world frame for plotting
        obs_x_local, obs_z_local = obs_all_local[:, 0], obs_all_local[:, 1]
        obs_x_world = px + np.cos(theta) * obs_z_local - np.sin(theta) * obs_x_local
        obs_y_world = py + np.sin(theta) * obs_z_local + np.cos(theta) * obs_x_local
        obs_all_world = np.vstack([obs_x_world, obs_y_world]).T
        
        # Plot individual points and the wall
        ax.scatter(obs_all_world[obs_classes==2, 0], obs_all_world[obs_classes==2, 1], c='blue', s=50, alpha=0.6)
        ax.scatter(obs_all_world[obs_classes==4, 0], obs_all_world[obs_classes==4, 1], c='green', s=50, alpha=0.6)
        _plot_wall_segments(ax, obs_all_world, obs_classes, particle, linestyle='--', label_prefix='Observed')

    # === 2. Process and Plot MAP Walls (Dotted Lines) ===
    map_poles_nearby = poles_coords[np.linalg.norm(poles_coords - (px, py), axis=1) < sensor_range]
    map_trunks_nearby = trunks_coords[np.linalg.norm(trunks_coords - (px, py), axis=1) < sensor_range]
    
    map_all_world = []
    map_classes = []
    if len(map_poles_nearby) > 0:
        map_all_world.extend(map_poles_nearby)
        map_classes.extend([2] * len(map_poles_nearby))
    if len(map_trunks_nearby) > 0:
        map_all_world.extend(map_trunks_nearby)
        map_classes.extend([4] * len(map_trunks_nearby))

    if map_all_world:
        map_all_world = np.array(map_all_world)
        map_classes = np.array(map_classes)

        # Plot individual points and the wall
        ax.scatter(map_all_world[map_classes==2, 0], map_all_world[map_classes==2, 1], c='blue', s=30, marker='x')
        ax.scatter(map_all_world[map_classes==4, 0], map_all_world[map_classes==4, 1], c='green', s=30, marker='x')
        _plot_wall_segments(ax, map_all_world, map_classes, particle, linestyle=':', label_prefix='Map')

    # === 3. Plot Particle Pose and Finalize Chart ===
    arrow_length = 0.5
    dx = arrow_length * np.cos(theta)
    dy = arrow_length * np.sin(theta)
    ax.quiver(px, py, dx, dy, angles='xy', scale_units='xy', scale=1,
              color='red', width=0.005, label='Particle Pose')

    ax.set_xlim(px - sensor_range, px + sensor_range)
    ax.set_ylim(py - sensor_range, py + sensor_range)
    ax.set_aspect('equal')
    ax.grid(True)
    ax.legend()

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

# ---------- MAIN ----------
def process_data_with_localization(csv_data_path, rgb_dir, depth_dir, output_folder="amcl_output"):
    os.makedirs(os.path.join(output_folder, "particles"), exist_ok=True)
    df_gps = load_csv_with_utm(csv_data_path)
    poles_coords, trunks_coords, gdf, center = load_landmarks_from_geojson(geojson_path)

    """ Initialize particles based on landmarks extent
    all_coords = np.vstack([poles_coords, trunks_coords])
    extent = [(all_coords[:, 0].min(), all_coords[:, 1].min(), -np.pi),
              (all_coords[:, 0].max()+5, all_coords[:, 1].max()+5, np.pi)]
    particles = initialize_particles(PARTICLE_COUNT, extent=extent)
    #"""

    #""" Initial GPS-based pose (centered coordinates, heading assumed 0)
    first_gps_row = df_gps.iloc[0]
    init_x = first_gps_row["utm_easting"] - center[0]
    init_y = first_gps_row["utm_northing"] - center[1]
    init_theta = INIT_HEADING  # Or set based on GPS heading if available
    particles = initialize_particles_around_pose(
        center_pose=(init_x, init_y, init_theta),
        std_dev=(PARTICLE_STD, PARTICLE_STD, np.deg2rad(360)),  # Tune based on expected GPS uncertainty
        count=PARTICLE_COUNT
    )
    #"""

    trajectory = []
    gps_trajectory = []
    prev_gps_x, prev_gps_y, prev_heading = None, None, None

    # Main loop over the CSV file rows
    pbar = tqdm(df_gps.iterrows(), total=df_gps.shape[0], desc="Processing frames")
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
        overlay = color_img.copy()  # Always start fresh
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

        # Get GPS data for current frame
        gps_x = row["utm_easting"] - center[0]
        gps_y = row["utm_northing"] - center[1]

        # --- Motion Update ---
        if prev_gps_x is not None and prev_gps_y is not None and prev_heading is not None:
            dx = gps_x - prev_gps_x
            dy = gps_y - prev_gps_y
            delta_distance = np.sqrt(dx ** 2 + dy ** 2)
            heading_now = np.arctan2(dy, dx)
            delta_theta = heading_now - prev_heading
            delta_theta = (delta_theta + np.pi) % (2 * np.pi) - np.pi # Normalize angle
            particles = motion_update(particles, delta_distance, delta_theta)
        else:
            # First frame, no motion update. Set initial heading.
            heading_now = INIT_HEADING


        prev_gps_x, prev_gps_y, prev_heading = gps_x, gps_y, heading_now

        weights = measurement_likelihood(poles_coords, trunks_coords, bev_poles_obs, bev_trunks_obs, particles, (gps_x - 0.5 * np.cos(heading_now), gps_y - 0.5 * np.sin(heading_now)))
        # It's possible for all weights to be zero if no particles are plausible.
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
                poles_coords, trunks_coords,
                output_folder)
        #"""

        
        est_pose = np.average(particles, axis=0, weights=weights)
        trajectory.append(est_pose[:2])
        gps_trajectory.append([gps_x, gps_y])
        particles = adaptive_resample(particles, weights)#, (gps_x - 0.5 * np.cos(heading_now), gps_y - 0.5 * np.sin(heading_now)))

        visualize_particles(
            poles_coords, trunks_coords,
            particles, frame_idx,
            output_folder, trajectory, gps_trajectory,
            rgb_overlay=overlay  # overlay = your RGB + YOLO mask image
        )

if __name__ == "__main__":
    # Call the main processing function with the configured paths
    process_data_with_localization(
        csv_data_path=CSV_DATA_PATH,
        rgb_dir=RGB_DIR,
        depth_dir=DEPTH_DIR,
        output_folder="amcl_output"
    )
    print("[INFO] Finished processing all frames from the CSV file.")