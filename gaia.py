import os
import cv2
import numpy as np
import csv
import pyrealsense2 as rs
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
from cv_bridge import CvBridge

# ---------- CONFIG ----------
FRAME_STRIDE = 10
device = 'cuda' if torch.cuda.is_available() else 'cpu'
yolo = YOLO("./models/yolo.pt").to(device)
CLASS_IDS = [2, 4]
CLASS_NAMES = {2: 'poles', 4: 'trunks'}
BEV_SIZE = (1000, 1000)
BEV_SCALE = 100
PARTICLE_COUNT = 100
PARTICLE_STD = 2.0
ANGLE_STD = np.deg2rad(10)
INIT_HEADING = np.deg2rad(-80)
SYNC_THRESHOLD = 1
SYNC_LAG = 19
SENSOR_RANGE = 5.0
HORIZONTAL_FOV = np.deg2rad(87)
MAX_RAY_HALF_ANGLE = np.deg2rad(5)
geojson_path = "data/riseholme_poles_trunk.geojson"
llh_path = "data/reachRS2plu_solution_202408021257.LLH"
CSV_PATH=f'amcl_output/csv/likelihood_stats_{PARTICLE_COUNT}_{PARTICLE_STD}.csv'

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

def load_llh_with_utm(llh_file, noise_std=PARTICLE_STD):
    with open(llh_file, "r") as f:
        lines = f.readlines()

    #lines = lines[620:]
    data = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 4:
            try:
                timestamp = datetime.strptime(parts[0] + " " + parts[1], "%Y/%m/%d %H:%M:%S.%f").replace(tzinfo=timezone.utc)
                lat = float(parts[2])
                lon = float(parts[3])
                data.append((timestamp, lat, lon))
            except:
                continue
    df = pd.DataFrame(data, columns=["timestamp", "lat", "lon"])
    df["timestamp"] = df["timestamp"] - timedelta(seconds=SYNC_LAG)
    transformer = Transformer.from_crs("epsg:4326", "epsg:32630", always_xy=True)
    df["utm_easting"], df["utm_northing"] = transformer.transform(df["lon"].values, df["lat"].values)
    np.random.seed(42)
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


def motion_update(particles, delta_distance, delta_theta, noise_std=(PARTICLE_STD, PARTICLE_STD, ANGLE_STD)):
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

def measurement_likelihood2(
    map_poles,
    map_trunks,
    bev_poles_obs,
    bev_trunks_obs,
    particles,
    gps_xy=None,
    gps_sigma=PARTICLE_STD,
    csv_path=CSV_PATH
):
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

        # Spatial Layout Consistency
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
        log_total = log_penalty  # + log_structure

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

def measurement_likelihood(map_poles, map_trunks, bev_poles_obs, bev_trunks_obs, particles,
                           gps_xy=None, gps_sigma=PARTICLE_STD, csv_path=None):
    """
    Calculates particle weights using a dynamic, ray-based observation model.

    For each particle, this function iterates through each actual observation and:
    1. Dynamically calculates a "ray" half-angle for the observation based on the
       angular separation of its nearest neighbors, capped at a maximum value.
    2. Searches the map for landmarks of the same class that are close to this ray
       and within sensor range.
    3. Finds the map landmark closest to the particle along this ray.
    4. Calculates an error based on the range difference between the actual
       observation and the expected map landmark.
    5. Applies a penalty if no map landmark is found along the ray.
    6. Applies a final boost based on GPS proximity.
    """
    sigma = PARTICLE_STD
    weights = np.zeros(len(particles))

    # Combine actual observations and their classes
    actual_obs_all = []
    actual_obs_classes = []
    if bev_poles_obs.size > 0:
        actual_obs_all.extend(bev_poles_obs.tolist())
        actual_obs_classes.extend([2] * len(bev_poles_obs))
    if bev_trunks_obs.size > 0:
        actual_obs_all.extend(bev_trunks_obs.tolist())
        actual_obs_classes.extend([4] * len(bev_trunks_obs))

    # Combine all map landmarks for efficient filtering
    map_all = np.vstack([map_poles, map_trunks])
    map_classes = np.array([2] * len(map_poles) + [4] * len(map_trunks))

    # Pre-calculate observation angles and sort them
    num_obs = len(actual_obs_all)
    if num_obs > 0:
        actual_obs_points = np.array(actual_obs_all)
        actual_obs_angles = np.arctan2(actual_obs_points[:, 0], actual_obs_points[:, 1])
        sorted_indices = np.argsort(actual_obs_angles)
        
        # Re-order all observation data based on the sorted angles
        sorted_obs_points = actual_obs_points[sorted_indices]
        sorted_obs_classes = np.array(actual_obs_classes)[sorted_indices]
        sorted_obs_angles = actual_obs_angles[sorted_indices]

    # --- Main loop through each particle ---
    for i, (px, py, theta) in enumerate(particles):
        log_likelihood = 0.0

        # Fallback to GPS-only weighting if there are no observations
        if num_obs == 0:
            if gps_xy is not None:
                gps_dist = np.linalg.norm(np.array([px, py]) - gps_xy)
                log_likelihood -= (gps_dist**2) / (2 * gps_sigma**2)
            weights[i] = np.exp(log_likelihood)
            continue

        # --- Transform all map landmarks into the particle's local frame ONCE ---
        dx = map_all[:, 0] - px
        dy = map_all[:, 1] - py
        map_local_z = dx * np.cos(theta) + dy * np.sin(theta)
        map_local_x = -dx * np.sin(theta) + dy * np.cos(theta)
        map_in_local_frame = np.column_stack([map_local_x, map_local_z])
        map_angles_local = np.arctan2(map_in_local_frame[:, 0], map_in_local_frame[:, 1])


        # --- Iterate through each SORTED ACTUAL observation to validate it ---
        for j in range(num_obs):
            obs_point = sorted_obs_points[j]
            obs_class = sorted_obs_classes[j]
            actual_angle = sorted_obs_angles[j]
            actual_range = np.linalg.norm(obs_point)

            # --- Dynamically determine the ray half-angle ---
            if num_obs < 2:
                # If only one observation, use the maximum angle
                ray_half_angle = MAX_RAY_HALF_ANGLE
            else:
                # Get previous and next angles, wrapping around for the first/last elements
                prev_angle = sorted_obs_angles[j - 1]
                next_angle = sorted_obs_angles[(j + 1) % num_obs]
                
                # Calculate the total angular distance between neighbors
                # Handle the wrap-around case at -pi / +pi
                angular_separation = (next_angle - prev_angle + np.pi) % (2 * np.pi) - np.pi
                
                # The half-angle is half the separation, capped at the max value
                dynamic_half_angle = np.abs(angular_separation) / 2.0
                ray_half_angle = min(dynamic_half_angle, MAX_RAY_HALF_ANGLE)

            # --- Find map landmarks along the observation's dynamic "ray" ---
            class_mask = (map_classes == obs_class)
            
            angle_diff = np.abs(map_angles_local - actual_angle)
            angle_mask = (angle_diff < ray_half_angle) | (angle_diff > (2 * np.pi - ray_half_angle))

            map_ranges = np.linalg.norm(map_in_local_frame, axis=1)
            range_mask = (map_ranges < SENSOR_RANGE) & (map_in_local_frame[:, 1] > 0)

            candidate_mask = class_mask & angle_mask & range_mask
            map_candidates_on_ray = map_in_local_frame[candidate_mask]
            
            if len(map_candidates_on_ray) > 0:
                # Find the single closest map landmark along this ray
                candidate_ranges = np.linalg.norm(map_candidates_on_ray, axis=1)
                closest_candidate_idx = np.argmin(candidate_ranges)
                expected_range = candidate_ranges[closest_candidate_idx]

                range_error = np.abs(actual_range - expected_range)
                log_likelihood += -((range_error**2) / (2 * sigma**2))
            else:
                # Penalty: The particle saw something, but there was nothing on the map along that ray.
                log_likelihood -= 20.0

        # --- Apply GPS Proximity Boost at the end ---
        if gps_xy is not None:
            gps_dist = np.linalg.norm(np.array([px, py]) - gps_xy)
            log_likelihood += -((gps_dist**2) / (2 * gps_sigma**2))

        weights[i] = np.exp(log_likelihood)

    # --- Normalize all particle weights to sum to 1 ---
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

def visualize_particle_overlap(frame_idx, overlay, particle, bev_poles_obs, bev_trunks_obs, poles_coords, trunks_coords, output_dir, sensor_range=5.0):
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
        ax.scatter(obs_poles_world[:, 0], obs_poles_world[:, 1], c='blue', s=20, label='Observed Poles')
    if len(obs_trunks_world) > 0:
        ax.scatter(obs_trunks_world[:, 0], obs_trunks_world[:, 1], c='green', s=20, label='Observed Trunks')

    # Plot nearby landmarks (within sensor range)
    def filter_nearby(landmarks):
        dists = np.linalg.norm(landmarks - np.array([px, py]), axis=1)
        return landmarks[dists < sensor_range]

    nearby_poles = filter_nearby(poles_coords)
    nearby_trunks = filter_nearby(trunks_coords)

    if len(nearby_poles) > 0:
        ax.scatter(nearby_poles[:, 0], nearby_poles[:, 1], c='blue', s=10, marker='x', label='Map Poles')
    if len(nearby_trunks) > 0:
        ax.scatter(nearby_trunks[:, 0], nearby_trunks[:, 1], c='green', s=10, marker='x', label='Map Trunks')

    # Plot particle
    arrow_length = 1.0  # Same scale as in visualize_particles
    dx = arrow_length * np.cos(theta)
    dy = arrow_length * np.sin(theta)
    ax.quiver(px, py, dx, dy, angles='xy', scale_units='xy', scale=1,
              color='red', width=0.005, label='Particle Pose')

    ax.set_xlim(px - sensor_range, px + sensor_range)
    ax.set_ylim(py - sensor_range, py + sensor_range)
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

# ---------- MAIN ----------
def process_rosbag_with_localization(bag_path, output_folder="amcl_output"):
    os.makedirs(os.path.join(output_folder, "particles"), exist_ok=True)
    df_llh = load_llh_with_utm(llh_path)
    poles_coords, trunks_coords, gdf, center = load_landmarks_from_geojson(geojson_path)
    all_coords = np.vstack([poles_coords, trunks_coords])
    extent = [(all_coords[:, 0].min(), all_coords[:, 1].min(), -np.pi),
              (all_coords[:, 0].max()+5, all_coords[:, 1].max()+5, np.pi)]
    #particles = initialize_particles(PARTICLE_COUNT, extent=extent)
    #"""
    # Initial GPS-based pose (centered coordinates, heading assumed 0)
    first_gps_row = df_llh.iloc[0]
    init_x = first_gps_row["utm_easting"] - center[0]
    init_y = first_gps_row["utm_northing"] - center[1]
    init_theta = INIT_HEADING  # Or set based on GPS heading if available

    particles = initialize_particles_around_pose(
        center_pose=(init_x, init_y, init_theta),
        std_dev=(PARTICLE_STD, PARTICLE_STD, ANGLE_STD),  # Tune based on expected GPS uncertainty
        count=PARTICLE_COUNT
    )
    #"""
    trajectory = []
    gps_trajectory = []

    prev_gps_x, prev_gps_y, prev_heading = None, None, None

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device_from_file(bag_path, repeat_playback=False)
    config.enable_stream(rs.stream.color)
    config.enable_stream(rs.stream.depth)
    profile = pipeline.start(config)
    playback = profile.get_device().as_playback()
    playback.set_real_time(False)
    ##########################
    # --- Seek to a preset start time (in milliseconds from beginning of bag) ---
    start_time_seconds =0.0
    if start_time_seconds > 0:
        start_time_nanoseconds = int(start_time_seconds * 1_000_000_000) # Convert seconds to nanoseconds
        playback.seek(timedelta(microseconds=start_time_nanoseconds / 1000)) # Seek expects timedelta or nanoseconds as int for some versions
        print(f"[INFO] Seeked to {start_time_seconds} seconds into the ROS bag.")
    #############################
    align = rs.align(rs.stream.color)
    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

    frame_idx = 0
    try:
        pbar = tqdm(desc="Processing frames")
        while True:
            frames = pipeline.wait_for_frames()
            
            if frame_idx % FRAME_STRIDE != 0:
                frame_idx += 1
                continue

            aligned_frames = align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            if not color_frame:
                frame_idx += 1
                continue
            """
            if frame_idx == 0:
                first_gps_row = df_llh.iloc[620]
                init_x = first_gps_row["utm_easting"] - center[0]
                init_y = first_gps_row["utm_northing"] - center[1]
                init_theta = INIT_HEADING  # Or set based on GPS heading if available

                particles = initialize_particles_around_pose(
                    center_pose=(init_x, init_y, init_theta),
                    std_dev=(PARTICLE_STD, PARTICLE_STD, ANGLE_STD),  # Tune based on expected GPS uncertainty
                    count=PARTICLE_COUNT
                )
            """
            color_img = cv2.cvtColor(np.asanyarray(color_frame.get_data()), cv2.COLOR_RGB2BGR)
            depth_frame = aligned_frames.get_depth_frame()
            depth_img = np.asanyarray(depth_frame.get_data())
            ros_time = datetime.fromtimestamp(frames.get_timestamp() / 1000.0, tz=timezone.utc)

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
            #"""
            ##################################################################3            
                    # Draw contours for mask
                    overlay_color = (255, 0, 0) if class_id == 2 else (0, 255, 0)
                    mask_vis = (mask_resized_color > 0.5).astype(np.uint8) * 255
                    contours, _ = cv2.findContours(mask_vis, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(overlay, contours, -1, overlay_color, 2)

            else:
                # No masks found: reset overlay to raw image
                overlay = color_img.copy()
            #"""
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

            ##################################################################3
            #"""
            # Convert to numpy arrays
            bev_poles_obs = np.array(bev_poles_obs)
            bev_trunks_obs = np.array(bev_trunks_obs)
            gps_row = df_llh[df_llh["timestamp"] <= ros_time]
            if gps_row.empty or (ros_time - gps_row.iloc[-1]["timestamp"]).total_seconds() > SYNC_THRESHOLD:
                frame_idx += 1
                continue

            gps_row = gps_row.iloc[-1]
            gps_x = gps_row["utm_easting"] - center[0]
            gps_y = gps_row["utm_northing"] - center[1]

            if prev_gps_x is not None and prev_gps_y is not None and prev_heading is not None:
                dx = gps_x - prev_gps_x
                dy = gps_y - prev_gps_y
                delta_distance = np.sqrt(dx ** 2 + dy ** 2)
                heading_now = np.arctan2(dy, dx)
                delta_theta = heading_now - prev_heading
                delta_theta = (delta_theta + np.pi) % (2 * np.pi) - np.pi
                particles = motion_update(particles, delta_distance, delta_theta)
            else:
                particles = motion_update(particles, PARTICLE_STD, ANGLE_STD)
                heading_now = INIT_HEADING


            prev_gps_x, prev_gps_y, prev_heading = gps_x, gps_y, heading_now

            weights = measurement_likelihood(poles_coords, trunks_coords, bev_poles_obs, bev_trunks_obs, particles, (gps_x, gps_y))        
            weights /= np.sum(weights)

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
            particles = adaptive_resample(particles, weights)

            visualize_particles(
                poles_coords, trunks_coords,
                particles, frame_idx,
                output_folder, trajectory, gps_trajectory,
                rgb_overlay=overlay  # overlay = your RGB + YOLO mask image
            )

            frame_idx += 1
            pbar.update(1)
    except RuntimeError:
        print("[INFO] Finished ROS bag.")
    finally:
        pipeline.stop()

if __name__ == "__main__":
    process_rosbag_with_localization("data/ground/row_1_to_6.bag")