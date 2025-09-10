import os
import cv2
import numpy as np
import csv
import argparse
from tqdm import tqdm
import torch
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from pyproj import Transformer


# ---------- CONFIG ----------
FRAME_STRIDE = 1
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
PARTICLE_COUNT = 100
PARTICLE_STD = 2.0
ANGLE_STD = np.deg2rad(10)
INIT_HEADING = np.deg2rad(-80)
SENSOR_RANGE = 5.0 # Max range for LIDAR scan
geojson_path = "data/riseholme_poles_trunk.geojson"
# Radii for map objects
POLE_RADIUS = 0.06 / 2  # 6cm diameter
TRUNK_RADIUS = 0.04 / 2 # 4cm diameter

# Paths for folder-based processing
DATA_PATH = "data/2025/1/"
CSV_DATA_PATH = DATA_PATH + "data.csv"

# Camera Intrinsics
class Intrinsics:
    def __init__(self):
        self.width_depth = 848
        self.height_depth = 480
        self.ppx = 426.27 # principal point x (in depth frame)
        self.ppy = 241.27 # principal point y (in depth frame)
        self.fx = 419.92  # focal length x (in depth frame)
        self.fy = 419.92  # focal length y (in depth frame)

intr = Intrinsics()

# ---------- UTILS ----------
def quaternion_to_yaw(x, y, z, w):
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    return np.arctan2(t3, t4)

def yaw_to_quaternion(yaw):
    cy = np.cos(yaw * 0.5); sy = np.sin(yaw * 0.5)
    return 0, 0, sy, cy # Simplified for yaw-only rotation

def save_tum_trajectory(trajectory_data, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write("# timestamp tx ty tz qx qy qz qw\n")
        for timestamp, x, y, theta in trajectory_data:
            qx, qy, qz, qw = yaw_to_quaternion(theta)
            f.write(f"{timestamp} {x} {y} 0.0 {qx} {qy} {qz} {qw}\n")
    print(f"[INFO] Trajectory saved to {output_path}")

def load_csv_with_utm(csv_path, noise_std=PARTICLE_STD):
    df = pd.read_csv(csv_path)
    transformer = Transformer.from_crs("epsg:4326", "epsg:32630", always_xy=True)
    df["utm_easting"], df["utm_northing"] = transformer.transform(df["longitude"].values, df["latitude"].values)
    df["utm_easting_noisy"] = df["utm_easting"] + np.random.normal(0, noise_std, len(df))
    df["utm_northing_noisy"] = df["utm_northing"] + np.random.normal(0, noise_std, len(df))
    return df

def initialize_particles_around_pose(center_pose, std_dev=(PARTICLE_STD, PARTICLE_STD, ANGLE_STD), count=PARTICLE_COUNT):
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

def adaptive_resample(particles, weights, ess_threshold=0.5):
    ess = 1.0 / np.sum(weights ** 2) if np.sum(weights**2) > 0 else 0
    if ess < ess_threshold * len(particles):
        idx = np.random.choice(len(particles), size=len(particles), p=weights)
        return particles[idx], np.full(len(particles), 1.0 / len(particles))
    return particles, weights


# ---------- NEW/MODIFIED LIDAR-BASED FUNCTIONS ----------

def load_landmarks_as_circles(path):
    """Loads landmarks from GeoJSON as a list of circle objects."""
    gdf = gpd.read_file(path)
    if not gdf.crs or not gdf.crs.is_projected:
        gdf = gdf.to_crs(gdf.estimate_utm_crs())

    center = gdf.geometry.union_all().centroid
    gdf['x_centered'] = gdf.geometry.x - center.x
    gdf['y_centered'] = gdf.geometry.y - center.y

    map_circles = []
    for _, row in gdf.iterrows():
        class_id = 2 if row['feature_type'] == 'row_post' else 4
        radius = POLE_RADIUS if class_id == 2 else TRUNK_RADIUS
        map_circles.append({
            'coords': np.array([row['x_centered'], row['y_centered']]),
            'radius': radius,
            'class_id': class_id
        })
    print(f"[INFO] Loaded {len(map_circles)} landmarks as circles.")
    return map_circles, np.array([center.x, center.y])

def depth_image_to_laserscan(
    depth_image, rgb_image, intrinsics, target_height,
    row_tolerance, depth_scale, camera_height, bev_radius,
    frame_idx, output_dir
):
    """Converts depth image to a 2D laser scan (list of 2D points)."""
    fx, fy = intrinsics["fx"], intrinsics["fy"]
    cx, cy = intrinsics["cx"], intrinsics["cy"]
    width, height = intrinsics["width"], intrinsics["height"]

    scan_points = []
    
    # Visualization setup
    scan_vis_dir = os.path.join(output_dir, "scan_visualization")
    os.makedirs(scan_vis_dir, exist_ok=True)
    scan_vis = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    scan_vis = cv2.applyColorMap(scan_vis, cv2.COLORMAP_JET)

    for u in range(width):
        for v in range(height):
            z = depth_image[v, u] * depth_scale
            if z == 0 or np.isnan(z) or z > bev_radius: continue
            y = (v - cy) * z / fy
            y_ground = camera_height + y
            if abs(y_ground - target_height) < row_tolerance:
                x = (u - cx) * z / fx
                scan_points.append((z, -x)) # Camera frame: X left, Z forward
                cv2.circle(scan_vis, (u, v), 1, (0, 255, 0), -1)
                break
    
    # Save visualization
    cv2.imwrite(os.path.join(scan_vis_dir, f"frame_{frame_idx:04d}.jpg"), scan_vis)
    
    return scan_points

def lidar_measurement_likelihood_circles(particles, scan_points_local, map_circles, gps_xy, gps_sigma, lidar_sigma):
    """Calculates particle weights based on a 2D laser scan and a map of circles."""
    num_particles = len(particles)
    weights = np.ones(num_particles)

    # If no scan points, rely purely on GPS
    if not scan_points_local:
        for i, (px, py, _) in enumerate(particles):
            gps_dist = np.linalg.norm(np.array([px, py]) - gps_xy)
            weights[i] = np.exp(-0.5 * (gps_dist / gps_sigma)**2)
        return weights / np.sum(weights) if np.sum(weights) > 0 else np.full(num_particles, 1.0/num_particles)

    scan_points_local = np.array(scan_points_local)
    map_coords = np.array([c['coords'] for c in map_circles])
    map_radii = np.array([c['radius'] for c in map_circles])

    for i, (px, py, p_theta) in enumerate(particles):
        # Transform scan points to world frame
        R = np.array([[np.cos(p_theta), -np.sin(p_theta)], [np.sin(p_theta), np.cos(p_theta)]])
        scan_points_world = np.dot(scan_points_local, R.T) + np.array([px, py])
        
        # --- Likelihood from laser scan (L2 distance to nearest circle edge) ---
        total_dist_error_sq = 0
        for pt in scan_points_world:
            # Calculate distance from the point to all circle centers
            dists_to_centers = np.linalg.norm(map_coords - pt, axis=1)
            # Find distance to the edge of each circle
            dists_to_edges = np.abs(dists_to_centers - map_radii)
            # The error is the distance to the *closest* circle edge
            min_dist = np.min(dists_to_edges)
            total_dist_error_sq += min_dist**2
        
        mean_sq_error = total_dist_error_sq / len(scan_points_world)
        log_semantic = -0.5 * (mean_sq_error / lidar_sigma**2)

        # --- Likelihood from GPS ---
        gps_dist = np.linalg.norm(np.array([px, py]) - gps_xy)
        log_gps = -0.5 * (gps_dist / gps_sigma)**2
        
        weights[i] = np.exp(log_semantic + log_gps)

    if np.sum(weights) == 0:
        return np.full(num_particles, 1.0 / num_particles)
    return weights / np.sum(weights)


def visualize_lidar_state_circles(map_circles, particles, frame_idx, output_dir, trajectory, gps_trajectory, estimated_pose, scan_points, rgb_overlay):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_title(f"LIDAR PF (Circles) - Frame {frame_idx:04d}")

    # 1. Plot Map Circles
    poles_plotted, trunks_plotted = False, False
    for landmark in map_circles:
        is_pole = landmark['class_id'] == 2
        color = 'blue' if is_pole else 'green'
        label = ""
        if is_pole and not poles_plotted: label = 'Poles'; poles_plotted = True
        if not is_pole and not trunks_plotted: label = 'Trunks'; trunks_plotted = True
        
        circle = plt.Circle(landmark['coords'], landmark['radius'], color=color, fill=False, label=label)
        ax.add_patch(circle)

    # 2. Plot Trajectories and Particles
    if len(gps_trajectory) > 1: ax.plot(np.array(gps_trajectory)[:, 0], np.array(gps_trajectory)[:, 1], 'k--', linewidth=1.5, label="GPS Path")
    if len(trajectory) > 1: ax.plot(np.array(trajectory)[:, 0], np.array(trajectory)[:, 1], 'r-', linewidth=2, label="Estimated Path")
    ax.quiver(particles[:, 0], particles[:, 1], np.cos(particles[:, 2])*0.5, np.sin(particles[:, 2])*0.5, 
              angles='xy', scale_units='xy', scale=1, color='red', width=0.005, alpha=0.6, label='Particles')

    # 3. Plot Observed Scan from Estimated Pose
    if scan_points:
        px, py, p_theta = estimated_pose
        R = np.array([[np.cos(p_theta), -np.sin(p_theta)], [np.sin(p_theta), np.cos(p_theta)]])
        scan_world = np.dot(np.array(scan_points), R.T) + np.array([px, py])
        ax.scatter(scan_world[:, 0], scan_world[:, 1], c='magenta', s=5, label='Observed Scan', zorder=10)

    # Finalize Plot
    ax.set_aspect('equal')
    ax.legend()
    ax.grid(True)
    
    # Set plot limits to zoom on the current action
    if estimated_pose is not None:
        px, py, _ = estimated_pose
        ax.set_xlim(px - 15, px + 15)
        ax.set_ylim(py - 15, py + 15)

    # Save visualization
    vis_dir = os.path.join(output_dir, "pf_frames")
    os.makedirs(vis_dir, exist_ok=True)
    plot_path = os.path.join(vis_dir, f"frame_{frame_idx:04d}_plot.png")
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close(fig)

    plot_img = cv2.imread(plot_path)
    if rgb_overlay is not None and plot_img is not None:
        h, w = plot_img.shape[:2]
        rgb_resized = cv2.resize(rgb_overlay, (w, int(w * rgb_overlay.shape[0] / rgb_overlay.shape[1])))
        final_h = min(h, rgb_resized.shape[0])
        combo = np.hstack((rgb_resized[:final_h, :], plot_img[:final_h, :]))
        cv2.imwrite(os.path.join(vis_dir, f"frame_{frame_idx:04d}.jpg"), combo)
        os.remove(plot_path)

# ---------- MAIN ----------
def process_data_with_lidar_localization(
    csv_data_path, rgb_dir, depth_dir, camera_height, target_height,
    row_tolerance, lidar_sigma, output_folder
):
    df_data = load_csv_with_utm(csv_data_path)
    map_circles, center = load_landmarks_as_circles(geojson_path)

    # Initial GPS-based pose
    first_row = df_data.iloc[0]
    init_x = first_row["utm_easting"] - center[0]
    init_y = first_row["utm_northing"] - center[1]
    particles = initialize_particles_around_pose(
        center_pose=(init_x, init_y, INIT_HEADING),
        std_dev=(PARTICLE_STD, PARTICLE_STD, np.deg2rad(360))
    )
    
    full_trajectory_data = []
    gps_trajectory = []
    prev_odom_pos_x, prev_odom_pos_y, prev_odom_yaw = None, None, None
    weights = np.full(PARTICLE_COUNT, 1.0 / PARTICLE_COUNT)

    intrinsics_dict = {
        "fx": intr.fx, "fy": intr.fy, "cx": intr.ppx, "cy": intr.ppy,
        "width": intr.width_depth, "height": intr.height_depth
    }

    pbar = tqdm(df_data.iterrows(), total=df_data.shape[0], desc="Processing (LIDAR PF Circles)")
    for frame_idx, row in pbar:
        if frame_idx % FRAME_STRIDE != 0: continue

        # --- Load images ---
        rgb_path = os.path.join(rgb_dir, row['rgb_image'])
        depth_path = os.path.join(depth_dir, row['depth_image'])
        if not (os.path.exists(rgb_path) and os.path.exists(depth_path)): continue
        color_img = cv2.imread(rgb_path)
        depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if color_img is None or depth_img is None: continue

        # --- Motion Update (from Odometry) ---
        current_odom_pos_x, current_odom_pos_y = row['odom_pos_x'], row['odom_pos_y']
        current_odom_yaw = quaternion_to_yaw(row['odom_orient_x'], row['odom_orient_y'], row['odom_orient_z'], row['odom_orient_w'])
        if prev_odom_pos_x is not None:
            dx = current_odom_pos_x - prev_odom_pos_x
            dy = current_odom_pos_y - prev_odom_pos_y
            d_dist = np.sqrt(dx**2 + dy**2)
            filtered_odom_yaw = 0.8 * current_odom_yaw + 0.2 * prev_odom_yaw
            d_theta = (filtered_odom_yaw - prev_odom_yaw + np.pi) % (2 * np.pi) - np.pi
            particles = motion_update(particles, d_dist, d_theta)
        prev_odom_pos_x, prev_odom_pos_y, prev_odom_yaw = current_odom_pos_x, current_odom_pos_y, current_odom_yaw

        # --- Measurement Update (from LIDAR scan) ---
        scan_points = depth_image_to_laserscan(
            depth_image=depth_img, rgb_image=color_img, intrinsics=intrinsics_dict,
            target_height=target_height, row_tolerance=row_tolerance, depth_scale=0.001,
            camera_height=camera_height, bev_radius=SENSOR_RANGE, frame_idx=frame_idx,
            output_dir=output_folder
        )
        
        gps_x_noisy, gps_y_noisy = (row["utm_easting_noisy"] - center[0]), (row["utm_northing_noisy"] - center[1])
        
        weights = lidar_measurement_likelihood_circles(
            particles, scan_points, map_circles, 
            gps_xy=(gps_x_noisy, gps_y_noisy), 
            gps_sigma=PARTICLE_STD, 
            lidar_sigma=lidar_sigma
        )
        
        # --- Resampling ---
        particles, weights = adaptive_resample(particles, weights)

        # --- State Estimation & Logging ---
        est_pose = np.average(particles, axis=0, weights=weights)
        full_trajectory_data.append((frame_idx, est_pose[0], est_pose[1], est_pose[2]))
        gps_trajectory.append([row["utm_easting"] - center[0], row["utm_northing"] - center[1]])
        
        # --- Visualization ---
        visualize_lidar_state_circles(
            map_circles, particles, frame_idx, output_folder, 
            [(t[1], t[2]) for t in full_trajectory_data], gps_trajectory,
            est_pose, scan_points, color_img
        )

    # --- Save Final Trajectory ---
    tum_output_path = os.path.join(output_folder, f"trajectory_lidar_pf_circles.tum")
    save_tum_trajectory(full_trajectory_data, tum_output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run 2D LIDAR-based Particle Filter with a circular object map.")
    parser.add_argument('--camera-height', type=float, default=0.0, help='Height of the camera from the ground in meters.')
    parser.add_argument('--target-height', type=float, default=0.9, help='The target height slice for creating the 2D scan.')
    parser.add_argument('--row-tolerance', type=float, default=0.05, help='Tolerance for the height slice.')
    parser.add_argument('--lidar-sigma', type=float, default=0.2, help='Standard deviation of the LIDAR measurement noise in meters.')
    
    args = parser.parse_args()

    print(f"[INFO] Running LIDAR Particle Filter with parameters: {args}")

    process_data_with_lidar_localization(
        csv_data_path=CSV_DATA_PATH,
        rgb_dir=DATA_PATH,
        depth_dir=DATA_PATH,
        camera_height=args.camera_height,
        target_height=args.target_height,
        row_tolerance=args.row_tolerance,
        lidar_sigma=args.lidar_sigma,
        output_folder="amcl_output/lidar_pf_circles_experiment"
    )
    print("[INFO] Finished processing all frames.")