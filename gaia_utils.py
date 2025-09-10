import os
import cv2
import numpy as np
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

# ---------- CONFIG ----------
FRAME_STRIDE = 9
device = 'cuda' if torch.cuda.is_available() else 'cpu'
yolo = YOLO("./models/yolo.pt").to(device)
CLASS_IDS = [2, 4]
CLASS_NAMES = {2: 'poles', 4: 'trunks'}
BEV_SIZE = (1000, 1000)
BEV_SCALE = 100
PARTICLE_COUNT = 1000
PARTICLE_STD = 0.01
SYNC_THRESHOLD = 1
geojson_path = "data/riseholme_poles_trunk.geojson"
llh_path = "data/reachRS2plu_solution_202408021257.LLH"

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

def load_llh_with_utm(llh_file, noise_std=0.5):
    with open(llh_file, "r") as f:
        lines = f.readlines()
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
    df["timestamp"] = df["timestamp"] - timedelta(seconds=17)
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


def motion_update(particles, delta_dist=0.1, delta_theta_std=0.05):
    noise = np.random.normal(0, PARTICLE_STD, particles.shape)
    particles[:, 0] += delta_dist * np.cos(particles[:, 2]) + noise[:, 0]
    particles[:, 1] += delta_dist * np.sin(particles[:, 2]) + noise[:, 1]
    particles[:, 2] += np.random.normal(0, delta_theta_std, size=particles.shape[0])
    return particles

def effective_sample_size(weights):
    return 1.0 / np.sum(weights ** 2)

def adaptive_resample(particles, weights, ess_threshold=0.5):
    ess = effective_sample_size(weights)
    if ess < ess_threshold * len(particles):
        idx = np.random.choice(len(particles), size=len(particles), p=weights)
        return particles[idx]
    return particles

def measurement_likelihood(map_poles, map_trunks, bev_poles, bev_trunks, particles):
    scores_poles, scores_trunks = [], []
    for x, z, _ in particles:
        ox = int(BEV_SIZE[1] // 2 + x * BEV_SCALE)
        oz = int(BEV_SIZE[0] - z * BEV_SCALE)
        if 0 <= ox < BEV_SIZE[1] and 0 <= oz < BEV_SIZE[0]:
            mp = lambda m: m[max(0, oz-20):oz+20, max(0, ox-20):ox+20]
            score_pole = np.sum(mp(map_poles) == 255)
            score_trunk = np.sum(mp(map_trunks) == 255)
        else:
            score_pole, score_trunk = 0, 0
        scores_poles.append(score_pole)
        scores_trunks.append(score_trunk)
    weights = 0.6 * np.exp(np.array(scores_poles) / 10.0) + 0.4 * np.exp(np.array(scores_trunks) / 10.0)
    return weights / np.sum(weights)

def render_map_to_bev(points):
    bev = np.zeros(BEV_SIZE, dtype=np.uint8)
    origin = (BEV_SIZE[1] // 2, BEV_SIZE[0])
    for x, z in points:
        mx = int(origin[0] + x * BEV_SCALE)
        mz = int(origin[1] - z * BEV_SCALE)
        if 0 <= mx < BEV_SIZE[1] and 0 <= mz < BEV_SIZE[0]:
            bev[mz, mx] = 255
    return bev

def visualize_particles(poles_coords, trunks_coords, particles, frame_idx, output_dir, trajectory, gps_trajectory, rgb_vis=None):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_title(f"Particles - Frame {frame_idx:04d}")
    
    # Combine all coordinates to compute plot bounds
    all_x = np.concatenate([particles[:, 0], np.array(trajectory)[:, 0] if trajectory else [], np.array(gps_trajectory)[:, 0] if gps_trajectory else []])
    all_y = np.concatenate([particles[:, 1], np.array(trajectory)[:, 1] if trajectory else [], np.array(gps_trajectory)[:, 1] if gps_trajectory else []])
    padding = 1.0
    if len(all_x) > 0 and len(all_y) > 0:
        x_min, x_max = all_x.min() - padding, all_x.max() + padding
        y_min, y_max = all_y.min() - padding, all_y.max() + padding
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

    # Plot landmarks
    ax.scatter(poles_coords[:, 0], poles_coords[:, 1], c='blue', s=10, label='Poles')
    ax.scatter(trunks_coords[:, 0], trunks_coords[:, 1], c='green', s=10, label='Trunks')

    # Plot particles
    ax.scatter(particles[:, 0], particles[:, 1], c='red', s=5, label='Particles', alpha=0.6)

    # Plot trajectories
    if len(trajectory) > 1:
        traj = np.array(trajectory)
        ax.plot(traj[:, 0], traj[:, 1], 'g--', linewidth=1.5, label="Estimated")

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
    if rgb_vis is not None:
        rgb_resized = cv2.resize(rgb_vis, (plot_img.shape[1], plot_img.shape[0]))
        combo = np.hstack((rgb_resized, plot_img))
        cv2.imwrite(vis_path, combo)
        os.remove(vis_path.replace('.jpg', '_plot.jpg'))
    else:
        os.rename(vis_path.replace('.jpg', '_plot.jpg'), vis_path)


# ---------- MAIN ----------
def process_rosbag_with_localization(bag_path, output_folder="amcl_output"):
    os.makedirs(os.path.join(output_folder, "particles"), exist_ok=True)
    df_llh = load_llh_with_utm(llh_path)
    poles_coords, trunks_coords, gdf, center = load_landmarks_from_geojson(geojson_path)
    map_bev_poles = render_map_to_bev(poles_coords)
    map_bev_trunks = render_map_to_bev(trunks_coords)
    all_coords = np.vstack([poles_coords, trunks_coords])
    extent = [(all_coords[:, 0].min(), all_coords[:, 1].min(), -np.pi),
              (all_coords[:, 0].max(), all_coords[:, 1].max(), np.pi)]
    particles = initialize_particles(PARTICLE_COUNT, extent=extent)
    trajectory = []
    gps_trajectory = []

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device_from_file(bag_path, repeat_playback=False)
    config.enable_stream(rs.stream.color)
    config.enable_stream(rs.stream.depth)
    profile = pipeline.start(config)
    playback = profile.get_device().as_playback()
    playback.set_real_time(False)
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

            color_img = np.asanyarray(color_frame.get_data())
            ros_time = datetime.fromtimestamp(frames.get_timestamp() / 1000.0, tz=timezone.utc)

            gps_row = df_llh[df_llh["timestamp"] <= ros_time]
            if gps_row.empty or (ros_time - gps_row.iloc[-1]["timestamp"]).total_seconds() > SYNC_THRESHOLD:
                frame_idx += 1
                continue

            gps_row = gps_row.iloc[-1]
            gps_x = gps_row["utm_easting"] - center[0]
            gps_y = gps_row["utm_northing"] - center[1]

            particles = motion_update(particles)
            dummy_bev = np.zeros_like(map_bev_poles)
            weights = measurement_likelihood(map_bev_poles, map_bev_trunks, dummy_bev, dummy_bev, particles)
            weights /= np.sum(weights)
            est_pose = np.average(particles, axis=0, weights=weights)
            trajectory.append(est_pose[:2])
            gps_trajectory.append([gps_x, gps_y])
            particles = adaptive_resample(particles, weights)

            visualize_particles(poles_coords, trunks_coords, particles, frame_idx, output_folder, trajectory, gps_trajectory, color_img)
            frame_idx += 1
            pbar.update(1)
    except RuntimeError:
        print("[INFO] Finished ROS bag.")
    finally:
        pipeline.stop()

if __name__ == "__main__":
    process_rosbag_with_localization("data/ground/row_1_to_6.bag")