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
FRAME_STRIDE = 1
device = 'cuda' if torch.cuda.is_available() else 'cpu'
yolo = YOLO("./models/yolo.pt").to(device)
CLASS_IDS = [2, 4]
CLASS_NAMES = {2: 'poles', 4: 'trunks'}
BEV_SIZE = (1000, 1000)
BEV_SCALE = 100
PARTICLE_COUNT = 200
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
    
    # Shift timestamps back by 16 seconds
    df["timestamp"] = df["timestamp"] - timedelta(seconds=17)

    transformer = Transformer.from_crs("epsg:4326", "epsg:32630", always_xy=True)
    df["utm_easting"], df["utm_northing"] = transformer.transform(df["lon"].values, df["lat"].values)
    
    np.random.seed(42)
    df["utm_easting_noisy"] = df["utm_easting"] + np.random.normal(0, noise_std, len(df))
    df["utm_northing_noisy"] = df["utm_northing"] + np.random.normal(0, noise_std, len(df))
    
    return df

def find_latest_past_gps(df, target_time):
    past = df[df["timestamp"] <= target_time]
    if not past.empty:
        return past.iloc[-1:]  # most recent past entry
    else:
        return df.iloc[:1]  # fallback to first if none found


def initialize_particles(n, extent=None):
    if extent:
        low = np.array(extent[0])
        high = np.array(extent[1])
        return np.random.uniform(low=low, high=high, size=(n, 3))
    else:
        return np.zeros((n, 3))

def render_map_to_bev(points):
    bev = np.zeros(BEV_SIZE, dtype=np.uint8)
    origin = (BEV_SIZE[1] // 2, BEV_SIZE[0])
    for x, z in points:
        mx = int(origin[0] + x * BEV_SCALE)
        mz = int(origin[1] - z * BEV_SCALE)
        if 0 <= mx < BEV_SIZE[1] and 0 <= mz < BEV_SIZE[0]:
            bev[mz, mx] = 255
    return bev

def measurement_likelihood(map_poles, map_trunks, bev_poles, bev_trunks, particles):
    scores_poles, scores_trunks = [], []
    for x, z, _ in particles:
        ox = int(BEV_SIZE[1] // 2 + x * BEV_SCALE)
        oz = int(BEV_SIZE[0] - z * BEV_SCALE)
        if 0 <= ox < BEV_SIZE[1] and 0 <= oz < BEV_SIZE[0]:
            mp = lambda m: m[max(0, oz-20):oz+20, max(0, ox-20):ox+20]
            obs_pole_patch = mp(bev_poles)[:, :, 2]
            obs_trunk_patch = mp(bev_trunks)[:, :, 2]
            map_pole_patch = mp(map_poles)
            map_trunk_patch = mp(map_trunks)
            score_pole = np.sum((obs_pole_patch == 255) & (map_pole_patch == 255)) if obs_pole_patch.shape == map_pole_patch.shape else 0
            score_trunk = np.sum((obs_trunk_patch == 255) & (map_trunk_patch == 255)) if obs_trunk_patch.shape == map_trunk_patch.shape else 0
        else:
            score_pole, score_trunk = 0, 0
        scores_poles.append(score_pole)
        scores_trunks.append(score_trunk)
    weights = 0.6 * np.exp(np.array(scores_poles) / 10.0) + 0.4 * np.exp(np.array(scores_trunks) / 10.0)
    return weights / np.sum(weights)

def resample_particles(particles, weights):
    idx = np.random.choice(len(particles), size=len(particles), p=weights)
    return particles[idx]

def project_full_bev(depth, fx, fy, cx, cy):
    h, w = depth.shape
    bev = np.zeros((BEV_SIZE[0], BEV_SIZE[1], 3), dtype=np.uint8)
    origin = (BEV_SIZE[1] // 2, BEV_SIZE[0])
    ys, xs = np.where(depth > 0)
    for u, v in zip(xs, ys):
        z = depth[v, u]
        x = (u - cx) * z / fx
        bx = int(origin[0] + x * BEV_SCALE)
        bz = int(origin[1] - z * BEV_SCALE)
        if 0 <= bx < BEV_SIZE[1] and 0 <= bz < BEV_SIZE[0]:
            bev[bz, bx] = [255, 255, 255]
    return bev

def overlay_mask_on_bev(mask, depth, bev, fx, fy, cx, cy, color=(0, 0, 255)):
    ys, xs = np.where(mask)
    origin = (bev.shape[1] // 2, bev.shape[0])
    for u, v in zip(xs, ys):
        z = depth[v, u]
        if z == 0:
            continue
        x = (u - cx) * z / fx
        bx = int(origin[0] + x * BEV_SCALE)
        bz = int(origin[1] - z * BEV_SCALE)
        if 0 <= bx < bev.shape[1] and 0 <= bz < bev.shape[0]:
            bev[bz, bx] = color
    return bev

def visualize_particles(map_points, particles, frame_idx, output_dir, trajectory, rgb_vis=None):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_title(f"Particles - Frame {frame_idx:04d}")
    padding = 1.0
    x_min, x_max = map_points[:, 0].min() - padding, map_points[:, 0].max() + padding
    y_min, y_max = map_points[:, 1].min() - padding, map_points[:, 1].max() + padding
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.scatter(map_points[:, 0], map_points[:, 1], c='black', s=10, label='Landmarks')
    ax.scatter(particles[:, 0], particles[:, 1], c='red', s=5, label='Particles', alpha=0.6)
    if len(trajectory) > 1:
        traj = np.array(trajectory)
        ax.plot(traj[:, 0], traj[:, 1], 'g--', linewidth=1.5)
    ax.legend()
    ax.grid(True)
    ax.set_aspect('equal')
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
def process_rosbag_with_localization(bag_path, output_folder="geojson_visualisation_output"):
    os.makedirs(os.path.join(output_folder, "combined"), exist_ok=True)

    df_llh = load_llh_with_utm(llh_path)
    poles_coords, trunks_coords, gdf, center = load_landmarks_from_geojson(geojson_path)

    fig_map, ax_map = plt.subplots(figsize=(6, 6))
    ax_map.set_title("Landmarks and GPS Positions")
    ax_map.scatter(poles_coords[:, 0], poles_coords[:, 1], c='blue', label='Poles', s=10)
    ax_map.scatter(trunks_coords[:, 0], trunks_coords[:, 1], c='green', label='Trunks', s=10)
    ax_map.set_xlabel("X (centered UTM)")
    ax_map.set_ylabel("Y (centered UTM)")
    ax_map.legend()
    ax_map.grid(True)
    ax_map.set_aspect('equal')

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
            ros_time = datetime.fromtimestamp(frames.get_timestamp() / 1000.0, tz=timezone.utc) #- timedelta(seconds=16)

            gps_row = df_llh[df_llh["timestamp"] <= ros_time]
            if gps_row.empty or (ros_time - gps_row.iloc[-1]["timestamp"]).total_seconds() > SYNC_THRESHOLD:
                frame_idx += 1
                continue

            gps_row = gps_row.iloc[-1]
            gps_x = gps_row["utm_easting"] - center[0]
            gps_y = gps_row["utm_northing"] - center[1]

            # Plot fresh map with GPS
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.scatter(poles_coords[:, 0], poles_coords[:, 1], c='blue', label='Poles', s=10)
            ax.scatter(trunks_coords[:, 0], trunks_coords[:, 1], c='green', label='Trunks', s=10)
            ax.scatter(gps_x, gps_y, c='red', label='GPS', s=20, marker='x')
            ros_str = ros_time.strftime('%H:%M:%S.%f')[:-3]
            gps_str = gps_row["timestamp"].strftime('%H:%M:%S.%f')[:-3]
            ax.set_title(f"Frame {frame_idx:04d}\nROS Time: {ros_str} | GPS Time: {gps_str}")

            ax.set_xlabel("X (centered UTM)")
            ax.set_ylabel("Y (centered UTM)")
            ax.legend()
            ax.grid(True)
            ax.set_aspect('equal')

            map_img_path = os.path.join(output_folder, f"combined/map_{frame_idx:04d}.jpg")
            plt.savefig(map_img_path, bbox_inches='tight')
            plt.close(fig)

            # Read back the map image
            map_img = cv2.imread(map_img_path)
            rgb_resized = cv2.resize(color_img, (map_img.shape[1], map_img.shape[0]))

            combo = np.hstack((rgb_resized, map_img))
            final_path = os.path.join(output_folder, f"combined/frame_{frame_idx:04d}.jpg")
            cv2.imwrite(final_path, combo)

            os.remove(map_img_path)  # Clean intermediate
            frame_idx += 1
            pbar.update(1)

    except RuntimeError:
        print("[INFO] Finished ROS bag.")
    finally:
        pipeline.stop()


if __name__ == "__main__":
    process_rosbag_with_localization("data/ground/row_1_to_6.bag")
