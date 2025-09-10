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

# ---------- CONFIG ----------
FRAME_STRIDE = 9
device = 'cuda' if torch.cuda.is_available() else 'cpu'
yolo = YOLO("./models/yolo.pt").to(device)
CLASS_IDS = [2, 4]
CLASS_NAMES = {2: 'poles', 4: 'trunks'}
BEV_SIZE = (1000, 1000)
BEV_SCALE = 100
PARTICLE_COUNT = 100
PARTICLE_STD = 0.1
geojson_path = "data/riseholme_poles_trunk.geojson"

# ---------- MAP ----------
def load_landmarks_from_geojson(path):
    gdf = gpd.read_file(path)
    if not gdf.crs or not gdf.crs.is_projected:
        gdf = gdf.to_crs(gdf.estimate_utm_crs())
    gdf["x"] = gdf.geometry.x
    gdf["y"] = gdf.geometry.y
    coords = gdf[["x", "y"]].values
    coords -= coords.mean(axis=0)
    gdf["x_centered"] = coords[:, 0]
    gdf["y_centered"] = coords[:, 1]
    return coords, gdf

def render_map_to_bev(points, bev_size=BEV_SIZE, scale=BEV_SCALE):
    bev = np.zeros(bev_size, dtype=np.uint8)
    origin = (bev_size[1] // 2, bev_size[0])
    for x, z in points:
        mx = int(origin[0] + x * scale)
        mz = int(origin[1] - z * scale)
        if 0 <= mx < bev_size[1] and 0 <= mz < bev_size[0]:
            bev[mz, mx] = 255
    return bev

# ---------- PARTICLE FILTER ----------
def initialize_particles(n, extent=None, center=(12, 10), spread=1.0):
    if extent:
        low = np.array(extent[0])
        high = np.array(extent[1])
        return np.random.uniform(low=low, high=high, size=(n, 3))
    else:
        particles = np.zeros((n, 3))
        particles[:, 0] = np.random.normal(loc=center[0], scale=spread, size=n)
        particles[:, 1] = np.random.normal(loc=center[1], scale=spread, size=n)
        particles[:, 2] = np.random.uniform(-np.pi, np.pi, size=n)
        return particles

def motion_update(particles, std=PARTICLE_STD):
    return particles + np.random.normal(0, std, particles.shape)

def measurement_likelihood(bev_map, obs_bev, particles):
    scores = []
    for x, z, _ in particles:
        ox = int(BEV_SIZE[1] // 2 + x * BEV_SCALE)
        oz = int(BEV_SIZE[0] - z * BEV_SCALE)
        if 0 <= ox < BEV_SIZE[1] and 0 <= oz < BEV_SIZE[0]:
            patch = obs_bev[max(0, oz-20):oz+20, max(0, ox-20):ox+20, 2]
            map_patch = bev_map[max(0, oz-20):oz+20, max(0, ox-20):ox+20]
            score = np.sum((patch == 255) & (map_patch == 255)) if patch.shape == map_patch.shape else 0
        else:
            score = 0
        scores.append(score)
    weights = np.exp(np.array(scores) / 10.0)
    return weights / np.sum(weights)

def resample_particles(particles, weights):
    idx = np.random.choice(len(particles), size=len(particles), p=weights)
    return particles[idx]

# ---------- BEV GENERATION ----------
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

def overlay_mask_on_bev(mask, depth, bev, fx, fy, cx, cy):
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
            bev[bz, bx] = [0, 0, 255]
    return bev

# ---------- VISUALIZATION ----------
def visualize_particles(map_points, particles, frame_idx, output_dir, trajectory):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_title(f"Particles - Frame {frame_idx:04d}")

    padding = 1.0
    x_min, x_max = map_points[:, 0].min() - padding, map_points[:, 0].max() + padding
    y_min, y_max = map_points[:, 1].min() - padding, map_points[:, 1].max() + padding
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    if map_points.size > 0:
        ax.scatter(map_points[:, 0], map_points[:, 1], c='black', s=10, label='Landmarks')

    ax.scatter(particles[:, 0], particles[:, 1], c='red', s=5, label='Particles', alpha=0.6)
    '''
    if len(trajectory) > 1:
        traj = np.array(trajectory)
        ax.plot(traj[:, 0], traj[:, 1], c='green', linewidth=2, label='Trajectory')

    if trajectory:
        ax.scatter(*trajectory[-1], c='lime', s=40, label='Estimated Pose')
    '''
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3)
    ax.grid(True)
    ax.set_aspect('equal')

    out_path = os.path.join(output_dir, f"particles/frame_{frame_idx:04d}.jpg")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()

# ---------- MAIN LOOP ----------
def process_rosbag_with_localization(bag_path, output_folder="bev_localisation_output"):
    os.makedirs(os.path.join(output_folder, "rgb_vis"), exist_ok=True)
    os.makedirs(os.path.join(output_folder, "bev"), exist_ok=True)

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
    fx, fy, cx, cy = intr.fx, intr.fy, intr.ppx, intr.ppy

    map_points, gdf = load_landmarks_from_geojson(geojson_path)

    x_min, x_max = map_points[:, 0].min(), map_points[:, 0].max()
    y_min, y_max = map_points[:, 1].min(), map_points[:, 1].max()
    extent = [(x_min, y_min, -np.pi), (x_max, y_max, np.pi)]
    particles = initialize_particles(PARTICLE_COUNT, extent=extent)

    map_bev = render_map_to_bev(map_points)
    trajectory = []

    frame_idx = 0
    try:
        pbar = tqdm(desc="Processing frames")
        while True:
            frames = pipeline.wait_for_frames()
            if frame_idx % FRAME_STRIDE != 0:
                frame_idx += 1
                continue
            aligned = align.process(frames)
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            color = np.asanyarray(color_frame.get_data())
            depth = np.asanyarray(depth_frame.get_data()).astype(np.float32) / 1000.0
            rgb_vis = color.copy()
            bev = project_full_bev(depth, fx, fy, cx, cy)
            results = yolo.predict(color, conf=0.2, classes=CLASS_IDS, verbose=False)[0]

            if results.masks is not None:
                masks = results.masks.data.cpu().numpy()
                for mask in masks:
                    mask_resized = cv2.resize(mask, (color.shape[1], color.shape[0]), interpolation=cv2.INTER_NEAREST)
                    binary_mask = (mask_resized > 0.5).astype(np.uint8)
                    color_mask = np.random.randint(100, 255, size=(3,))
                    rgb_vis[binary_mask.astype(bool)] = color_mask
                    bev = overlay_mask_on_bev(binary_mask, depth, bev, fx, fy, cx, cy)

            particles = motion_update(particles)
            weights = measurement_likelihood(map_bev, bev, particles)
            particles = resample_particles(particles, weights)

            for x, z, _ in particles:
                px = int(BEV_SIZE[1] // 2 + x * BEV_SCALE)
                pz = int(BEV_SIZE[0] - z * BEV_SCALE)
                if 0 <= px < BEV_SIZE[1] and 0 <= pz < BEV_SIZE[0]:
                    cv2.circle(bev, (px, pz), 1, (0, 255, 255), -1)

            mean_pose = np.mean(particles[:, :2], axis=0)
            trajectory.append(mean_pose)

            px = int(BEV_SIZE[1] // 2 + mean_pose[0] * BEV_SCALE)
            pz = int(BEV_SIZE[0] - mean_pose[1] * BEV_SCALE)
            cv2.circle(bev, (px, pz), 5, (0, 255, 0), -1)

            base_name = f"frame_{frame_idx:04d}"
            cv2.imwrite(f"{output_folder}/rgb_vis/{base_name}.jpg", rgb_vis)
            cv2.imwrite(f"{output_folder}/bev/{base_name}.jpg", bev)
            visualize_particles(map_points, particles, frame_idx, output_folder, trajectory)

            frame_idx += 1
            pbar.update(1)

    except RuntimeError:
        print("[INFO] Finished ROS bag.")
    finally:
        pipeline.stop()

if __name__ == "__main__":
    bag_file = "data/ground/row_1_to_6.bag"
    process_rosbag_with_localization(bag_file)
