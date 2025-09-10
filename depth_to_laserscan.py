import pyrealsense2 as rs
import numpy as np
import cv2
import os

def depth_image_to_laserscan(
    depth_image,
    rgb_image,
    intrinsics,
    target_height=0.3,
    row_tolerance=0.02,
    depth_scale=1.0,
    camera_height=0.7,
    bev_radius=5.0,
    frame_idx=None,
    output_dir=None
):
    fx, fy = intrinsics["fx"], intrinsics["fy"]
    cx, cy = intrinsics["cx"], intrinsics["cy"]
    width, height = intrinsics["width"], intrinsics["height"]

    scan_ranges = []
    scan_points = []

    os.makedirs(output_dir, exist_ok=True)

    # --- SCAN VISUALIZATION SETUP ---
    scan_vis = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)
    scan_vis = scan_vis.astype(np.uint8)
    scan_vis = cv2.applyColorMap(scan_vis, cv2.COLORMAP_JET)

    for u in range(width):
        best_range = None
        best_v = None
        best_point = None
        r = None

        for v in range(height):
            z = depth_image[v, u] * depth_scale
            if z == 0 or np.isnan(z):
                continue

            x = (u - cx) * z / fx
            y = (v - cy) * z / fy
            y_ground = camera_height + y

            if abs(y_ground - target_height) < row_tolerance:
                r = np.sqrt(x ** 2 + z ** 2)
                best_range = r
                best_v = v
                best_point = (x, z)
                break

        scan_ranges.append(best_range if best_range else 0.0)
        scan_points.append(best_point)

        # --- SCAN VIEW POINT COLORING ---
        if best_v is not None:
            if best_range and best_range <= bev_radius:
                cv2.circle(scan_vis, (u, best_v), 1, (0, 255, 0), -1)  # green (within radius)
            else:
                cv2.circle(scan_vis, (u, best_v), 1, (0, 255, 255), -1)  # yellow (out of radius)

    scan_vis_resized = cv2.resize(scan_vis, (rgb_image.shape[1], rgb_image.shape[0]))

    # --- BEV VISUALIZATION ---
    bev_size = 500
    scale = 50  # pixels per meter
    bev_img = np.ones((bev_size, bev_size, 3), dtype=np.uint8) * 255
    center = (bev_size // 2, bev_size - 50)

    for pt in scan_points:
        if pt is None:
            continue
        x_m, z_m = pt
        r = np.sqrt(x_m**2 + z_m**2)
        if r > bev_radius:
            continue  # skip far points

        px = int(center[0] + x_m * scale)
        py = int(center[1] - z_m * scale)
        if 0 <= px < bev_size and 0 <= py < bev_size:
            cv2.circle(bev_img, (px, py), 2, (0, 0, 255), -1)

    bev_resized = cv2.resize(bev_img, (rgb_image.shape[1], rgb_image.shape[0]))

    # --- MERGE FINAL VISUALIZATION ---
    merged = np.hstack((scan_vis_resized, rgb_image, bev_resized))
    out_path = os.path.join(output_dir, f"frame_{frame_idx:04d}.jpg")
    cv2.imwrite(out_path, merged)

    return scan_ranges


# -------- CONFIG --------
BAG_PATH = "data/ground/row_1_to_6.bag"
FRAME_STRIDE = 10
TARGET_HEIGHT = 0.3  # meters
ROW_TOLERANCE = 0.02
OUTPUT_DIR = "output/scan_viz"

# -------- PIPELINE SETUP --------
pipeline = rs.pipeline()
config = rs.config()
rs.config.enable_device_from_file(config, BAG_PATH, repeat_playback=False)
config.enable_stream(rs.stream.depth)
config.enable_stream(rs.stream.color)

profile = pipeline.start(config)

# Get depth scale and intrinsics
device = profile.get_device()
depth_sensor = device.first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()

depth_stream = profile.get_stream(rs.stream.depth).as_video_stream_profile()
intr = depth_stream.get_intrinsics()
intrinsics = {
    "fx": intr.fx,
    "fy": intr.fy,
    "cx": intr.ppx,
    "cy": intr.ppy,
    "width": intr.width,
    "height": intr.height,
}

# -------- MAIN LOOP --------
frame_idx = 0
saved_idx = 0

try:
    while True:
        frames = pipeline.wait_for_frames(timeout_ms=10000)
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            break

        if frame_idx % FRAME_STRIDE != 0:
            frame_idx += 1
            continue

        depth_image = np.asanyarray(depth_frame.get_data())
        rgb_image = np.asanyarray(color_frame.get_data())

        _ = depth_image_to_laserscan(
            depth_image=depth_image,
            rgb_image=rgb_image,
            intrinsics=intrinsics,
            target_height=TARGET_HEIGHT,
            row_tolerance=ROW_TOLERANCE,
            depth_scale=depth_scale,
            frame_idx=saved_idx,
            output_dir=OUTPUT_DIR
        )

        print(f"Saved: frame_{saved_idx:04d}.jpg")
        saved_idx += 1
        frame_idx += 1

except Exception as e:
    print("Finished or encountered error:", e)

finally:
    pipeline.stop()
    print(f"Total frames saved: {saved_idx}")
