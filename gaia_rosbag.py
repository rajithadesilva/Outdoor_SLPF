import os
import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO
from tqdm import tqdm
import torch

# Load YOLO model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
yolo = YOLO("./models/yolo.pt").to(device)

CLASS_IDS = [2, 4]  # e.g., poles and trunks
CLASS_NAMES = {2: 'poles', 4: 'trunks'}

def project_full_bev(depth, fx, fy, cx, cy, bev_size=(1000, 1000), bev_scale=100):
    """Project all valid depth points into white pixels in BEV."""
    h, w = depth.shape
    bev = np.zeros((bev_size[0], bev_size[1], 3), dtype=np.uint8)
    origin = (bev_size[1] // 2, bev_size[0])  # x-mid, z-bottom

    ys, xs = np.where(depth > 0)
    for u, v in zip(xs, ys):
        z = depth[v, u]
        x = (u - cx) * z / fx
        bev_x = int(origin[0] + x * bev_scale)
        bev_z = int(origin[1] - z * bev_scale)
        if 0 <= bev_x < bev_size[1] and 0 <= bev_z < bev_size[0]:
            bev[bev_z, bev_x] = [255, 255, 255]  # white
    return bev

def overlay_mask_on_bev(mask, depth, bev_image, fx, fy, cx, cy, bev_scale=100):
    """Overlay mask regions in red on the BEV image."""
    ys, xs = np.where(mask)
    origin = (bev_image.shape[1] // 2, bev_image.shape[0])  # x-mid, z-bottom

    for u, v in zip(xs, ys):
        z = depth[v, u]
        if z == 0:
            continue
        x = (u - cx) * z / fx
        bev_x = int(origin[0] + x * bev_scale)
        bev_z = int(origin[1] - z * bev_scale)
        if 0 <= bev_x < bev_image.shape[1] and 0 <= bev_z < bev_image.shape[0]:
            bev_image[bev_z, bev_x] = [0, 0, 255]  # red
    return bev_image

def process_rosbag_with_masked_bev(bag_path, output_folder="mask_bev_output"):
    os.makedirs(os.path.join(output_folder, "rgb_vis"), exist_ok=True)
    os.makedirs(os.path.join(output_folder, "bev"), exist_ok=True)

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device_from_file(bag_path, repeat_playback=False)
    config.enable_stream(rs.stream.color)
    config.enable_stream(rs.stream.depth)

    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)

    # Get intrinsics
    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    fx, fy, cx, cy = intr.fx, intr.fy, intr.ppx, intr.ppy
    print(f"[INFO] Intrinsics: fx={fx}, fy={fy}, cx={cx}, cy={cy}")

    frame_idx = 0
    try:
        pbar = tqdm(total=0, desc="Processing frames")
        while True:
            frames = pipeline.wait_for_frames()
            aligned = align.process(frames)

            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            color = np.asanyarray(color_frame.get_data())
            depth = np.asanyarray(depth_frame.get_data()).astype(np.float32) / 1000.0
            rgb_vis = color.copy()

            # Full BEV projection
            bev = project_full_bev(depth, fx, fy, cx, cy)

            # Get YOLO masks
            results = yolo.predict(color, conf=0.2, classes=CLASS_IDS, verbose=False)[0]
            if results.masks is not None:
                masks = results.masks.data.cpu().numpy()
                for i, mask in enumerate(masks):
                    cls_id = int(results.boxes.cls[i].item())
                    label = CLASS_NAMES.get(cls_id, str(cls_id))

                    # Resize mask to match color/depth image
                    mask_resized = cv2.resize(mask, (color.shape[1], color.shape[0]), interpolation=cv2.INTER_NEAREST)
                    binary_mask = (mask_resized > 0.5).astype(np.uint8)

                    # Random color for RGB mask
                    color_mask = np.random.randint(100, 255, size=(3,))
                    rgb_vis[binary_mask.astype(bool)] = color_mask

                    # Overlay mask onto BEV
                    bev = overlay_mask_on_bev(binary_mask, depth, bev, fx, fy, cx, cy)

                    # Draw bounding box and label
                    x1, y1, x2, y2 = map(int, results.boxes.xyxy[i].tolist())
                    cv2.rectangle(rgb_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(rgb_vis, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Save outputs
            cv2.imwrite(f"{output_folder}/rgb_vis/frame_{frame_idx:04d}.jpg", rgb_vis)
            cv2.imwrite(f"{output_folder}/bev/frame_{frame_idx:04d}.jpg", bev)
            frame_idx += 1
            pbar.update(1)

    except RuntimeError:
        print("[INFO] Finished processing.")
    finally:
        pipeline.stop()

if __name__ == "__main__":
    bag_file = "data/ground/row_1_to_6.bag"
    process_rosbag_with_masked_bev(bag_file)
