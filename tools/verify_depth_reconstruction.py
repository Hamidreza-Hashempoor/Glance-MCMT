import json
import os
import h5py
import numpy as np


warehouse_dir = "AIC25_Track1/Train/Warehouse_000"
gt_path = os.path.join(warehouse_dir, "ground_truth.json")

with open(gt_path, "r") as f:
    gt_data = json.load(f)

# Find frames where Camera_0005 has annotations
import random

cam_id = "Camera_0005"

# Find frames with visible bounding boxes in the desired camera
valid_frames = []
for frame_id, objs in gt_data.items():
    for obj in objs:
        if cam_id in obj.get("2d bounding box visible", {}):
            valid_frames.append((frame_id, obj))
            if len(valid_frames) >5:
                break  # One match per frame is enough

# Sample 1 frames
sampled = random.sample(valid_frames, min(5, len(valid_frames)))

# Print results
for frame_id, _ in sampled:
    print(f"Frame ID: {frame_id}")
    for obj in gt_data[frame_id]:
        if cam_id in obj["2d bounding box visible"]:
            bbox = obj["2d bounding box visible"][cam_id]
            cls = obj["object type"]
            loc3d = obj["3d location"]
            print(f"  - Class: {cls}, BBox: {bbox}, GT 3D: {loc3d}")



depth_map_path = os.path.join(warehouse_dir, "depth_map", "Camera_0005.h5")
depth_h5 = h5py.File(depth_map_path, "r")
with open(os.path.join(warehouse_dir, "calibration.json")) as f:
    calib = json.load(f)

bboxes_by_frame = {}
for frame_id, obj in sampled:
    bbox = obj["2d bounding box visible"][cam_id]
    label = obj["object type"]
    bboxes_by_frame.setdefault(frame_id, []).append((bbox, label, obj["3d location"]))

depth_h5 = h5py.File(depth_map_path, "r")
sensor_map = {sensor["id"]: sensor for sensor in calib["sensors"]}
intrinsic = np.array(sensor_map[cam_id]["intrinsicMatrix"], dtype=np.float32).reshape(3, 3)
# Load 3x4 extrinsic matrix

extrinsic_raw = np.array(sensor_map[cam_id]["extrinsicMatrix"], dtype=np.float32).reshape(3, 4)
extrinsic = np.eye(4)
extrinsic[:3, :] = extrinsic_raw

# Get inverse for camera → world
cam_to_world = np.linalg.inv(extrinsic)

for frame_id in list(bboxes_by_frame.keys())[:5]:  # Test one frame
    frame_idx = int(frame_id)  # ensure it's integer
    frame_key = f"distance_to_image_plane_{frame_idx:05d}.png"  # 5-digit zero-padding
    if frame_key in depth_h5:
        depth_frame = depth_h5[frame_key][:]
    else:
        print(f"[WARN] Frame {frame_key} not found in HDF5 depth map.")
    h, w = depth_frame.shape
    for (bbox, label, gt_3d) in bboxes_by_frame[frame_id]:
        x1, y1, x2, y2 = bbox
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

        if 0 <= cy < h and 0 <= cx < w:
            depth = depth_frame[cy, cx] / 1000.0  # mm → meters

            pixel = np.array([cx, cy, 1.0])
            cam_coords = np.linalg.inv(intrinsic) @ (pixel * depth)
            cam_coords_h = np.append(cam_coords, 1.0)  # homogeneous

            world_coords = cam_to_world @ cam_coords_h  # ← FIXED HERE
            print(f"Class {label}")
            print(f"→ Depth @center: {depth:.2f} m")
            print(f"→ Estimated 3D: {world_coords[:3]}")
            print(f"→ GT 3D: {gt_3d}\n")