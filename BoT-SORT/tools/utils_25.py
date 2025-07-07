import argparse
import json
from collections import defaultdict, OrderedDict
from pathlib import Path
import os
import numpy as np
import cv2
import re
import math
from sklearn.preprocessing import normalize
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

import sys
try:
    sys.path.append('/data2/Hamid/AI_city_challenge_2025/AIC24_Track1_YACHIYO_RIIPS/BoT-SORT')
except:
    print( "bot sort already in path")


from collections import OrderedDict
from pathlib import Path


_COLORS = np.random.randint(0, 255, size=(500, 3), dtype=np.uint8)

def load_depth_map(h5_path):
    import h5py
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"Depth HDF5 not found: {h5_path}")
    return h5py.File(h5_path, 'r')


    

class CameraCalibration:
    def __init__(self):
        self.intrinsic_matrix = None
        self.cam_to_world_matrix = None
        self.homography_matrix = None
        self.camera_projection_matrix = None
        self.extrinsic_matrix = None
        self.camera_coordinates = {}
        self.scale_factor = None
        self.translation_to_global = {}

    def load_calibration(self, calib_path, camera_id):
            if not os.path.isfile(calib_path):
                print(f'\033[33mwarning\033[0m : Calibration file not found at: {calib_path}')
                print(f'\033[33mwarning\033[0m : World coordinate calculations are ignored.')
                return

            with open(calib_path, 'r') as file:
                data = json.load(file)

            for sensor in data.get("sensors", []):
                if sensor.get("type") == "camera" and sensor.get("id") == camera_id:
                    try:
                        # Load camera projection matrix (3x4)
                        self.camera_projection_matrix = np.array(sensor["cameraMatrix"], dtype=np.float32)

                        # Load homography matrix (3x3)
                        self.homography_matrix = np.array(sensor["homography"], dtype=np.float32)

                        # Load intrinsic matrix (3x3)
                        self.intrinsic_matrix = np.array(sensor["intrinsicMatrix"], dtype=np.float32)

                        # Load extrinsic matrix (3x4) and convert to 4x4
                        extrinsic_3x4 = np.array(sensor["extrinsicMatrix"], dtype=np.float32)
                        extrinsic_4x4 = np.eye(4, dtype=np.float32)
                        extrinsic_4x4[:3, :] = extrinsic_3x4
                        self.extrinsic_matrix = extrinsic_4x4

                        # Compute camera-to-world transformation
                        self.cam_to_world_matrix = np.linalg.inv(self.extrinsic_matrix)

                        # (Optional) Store origin and direction info
                        self.camera_coordinates = sensor.get("coordinates", {})
                        self.scale_factor = sensor.get("scaleFactor", None)
                        self.translation_to_global = sensor.get("translationToGlobalCoordinates", {})

                        print(f"✔ Loaded calibration for camera: {camera_id}")
                        return
                    except Exception as e:
                        print(f'\033[31merror\033[0m : Failed to parse calibration for {camera_id}: {e}')
                        return

            print(f'\033[33mwarning\033[0m : Camera ID "{camera_id}" not found in calibration file.')
    
    def convert_coordinates_to_3d_world(self, cx, cy, depth_frame):
        """
        Convert 2D image coordinates (cx, cy) and depth to 3D world coordinates.
        Applies clamping to ensure valid pixel access.

        Args:
            cx (int): X coordinate (center of bbox).
            cy (int): Y coordinate (bottom of bbox).
            depth_frame (ndarray): 2D numpy array with depth values.

        Returns:
            np.ndarray or None: 3D world coordinates [x, y, z] or None if invalid.
        """
        if self.intrinsic_matrix is None or self.cam_to_world_matrix is None:
            print("[ERROR] Calibration matrices not loaded.")
            return None

        h, w = depth_frame.shape
        clamped = False

        if cx < 0:
            print(f"[WARN] cx={cx} < 0 — clamped to 0")
            cx = 0
            clamped = True
        elif cx >= w:
            print(f"[WARN] cx={cx} >= width={w} — clamped to {w - 1}")
            cx = w - 1
            clamped = True

        if cy < 0:
            print(f"[WARN] cy={cy} < 0 — clamped to 0")
            cy = 0
            clamped = True
        elif cy >= h:
            print(f"[WARN] cy={cy} >= height={h} — clamped to {h - 1}")
            cy = h - 1
            clamped = True

        if clamped:
            print(f"[WARN] Pixel ({cx},{cy}) was out of bounds and has been clamped.")

        depth = depth_frame[cy, cx] / 1000.0  # convert mm to meters
        if depth == 0:
            print(f"[WARN] No depth at ({cx},{cy})")
            return None

        pixel = np.array([cx, cy, 1.0])
        cam_coords = np.linalg.inv(self.intrinsic_matrix) @ (pixel * depth)
        cam_coords_h = np.append(cam_coords, 1.0)
        world_coords = self.cam_to_world_matrix @ cam_coords_h

        return world_coords[:3]
    
    def convert_coordinates_to_3d_ground(self, cx, cy):
        """
        Convert 2D image coordinates (cx, cy) to 3D world coordinates assuming Z=0 (on ground).
        Uses homography computed from camera intrinsics and extrinsics.

        Args:
            cx (int): X coordinate (center of bbox).
            cy (int): Y coordinate (bottom of bbox).

        Returns:
            np.ndarray or None: 3D world coordinates [Xw, Yw, Zw=0] or None if calibration not loaded.
        """
        if self.intrinsic_matrix is None or self.cam_to_world_matrix is None:
            print("[ERROR] Calibration matrices not loaded.")
            return None

        # Compute inverse extrinsics: world-to-camera = inv(camera-to-world)
        world_to_cam = np.linalg.inv(self.cam_to_world_matrix)
        R = world_to_cam[:3, :3]
        t = world_to_cam[:3, 3:]

        # Build homography from world Z=0 plane to image
        H = self.intrinsic_matrix @ np.hstack((R[:, :2], t))  # 3x3

        try:
            H_inv = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            print("[ERROR] Cannot invert homography matrix.")
            return None

        pixel = np.array([cx, cy, 1.0])
        world_point_h = H_inv @ pixel
        world_point_h /= world_point_h[2]  # normalize homogeneous coordinates

        # Append Z=0 since this is a planar projection
        return np.array([world_point_h[0], world_point_h[1], 0.0])

def unload_depth_map(depth_h5):
    if depth_h5:
        depth_h5.close()
        print("[INFO] Depth map closed.")

def plot_tracking(image, tlwhs, track_ids, frame_id=None, line_thickness=2):
    im = image.copy()

    for tlwh, tid in zip(tlwhs, track_ids):
        x1, y1, w, h = map(int, tlwh)
        x2, y2 = x1 + w, y1 + h

        color = tuple(map(int, _COLORS[tid % len(_COLORS)]))
        cv2.rectangle(im, (x1, y1), (x2, y2), color, line_thickness)

        label = f'ID {tid}'
        cv2.putText(im, label, (x1, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    if frame_id is not None:
        cv2.putText(im, f'Frame {frame_id}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

    return im

def plot_tracking_with_world(
    image, tlwhs, track_ids, world_coords=None, gt_coords=None, frame_id=None, line_thickness=2
):
    im = image.copy()

    for i, (tlwh, tid) in enumerate(zip(tlwhs, track_ids)):
        x1, y1, w, h = map(int, tlwh)
        x2, y2 = x1 + w, y1 + h
        color = tuple(map(int, _COLORS[tid % len(_COLORS)]))

        cv2.rectangle(im, (x1, y1), (x2, y2), color, line_thickness)
        label = f'ID {tid}'

        # Append estimated 3D coordinate if available
        if world_coords is not None and i < len(world_coords):
            wc = world_coords[i]
            if wc is not None and len(wc) == 3:
                label += f' | 3D({wc[0]:.1f},{wc[1]:.1f},{wc[2]:.1f})'

        # Append GT 3D coordinate if available
        if gt_coords is not None and i < len(gt_coords):
            gt = gt_coords[i]
            if gt is not None and len(gt) == 3:
                label += f' | GT({gt[0]:.1f},{gt[1]:.1f},{gt[2]:.1f})'

        cv2.putText(im, label, (x1, max(0, y1 - 7)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    if frame_id is not None:
        cv2.putText(im, f'Frame {frame_id}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

    return im

def get_gt_coords_for_tracks(gt_data, camera_id, frame_id, tlwhs):
    """
    For a list of predicted bboxes, find closest GT object in 2D and return its 3D coordinates.

    Args:
        gt_data (dict): Parsed ground truth JSON (already loaded).
        camera_id (str): e.g., "Camera_0005"
        frame_id (int): Frame number
        tlwhs (List[List[float]]): List of predicted TLWH boxes

    Returns:
        List[Optional[Tuple[float, float, float]]]: GT 3D coordinates for each input box, or None if not found
    """
    if isinstance(frame_id, int):
        frame_id = str(frame_id)

    if frame_id not in gt_data:
        print(f"[WARN] Frame {frame_id} not found in GT data.")
        return [None] * len(tlwhs)

    gt_objs = gt_data[frame_id]
    gt_bboxes = []
    for obj in gt_objs:
        bbox_dict = obj.get("2d bounding box visible", {})
        if camera_id in bbox_dict:
            bbox = bbox_dict[camera_id]
            center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
            gt_bboxes.append((center, obj.get("3d location")))

    results = []
    for tlwh in tlwhs:
        x, y, w, h = tlwh
        pred_center = ((x + x + w) / 2, (y + y + h) / 2)

        min_dist = float('inf')
        closest_3d = None
        for gt_center, gt_3d in gt_bboxes:
            if gt_3d is not None:
                dist = np.sqrt((gt_center[0] - pred_center[0])**2 + (gt_center[1] - pred_center[1])**2)
                if dist < min_dist:
                    min_dist = dist
                    closest_3d = gt_3d

        results.append(tuple(closest_3d) if closest_3d else None)

    return results

def compute_average_3d_scales(gt_data):
    """
    Compute average 3D bounding box scale for each object class in the ground truth file.

    Args:
        gt_path (str): Path to ground_truth.json

    Returns:
        dict: Mapping from class name to average [x, y, z] scale
    """
    class_scales = defaultdict(list)

    for frame_objs in gt_data.values():
        for obj in frame_objs:
            cls_name = obj.get("object type")
            scale = obj.get("3d bounding box scale")
            if cls_name and scale:
                class_scales[cls_name].append(scale)

    stats = {}
    for cls_name, scales in class_scales.items():
        scales_np = np.array(scales)
        mean_scale = np.mean(scales_np, axis=0).tolist()
        std_scale = np.std(scales_np, axis=0).tolist()
        stats[cls_name] = {
            "mean": mean_scale,
            "std": std_scale,
        }

    return stats

def load_full_detections_per_frame(json_path):
    """
    Loads full detection info grouped by chronological frame order.

    Returns:
        OrderedDict[int, list[dict]]: Sorted by frame ID (ascending).
    """
    with open(json_path, 'r') as f:
        raw_data = json.load(f)

    temp_group = defaultdict(list)

    for item in raw_data.values():
        frame_id = item["Frame"]
        temp_group[frame_id].append({
            "ImgPath": item["ImgPath"],
            "NpyPath": item["NpyPath"],
            "Coordinate": item["Coordinate"],
            "ClassID": item.get("ClassID", -1),
            "Frame": frame_id
        })

    # Sort by frame ID and convert to OrderedDict
    frame_to_detections = OrderedDict(sorted(temp_group.items()))

    return frame_to_detections

def preview_frames(frame_detections, max_frames=3):
    """
    Print a preview of detection entries for the first few frames.

    Args:
        frame_detections (dict): Output from load_full_detections_per_frame.
        max_frames (int): Number of frames to preview.
    """
    for frame_id in sorted(frame_detections.keys())[:max_frames]:
        detections = frame_detections[frame_id]
        print(f"\nFrame {frame_id} has {len(detections)} objects:")
        for det in detections:
            coord = det["Coordinate"]
            print(f"  BBox: ({coord['x1']},{coord['y1']}) to ({coord['x2']},{coord['y2']}), Class: {det['ClassID']}, Npy: {det['NpyPath']}")

def print_nth_frame_detections(frame_detections, n=1):
    """
    Print all detection entries from the N-th frame (ordered by frame ID).
    
    Args:
        frame_detections (OrderedDict[int, list[dict]]): Frame → list of detections
        n (int): N-th frame to show (1-based index)
    """
    frame_keys = list(frame_detections.keys())

    if n < 1 or n > len(frame_keys):
        print(f"Invalid N: {n}. Only {len(frame_keys)} frames available.")
        return

    frame_id = frame_keys[n - 1]
    detections = frame_detections[frame_id]

    print(f"\n=== Frame {n} (Frame ID: {frame_id}) has {len(detections)} detections ===")
    for i, det in enumerate(detections):
        print(f"\nObject {i + 1}:")
        for k, v in det.items():
            print(f"  {k}: {v}")


def estimate_yaw_from_world_motion(track_world_coords, min_motion_threshold=0.05, gap=6):
    """
    Estimate 3D rotation (yaw) from object motion in world coordinates with frame gap.

    Args:
        track_world_coords (List[Tuple[float, float]]): (x, y) for each frame.
        min_motion_threshold (float): Ignore very small motion steps.
        gap (int): Frame gap to skip between comparisons.

    Returns:
        float: estimated yaw angle in radians.
    """
    directions = []

    for i in range(len(track_world_coords) - gap):
        x1, y1 = track_world_coords[i]
        x2, y2 = track_world_coords[i + gap]

        dx, dy = x2 - x1, y2 - y1
        motion_norm = math.hypot(dx, dy)

        if motion_norm > min_motion_threshold:
            yaw = math.atan2(dy, dx)
            directions.append(yaw)

    if not directions:
        return 0.0

    # Average using atan2 of summed vectors (avoids angle wrap issues)
    sin_sum = sum(math.sin(a) for a in directions)
    cos_sum = sum(math.cos(a) for a in directions)
    return math.atan2(sin_sum, cos_sum)

def analyze_gt_rotations_by_class(gt_data):
    """
    Analyze ground-truth 3D rotations per class from a ground_truth.json file.

    Args:
        gt_path (str): Path to ground_truth.json

    Returns:
        dict: {
            class_name: {
                "mean": [roll_mean, pitch_mean, yaw_mean],
                "std":  [roll_std, pitch_std, yaw_std],
                "count": int
            }
        }
    """

    class_rotations = defaultdict(list)

    for frame_objs in gt_data.values():
        for obj in frame_objs:
            class_name = obj.get("object type")
            rotation = obj.get("3d bounding box rotation")
            if class_name and rotation and len(rotation) == 3:
                class_rotations[class_name].append(rotation)

    results = {}
    for class_name, rots in class_rotations.items():
        rot_array = np.array(rots)
        mean = rot_array.mean(axis=0).tolist()
        std = rot_array.std(axis=0).tolist()
        results[class_name] = {
            "mean": mean,
            "std": std,
            "count": len(rots)
        }
        print(f"{class_name}:")
        print(f"  ➤ Mean Rotation   = {mean}")
        print(f"  ➤ Std Deviation   = {std}")
        print(f"  ➤ Total Instances = {len(rots)}\n")

    return results

def get_yaw_for_tracks(online_targets, min_motion_threshold=0.05):
    """
    Estimate yaw for each STrack in online_targets using their world coordinates.

    Args:
        online_targets (List[STrack]): List of currently active tracks.
        min_motion_threshold (float): Minimum motion to consider valid.

    Returns:
        Dict[int, float]: Mapping of track_id to estimated yaw angle (radians).
    """
    track_yaws = {}

    for track in online_targets:
        # Extract full history of world coordinates if available
        # Example: you store it as `track.world_coord_history`
        coords = getattr(track, 'world_coord_history', [])

        # Extract (x, y) only from (x, y, z)
        xy_coords = [(pt[0], pt[1]) for pt in coords if pt is not None and len(pt) >= 2]

        if len(xy_coords) >= 2:
            yaw = estimate_yaw_from_world_motion(xy_coords, min_motion_threshold)
        else:
            yaw = 0.0

        track_yaws[track.track_id] = yaw

    return track_yaws

def fixed_compute_iou(box1, box2):
    """Compute IoU between two 2D bboxes, with consideration for contained boxes"""
    x1, y1, x2, y2 = box1
    x1_p, y1_p, x2_p, y2_p = box2

    xi1, yi1 = max(x1, x1_p), max(y1, y1_p)
    xi2, yi2 = min(x2, x2_p), min(y2, y2_p)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x2_p - x1_p) * (y2_p - y1_p)
    union_area = area1 + area2 - inter_area

    if union_area == 0:
        return 0.0

    # To emphasize overlap when one box is contained in another
    smaller_area = min(area1, area2)

    # Adjust the IoU score for situations where one box is inside the other
    iou = inter_area / smaller_area  # Normalize by the smaller box

    return iou

def compute_iou_intersection(box1, box2):
    """Compute IoU between two 2D bboxes, with consideration for contained boxes"""
    x1, y1, x2, y2 = box1
    x1_p, y1_p, x2_p, y2_p = box2

    xi1, yi1 = max(x1, x1_p), max(y1, y1_p)
    xi2, yi2 = min(x2, x2_p), min(y2, y2_p)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x2_p - x1_p) * (y2_p - y1_p)
    union_area = area1 + area2 - inter_area

    if union_area == 0:
        return 0.0

    # To emphasize overlap when one box is contained in another
    smaller_area = min(area1, area2)

    # Adjust the IoU score for situations where one box is inside the other
    iou = inter_area / smaller_area  # Normalize by the smaller box

    return iou

def compute_iou(box1, box2):
    """Compute IoU between two 2D bboxes"""
    x1, y1, x2, y2 = box1
    x1_p, y1_p, x2_p, y2_p = box2

    xi1, yi1 = max(x1, x1_p), max(y1, y1_p)
    xi2, yi2 = min(x2, x2_p), min(y2, y2_p)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x2_p - x1_p) * (y2_p - y1_p)
    union_area = area1 + area2 - inter_area

    return inter_area / union_area if union_area > 0 else 0.0

def assign_features_to_tracklets(tracklet_data, det_by_frame, camera_id, iou_thresh=0.8, overlap_block_thresh=0.4):
    for frame_id, objects in tracklet_data.items():
        dets = det_by_frame.get(frame_id, [])

        for obj in objects:
            bbox = obj["2d bounding box visible"].get(camera_id)
            if not bbox:
                continue

            best_iou = 0.0
            best_npy_path = None
            overlapping_dets = 0

            for det in dets:
                det_box = det["bbox"]
                iou = compute_iou(bbox, det_box)

                if iou > best_iou:
                    best_iou = iou
                    best_npy_path = det["npy_path"]

                if iou >= overlap_block_thresh:
                    overlapping_dets += 1

            if best_iou >= iou_thresh and overlapping_dets == 1:
                obj["feature path"] = best_npy_path
            else:
                obj["feature path"] = None

    return tracklet_data

# def visualize_fixed_tracklets(data, image_root, camera_id, save_root, max_frames=500):
#     os.makedirs(save_root, exist_ok=True)
#     frame_ids = sorted(map(int, data.keys()))[:max_frames]

#     for frame_id in frame_ids:
#         img_path = os.path.join(image_root, f"{frame_id:06d}.jpg")
#         if not os.path.exists(img_path):
#             continue

#         img = cv2.imread(img_path)
#         objects = data[str(frame_id)]

#         tlwhs, ids, world_coords = [], [], []
#         for obj in objects:
#             bbox = obj["2d bounding box visible"].get(camera_id)
#             if not bbox:
#                 continue
#             x1, y1, x2, y2 = bbox
#             tlwhs.append([x1, y1, x2 - x1, y2 - y1])
#             ids.append(obj["object sc id"])
#             world_coords.append(obj.get("3d location", [0.0, 0.0, 0.0]))

#         out_img = plot_tracking_with_world(img, tlwhs, ids, world_coords=world_coords, frame_id=frame_id)
#         out_path = os.path.join(save_root, f"{frame_id:06d}.jpg")
#         cv2.imwrite(out_path, out_img)

def visualize_fixed_tracklets(data, image_root, camera_id, save_root, max_frames=500):
    os.makedirs(save_root, exist_ok=True)
    all_frame_ids = sorted(map(int, data.keys()))

    # Uniformly sample frames: e.g., 1, 51, 101, ...
    total_frames = len(all_frame_ids)
    if total_frames == 0:
        print("[WARN] No frames found in data.")
        return

    step = max(1, total_frames // max_frames)
    sampled_frame_ids = all_frame_ids[::step][:max_frames]

    for frame_id in sampled_frame_ids:
        img_path = os.path.join(image_root, f"{frame_id:06d}.jpg")
        if not os.path.exists(img_path):
            continue

        img = cv2.imread(img_path)
        objects = data[str(frame_id)]

        tlwhs, ids, world_coords = [], [], []
        for obj in objects:
            bbox = obj["2d bounding box visible"].get(camera_id)
            if not bbox:
                continue
            x1, y1, x2, y2 = bbox
            tlwhs.append([x1, y1, x2 - x1, y2 - y1])
            ids.append(obj["object sc id"])
            world_coords.append(obj.get("3d location", [0.0, 0.0, 0.0]))

        out_img = plot_tracking_with_world(img, tlwhs, ids, world_coords=world_coords, frame_id=frame_id)
        out_path = os.path.join(save_root, f"{frame_id:06d}.jpg")
        cv2.imwrite(out_path, out_img)
        

def load_detection_json(det_json_path):
    with open(det_json_path, "r") as f:
        detections = json.load(f)

    det_by_frame = defaultdict(list)
    for det_id, det_info in detections.items():
        frame_id = str(det_info["Frame"])
        det_by_frame[frame_id].append({
            "bbox": list(det_info["Coordinate"].values()),
            "npy_path": det_info["NpyPath"],
            "class_id": det_info["ClassID"]
        })
    return det_by_frame

def fix_zero_yaws_by_tracklet(tracklet_data, yaw_index=2):
    from collections import defaultdict

    # Group objects by object sc id
    tracks = defaultdict(list)
    for frame_id, objs in tracklet_data.items():
        for obj in objs:
            tid = obj["object sc id"]
            tracks[tid].append((int(frame_id), obj))

    # For each tracklet, fix zero yaw
    for tid, entries in tracks.items():
        entries.sort()  # Sort by frame ID

        # Extract all yaw values in order
        yaws = [obj["3d bounding box rotation"][yaw_index] for _, obj in entries]

        # Pass 1: collect valid (non-zero) yaws
        nonzero_yaws = [(i, y) for i, y in enumerate(yaws) if abs(y) > 1e-3]

        if not nonzero_yaws:
            continue  # All yaws are 0, skip

        for i, (frame_id, obj) in enumerate(entries):
            if abs(yaws[i]) <= 1e-3:  # 0.0 or near-zero
                # Find nearest non-zero yaw
                nearest = min(nonzero_yaws, key=lambda x: abs(x[0] - i))
                obj["3d bounding box rotation"][yaw_index] = nearest[1]

    return tracklet_data

def remove_invalid_and_short_tracklets(data, min_track_len=20):
    """
    Remove tracklets with None object type and those shorter than min_track_len.
    Reassigns object IDs sequentially.
    """
    from collections import defaultdict

    # Group valid tracklets by ID
    track_by_id = defaultdict(list)
    for frame_id, objects in data.items():
        for obj in objects:
            tid = obj["object sc id"]
            if obj.get("object type") is not None:
                track_by_id[tid].append((int(frame_id), obj))

    print(f"[CLEANUP] Number of tracklets after none object type filtering: {len(track_by_id)}")

    # Filter short tracklets
    track_by_id = {
        tid: track for tid, track in track_by_id.items()
        if len(track) >= min_track_len
    }

    print(f"[CLEANUP] Number of tracklets after track len limit filtering: {len(track_by_id)}")

    # Rebuild data and reassign IDs
    new_data = defaultdict(list)
    for new_id, track in enumerate(track_by_id.values(), 1):
        for frame_id, obj in track:
            obj["object sc id"] = new_id
            new_data[str(frame_id)].append(obj)

    return new_data

def print_tracklet_coords_by_frame(data, tracklet_ids=(1, 2, 3, 4, 5)):
    """
    Print frame-wise 3D coordinates of specified tracklet IDs.
    If a tracklet ID is missing in a frame, print 'None' for that ID.
    """
    sorted_frame_ids = sorted(map(int, data.keys()))

    for frame_id in sorted_frame_ids:
        if frame_id > 50:
            continue
        frame_objects = data[str(frame_id)]
        # Create a lookup from track ID to 3D coords
        id_to_coords = {
            obj["object sc id"]: obj.get("3d location")
            for obj in frame_objects
        }

        for tid in tracklet_ids:
            coords = id_to_coords.get(tid, None)
            print(f"frame={frame_id}, id={tid}, coords={coords}")
        

def interpolate_missing_tracklet_frames(data, camera_key="Camera"):
    from collections import defaultdict
    import numpy as np

    # Group by track ID
    track_by_id = defaultdict(dict)
    for frame_id, objects in data.items():
        for obj in objects:
            tid = obj["object sc id"]
            track_by_id[tid][int(frame_id)] = obj

    for tid, frames in track_by_id.items():
        all_frames = sorted(frames.keys())
        if len(all_frames) < 2:
            continue

        first_f, last_f = all_frames[0], all_frames[-1]

        for f in range(first_f, last_f + 1):
            if f in frames:
                continue

            # Find neighbors
            prev_f = max([ff for ff in all_frames if ff < f], default=None)
            next_f = min([ff for ff in all_frames if ff > f], default=None)
            if prev_f is None or next_f is None:
                continue

            prev_obj = frames[prev_f]
            next_obj = frames[next_f]

            ratio = (f - prev_f) / (next_f - prev_f + 1e-6)

            def interp(a, b):
                return [a[i] + (b[i] - a[i]) * ratio for i in range(len(a))]

            interp_obj = {
                "object type": prev_obj["object type"],
                "object sc id": tid,
                "3d location": interp(prev_obj["3d location"], next_obj["3d location"]),
                "3d bounding box scale": prev_obj["3d bounding box scale"],
                "3d bounding box rotation": [
                    0.0, 0.0,
                    prev_obj["3d bounding box rotation"][2] +
                    (next_obj["3d bounding box rotation"][2] - prev_obj["3d bounding box rotation"][2]) * ratio
                ],
                "2d bounding box visible": {
                    camera_key: interp(
                        prev_obj["2d bounding box visible"][camera_key],
                        next_obj["2d bounding box visible"][camera_key]
                    )
                },
                "feature path": None
            }


            frame_str = str(f)
            existing_objs = data.get(frame_str, [])

            # Remove any existing object with same ID (in case it's a placeholder or duplicate)
            existing_objs = [o for o in existing_objs if o["object sc id"] != tid]

            # Add interpolated object
            existing_objs.append(interp_obj)

            # Update frame data
            data[frame_str] = existing_objs
    print(f"[INTERPOLATION] Interpolation process is finished for : {camera_key}")
    return data


def is_trajectory_similar(coords1, coords2, threshold=2.5):
    coords1 = np.array(coords1)
    coords2 = np.array(coords2)
    min_len = min(len(coords1), len(coords2))
    if min_len == 0:
        return False
    dist = np.linalg.norm(coords1[:min_len] - coords2[:min_len], axis=1)
    return np.mean(dist) < threshold

def compute_feature_similarity(f1_list, f2_list, args):
    if not f1_list or not f2_list:
        return -1
    f1 = np.stack(f1_list)
    f2 = np.stack(f2_list)
    sim_matrix = cosine_similarity(f1, f2)
    k = min(len(f1), len(f2), 5)
    top_similarities = np.sort(sim_matrix.flatten())[-k:]
    return np.mean(top_similarities)

def is_feature_similar(f1, f2, sim_thresh=0.69, metric="cosine"):
    if len(f1) == 0 or len(f2) == 0:
        return False

    f1 = np.stack(f1)
    f2 = np.stack(f2)

    # Normalize if using cosine similarity and features not already normalized
    if metric == "cosine":
        f1_norm = np.linalg.norm(f1, axis=1)
        f2_norm = np.linalg.norm(f2, axis=1)

        if not (np.allclose(f1_norm, 1.0, atol=1e-3) and np.allclose(f2_norm, 1.0, atol=1e-3)):
            f1 = normalize(f1, norm='l2')
            f2 = normalize(f2, norm='l2')

        sim_matrix = cosine_similarity(f1, f2)
    elif metric == "euclidean":
        # Convert distance to similarity (larger value = more similar)
        dist_matrix = euclidean_distances(f1, f2)
        sim_matrix = -dist_matrix  # You can use a threshold on -distance or convert to similarity another way
    else:
        raise ValueError("Unsupported metric. Use 'cosine' or 'euclidean'.")

    # For cosine: higher = better. For euclidean: lower = better.
    if metric == "cosine":
        compare_matrix = sim_matrix >= sim_thresh
    elif metric == "euclidean":
        compare_matrix = sim_matrix >= -sim_thresh  # since sim_matrix = -distances

    # Count how many feature pairs exceed the threshold
    match_count = np.count_nonzero(compare_matrix)

    k = min(len(f1), len(f2), 5)
    return match_count >= k

def merge_redundant_global_ids(global_tracklets, args):
    from collections import defaultdict

    # Step 1: group tracklets by global_id
    gid_to_tracklets = defaultdict(list)
    for tr in global_tracklets:
        gid_to_tracklets[tr["global_id"]].append(tr)

    gids = sorted(gid_to_tracklets.keys())

    # Step 2: build merge map using Union-Find
    parent = {gid: gid for gid in gids}

    def find(gid):
        while parent[gid] != gid:
            parent[gid] = parent[parent[gid]]
            gid = parent[gid]
        return gid

    def union(gid1, gid2):

        p1, p2 = find(gid1), find(gid2)
        if p1 != p2:
            if p1 < p2:
                parent[p2] = p1
            else:
                parent[p1] = p2

    # Step 3: Compare each gid pair (gid2 > gid1)
    for i in range(len(gids)):
        for j in range(i + 1, len(gids)):
            gid1, gid2 = gids[i], gids[j]
            tracklets1 = gid_to_tracklets[gid1]
            tracklets2 = gid_to_tracklets[gid2]

            cam_ids1 = set(tr["camera_id"] for tr in tracklets1)
            cam_ids2 = set(tr["camera_id"] for tr in tracklets2)

            # Skip merging if both global IDs already exist in the same camera
            if cam_ids1 & cam_ids2:
                continue

            matched = False
            for tr1 in tracklets1:
                for tr2 in tracklets2:
                    if is_trajectory_similar(tr1["coords"], tr2["coords"], threshold=args.trajectory_dist_thresh_merge):
                        union(gid1, gid2)
                        matched = True
                        break
                if matched:
                    break

    # Step 4: Relabel global IDs
    gid_map = {gid: find(gid) for gid in gids}

    for tr in global_tracklets:
        tr["global_id"] = gid_map[tr["global_id"]]

    merged_gids = set(tr["global_id"] for tr in global_tracklets)
    print(f"[INFO] After merging: {len(global_tracklets)} tracklets → {len(merged_gids)} unique global IDs")
    return global_tracklets

def filter_expressive_features(object_features: dict, cam_folder, args) -> dict:
    filtered = {}
    representitive_feature_len = args.representitive_feature_len
    metric = args.metric
    for sc_id, features in object_features.items():
        if len(features) == 0:
            continue

        feats = np.stack(features)  # shape: (N, D)
        if len(features) == 1:
            filtered[sc_id] = features
            continue

        # Compute pairwise similarity
        if metric == "cosine":
            # Normalize for cosine
            # Check if features are already normalized
            dot_products = np.einsum('ij,ij->i', feats, feats)  # ⟨f, f⟩ per row
            is_normalized = np.allclose(dot_products, 1.0, atol=1e-3)

            if not is_normalized:
                feats_norm = normalize(feats, norm='l2')
            else:
                feats_norm = feats

            # Cluster with DBSCAN (cosine distance = 1 - cosine similarity)
            clustering = DBSCAN(eps=0.2, min_samples=3, metric='cosine').fit(feats_norm)
            labels = clustering.labels_

            # Pick the largest non-noise cluster
            unique_labels, counts = np.unique(labels[labels >= 0], return_counts=True)
            if len(unique_labels) == 0:
                print(f"[WARN] {cam_folder} sc_id={sc_id} has no dense cluster → fallback to first feature")
                filtered[sc_id] = [features[0]]
                continue

            main_cluster = unique_labels[np.argmax(counts)]
            inlier_indices = np.where(labels == main_cluster)[0]
            selected_indices = inlier_indices[:representitive_feature_len]  # limit to feature_len

            filtered_features = [features[i] for i in selected_indices]
            filtered[sc_id] = filtered_features
        elif metric == "euclidean":
            # Compute pairwise similarity as negative distances
            sim_matrix = -euclidean_distances(feats)

            # Mean similarity for each feature (excluding self)
            mean_sim = np.sum(sim_matrix, axis=1) - np.diag(sim_matrix)
            mean_sim /= (len(features) - 1)

            # Rank and select top-K
            sorted_indices = np.argsort(-mean_sim)
            top_indices = sorted_indices[:min(representitive_feature_len, len(features))]
            filtered_features = [features[i] for i in top_indices]

            filtered[sc_id] = filtered_features
        else:
            raise ValueError("Unsupported metric. Use 'cosine' or 'euclidean'.")


    return filtered

# def ensure_all_coords_have_features(data, object_coords, object_features, camera_folder):
#     missing_ids = [sc_id for sc_id in object_coords if sc_id not in object_features]

#     for sc_id in missing_ids:
#         object_features[sc_id] = []

#         # Scan all frames sorted by frame number
#         for frame_str in sorted(data.keys(), key=lambda x: int(x)):
#             objs = data[frame_str]

#             for obj in objs:
#                 if obj["object sc id"] != sc_id:
#                     continue

#                 feat_path = obj.get("feature path", None)
#                 if feat_path and feat_path.lower().endswith(".npy"):
#                     try:
#                         feat_full_path = os.path.join("EmbedFeature", feat_path)
#                         feature = np.load(feat_full_path)
#                         object_features[sc_id].append(feature)
#                         print(f"[INFO] Loaded fallback feature for {camera_folder} sc_id={sc_id} from frame {frame_str}")
#                         break  # Stop after first found feature
#                     except Exception as e:
#                         print(f"[WARN] Failed to load fallback feature for {camera_folder} sc_id={sc_id} from {feat_path}: {e}")
#             if object_features[sc_id]:
#                 break  # Done once we have one feature

def ensure_all_coords_have_features(data, object_coords, object_features, camera_folder):
    missing_ids = [sc_id for sc_id in object_coords if sc_id not in object_features]

    for sc_id in missing_ids:
        object_features[sc_id] = []
        found_valid_path = False  # Track if we even saw a non-None .npy path
        # Scan all frames sorted by frame number
        for frame_str in sorted(data.keys(), key=lambda x: int(x)):
            objs = data[frame_str]

            for obj in objs:
                if obj["object sc id"] != sc_id:
                    continue

                feat_path = obj.get("feature path", None)

                if feat_path is None:
                    continue  # Not even a path to try

                if not feat_path.lower().endswith(".npy"):
                    continue  # Not a valid .npy path

                found_valid_path = True

                try:
                    feat_full_path = os.path.join("EmbedFeature", feat_path)
                    feature = np.load(feat_full_path)
                    object_features[sc_id].append(feature)
                    print(f"[INFO] Loaded fallback feature for {camera_folder} sc_id={sc_id} from frame {frame_str}")
                    break  # Stop after first successful load
                except Exception as e:
                    print(f"[WARN] Failed to load fallback feature for {camera_folder} sc_id={sc_id} from {feat_path}: {e}")

            if object_features[sc_id]:
                break  # Stop outer loop once at least one feature loaded

        if not found_valid_path:
            print(f"[WARN] No valid feature path (.npy) found for {camera_folder} sc_id={sc_id} — all were missing or invalid")
        elif len(object_features[sc_id]) == 0:
            print(f"[WARN] All feature paths failed to load for {camera_folder} sc_id={sc_id}")

def ensure_all_frame_coords_have_features(data, object_coords, object_features, camera_folder):
    """
    Ensures each sc_id in object_coords has at least one feature in object_features.
    If not, attempts to load one from the data based on frame_id.

    Args:
        data: Full per-frame JSON object data for the camera.
        object_coords: dict of sc_id -> list of (frame_id, coord)
        object_features: dict of sc_id -> list of np.array
        camera_folder: e.g. "Camera_01"
    """
    missing_ids = [sc_id for sc_id in object_coords if sc_id not in object_features or not object_features[sc_id]]

    for sc_id in missing_ids:
        object_features[sc_id] = []
        found_valid_path = False

        for frame_id, _ in object_coords[sc_id]:
            frame_str = str(frame_id)
            objs = data.get(frame_str, [])

            for obj in objs:
                if obj["object sc id"] != sc_id:
                    continue

                feat_path = obj.get("feature path", None)
                if feat_path is None or not feat_path.lower().endswith(".npy"):
                    continue

                found_valid_path = True

                try:
                    feat_full_path = os.path.join("EmbedFeature", feat_path)
                    feature = np.load(feat_full_path)
                    object_features[sc_id].append(feature)
                    print(f"[INFO] Loaded fallback feature for {camera_folder} sc_id={sc_id} from frame {frame_str}")
                    break  # Only need one valid feature
                except Exception as e:
                    print(f"[WARN] Failed to load fallback feature for {camera_folder} sc_id={sc_id} from {feat_path}: {e}")

            if object_features[sc_id]:
                break  # Stop outer loop once loaded

        if not found_valid_path:
            print(f"[WARN] No valid feature path (.npy) found for {camera_folder} sc_id={sc_id} — all were missing or invalid")
        elif len(object_features[sc_id]) == 0:
            print(f"[WARN] All feature paths failed to load for {camera_folder} sc_id={sc_id}")


def assign_initial_global_ids(all_tracklets_by_camera, args):
    global_tracklets = []  # List of assigned tracklets
    global_id_counter = 0

    for cam_index, tracklets in enumerate(all_tracklets_by_camera):
        local_used_gids = set()  # Prevent duplicate global IDs in the same camera
        for tracklet in tracklets:
            matched_gid = None

            for global_tr in global_tracklets:
                if is_trajectory_similar(tracklet["coords"], global_tr["coords"], threshold=args.trajectory_dist_thresh_init):
                    if is_feature_similar(tracklet["features"], global_tr["features"], sim_thresh=args.feature_similarity_thresh_init, metric= args.metric):
                        if global_tr["global_id"] not in local_used_gids:
                            matched_gid = global_tr["global_id"]
                            break

            if matched_gid is None:
                matched_gid = global_id_counter
                global_id_counter += 1

            tracklet["global_id"] = matched_gid
            local_used_gids.add(matched_gid)
            global_tracklets.append(tracklet)

    unique_gids = set(t["global_id"] for t in global_tracklets)
    print(f"[INFO] Assigned initial global IDs: {len(global_tracklets)} tracklets → {len(unique_gids)} unique global IDs")
    return global_tracklets

def extend_global_tracklets(global_tracklets, object_coords, object_features, cam_folder):
    # Build lookup from (camera_id, local_id) to tracklet reference
    global_lookup = {(tr["camera_id"], tr["local_id"]): tr for tr in global_tracklets}

    extended_count = 0
    for sc_id, coords in object_coords.items():
        key = (cam_folder, sc_id)
        features = object_features.get(sc_id, [])

        if key in global_lookup:
            tr = global_lookup[key]
            tr["coords"].extend(coords)
            tr["features"].extend(features)
            extended_count += 1

    return global_tracklets, extended_count

def extend_global_tracklets_with_full_data_fixed(global_tracklets, data, cam_folder, args):
    """
    Extend global tracklets by copying the entire local tracklet (all frames) for matched (camera_id, sc_id).
    """
    global_lookup = {(tr["camera_id"], tr["local_id"]): tr for tr in global_tracklets}
    local_tracklets = {}

    for frame_str, objs in data.items():
        frame_id = int(frame_str)
        for obj in objs:
            sc_id = obj.get("object sc id")
            if sc_id is None:
                continue
            if sc_id not in local_tracklets:
                local_tracklets[sc_id] = []
            local_tracklets[sc_id].append((frame_id, obj))

    extended_count = 0

    for sc_id, frame_obj_list in local_tracklets.items():
        key = (cam_folder, sc_id)
        if key in global_lookup:
            tr = global_lookup[key]
            for frame_id, obj in frame_obj_list:
                if "3d location" in obj:
                    coord = obj["3d location"]

                    # Ensure all coords are in (frame_id, coord) format
                    if "coords" in tr:
                        if tr["coords"] and not isinstance(tr["coords"][0], tuple):
                            # Assume pre-glance coords match frame range 1...N
                            pre_coords = tr["coords"]
                            tr["coords"] = [(fid, c) for fid, c in zip(range(1, args.end_glance_frame + 1), pre_coords)]

                    tr.setdefault("coords", []).append((frame_id, coord))

            # Optional: deduplicate and sort coords by frame_id
            if "coords" in tr:
                dedup = {fid: coord for fid, coord in tr["coords"]}
                tr["coords"] = sorted(dedup.items())  # list of (frame_id, coord)

            for frame_id, obj in frame_obj_list:
                if "feature path" in obj:
                    tr.setdefault("feature_paths", []).append(obj["feature path"])

                if "object type" in obj and "object type" not in tr:
                    tr["object type"] = obj["object type"]

                for attr in ["3d bounding box scale", "3d bounding box rotation", "2d bounding box visible"]:
                    if attr in obj:
                        tr.setdefault(attr + "_sequence", []).append(obj[attr])

                # Always accumulate metadata
                for attr in ["3d bounding box scale", "3d bounding box rotation", "2d bounding box visible"]:
                    if attr in obj:
                        tr.setdefault(attr + "_sequence", []).append(obj[attr])

            extended_count += 1

    return global_tracklets, extended_count

def extract_coords_features_from_frames(data, cam_folder, frame_start):
    """
    Extract object coordinates and features from JSON data after a certain frame.

    Args:
        data: JSON dictionary loaded from fixed_<camera>.json
        cam_folder: camera folder name (for debug)
        frame_start: starting frame to include (e.g., args.end_glance_frame + 1)

    Returns:
        object_coords: dict of local_id → list of 3D locations
        object_features: dict of local_id → list of feature vectors
    """
    object_coords = {}
    object_features = {}

    for frame_str, objs in data.items():
        frame_id = int(frame_str)
        if frame_id < frame_start:
            continue

        for obj in objs:
            sc_id = obj.get("object sc id")
            coord = obj.get("3d location")
            feat_path = obj.get("feature path")

            if coord is not None:
                object_coords.setdefault(sc_id, []).append(coord)

            if feat_path and feat_path.lower().endswith(".npy"):
                try:
                    feat_full_path = os.path.join("EmbedFeature", feat_path)
                    feature = np.load(feat_full_path)
                    object_features.setdefault(sc_id, []).append(feature)
                except Exception as e:
                    print(f"[{cam_folder}] Failed to load feature for sc_id={sc_id} from {feat_path}: {e}")

    return object_coords, object_features

# def extract_unassigned_coords_features(data, cam_folder, frame_start, existing_keys):
#     """
#     Extracts coordinates and features of local tracklets that are NOT already assigned
#     to global IDs (based on camera_id, local_id), and only from post-glance frames.

#     Returns:
#         object_coords, object_features
#     """
#     object_coords = {}
#     object_features = {}

#     for frame_str, objs in data.items():
#         frame_id = int(frame_str)
#         if frame_id < frame_start:
#             continue

#         for obj in objs:
#             sc_id = obj["object sc id"]
#             key = (cam_folder, sc_id)
#             if key in existing_keys:
#                 continue  # Skip already assigned

#             coord = obj["3d location"]
#             feat_path = obj.get("feature path", None)

#             object_coords.setdefault(sc_id, []).append(coord)

#             if feat_path and feat_path.lower().endswith(".npy"):
#                 try:
#                     feat_full_path = os.path.join("EmbedFeature", feat_path)
#                     feature = np.load(feat_full_path)
#                     object_features.setdefault(sc_id, []).append(feature)
#                 except Exception as e:
#                     print(f"Warning: Failed to load feature for sc_id={sc_id} from {feat_path}: {e}")
#     # Final pass: warn if any sc_id is missing all features
#     for sc_id in object_coords:
#         if sc_id not in object_features or len(object_features[sc_id]) == 0:
#             print(f"Warning: No feature found for camera={cam_folder}, sc_id={sc_id} in post-glance frames")
#     return object_coords, object_features

def extract_unassigned_coords_features_with_frames(data, cam_folder, frame_start, existing_keys):
    """
    Extracts coordinates and features of local tracklets that are NOT already assigned
    to global IDs (based on camera_id, local_id), and only from post-glance frames.

    Returns:
        object_coords: dict of sc_id -> list of (frame_id, coord)
        object_features: dict of sc_id -> list of features
    """
    object_coords = {}
    object_features = {}

    for frame_str, objs in data.items():
        frame_id = int(frame_str)
        if frame_id < frame_start:
            continue

        for obj in objs:
            sc_id = obj["object sc id"]
            key = (cam_folder, sc_id)
            if key in existing_keys:
                continue  # Skip already assigned

            coord = obj["3d location"]
            feat_path = obj.get("feature path", None)

            object_coords.setdefault(sc_id, []).append((frame_id, coord))

            if feat_path and feat_path.lower().endswith(".npy"):
                try:
                    feat_full_path = os.path.join("EmbedFeature", feat_path)
                    feature = np.load(feat_full_path)
                    object_features.setdefault(sc_id, []).append(feature)
                except Exception as e:
                    print(f"Warning: Failed to load feature for sc_id={sc_id} from {feat_path}: {e}")

    # Final pass: warn if any sc_id is missing all features
    for sc_id in object_coords:
        if sc_id not in object_features or len(object_features[sc_id]) == 0:
            print(f"Warning: No feature found for camera={cam_folder}, sc_id={sc_id} in post-glance frames")

    return object_coords, object_features


def find_best_matching_global_id(coords_with_frames, features, cam_folder, global_tracklets, args, force_assign=False):
    best_gid = None
    best_traj_dist = float("inf")
    best_feat_sim = -1

    fallback_gid = None
    fallback_traj_dist = float("inf")
    fallback_feat_sim = -1

    for global_tr in global_tracklets:
        if global_tr["camera_id"] == cam_folder:
            continue  # Skip same camera

        g_coords_with_frames = global_tr.get("coords", [])
        if not g_coords_with_frames:
            continue

        # Convert to dicts for fast frame alignment
        coords_dict = dict(coords_with_frames)
        g_coords_dict = dict(g_coords_with_frames)
        shared_frames = sorted(set(coords_dict.keys()) & set(g_coords_dict.keys()))

        if not shared_frames:
            continue

        traj_diffs = [
            np.linalg.norm(np.array(coords_dict[f]) - np.array(g_coords_dict[f]))
            for f in shared_frames
        ]
        traj_sim = np.mean(traj_diffs)

        # Save best regardless of threshold
        if traj_sim < fallback_traj_dist:
            fallback_gid = global_tr["global_id"]
            fallback_traj_dist = traj_sim
            fallback_feat_sim = -1  # default if feature fails

        if traj_sim < args.trajectory_dist_thresh_rest:
            if not features or not global_tr.get("features"):
                # Accept based on trajectory only
                if traj_sim < best_traj_dist:
                    best_gid = global_tr["global_id"]
                    best_traj_dist = traj_sim
                    best_feat_sim = -1
            else:
                feat_sim = compute_feature_similarity(features, global_tr["features"], args)

                # Save fallback if better than last
                if traj_sim < fallback_traj_dist or (traj_sim == fallback_traj_dist and feat_sim > fallback_feat_sim):
                    fallback_gid = global_tr["global_id"]
                    fallback_traj_dist = traj_sim
                    fallback_feat_sim = feat_sim

                if feat_sim > args.feature_similarity_thresh_rest:
                    if traj_sim < best_traj_dist or (traj_sim == best_traj_dist and feat_sim > best_feat_sim):
                        best_gid = global_tr["global_id"]
                        best_traj_dist = traj_sim
                        best_feat_sim = feat_sim

    # Fallback if no threshold-qualified match found
    if best_gid is None and force_assign:
        best_gid = fallback_gid
        best_traj_dist = fallback_traj_dist
        best_feat_sim = fallback_feat_sim
        print(f"[FALLBACK] Assigned to closest global ID: local tracklet → global_id={best_gid} (trajectory distance={best_traj_dist:.3f}, feature similarity={best_feat_sim:.3f})")

    return best_gid, best_traj_dist, best_feat_sim