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
from tqdm import tqdm
from itertools import combinations

import sys
try:
    sys.path.append('BoT-SORT')
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

def to_serializable(val):
    if isinstance(val, np.ndarray):
        return val.tolist()
    if isinstance(val, (np.integer, np.floating)):
        return val.item()
    return val

def to_serializable(val):
    if isinstance(val, np.ndarray):
        return val.tolist()
    if isinstance(val, (np.integer, np.floating)):
        return val.item()
    return val

# Function to load prediction JSON
def load_predictions(json_file):
    with open(json_file, 'r') as f:
        return json.load(f)

# Function to draw bounding boxes on the image
def draw_bboxes_on_image(img, bboxes, color=(0, 255, 0), thickness=2):
    for bbox in bboxes:
        x1, y1, x2, y2 = map(int, bbox)  # Convert to integers
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    return img




def get_color_for_id(global_id):
    """
    Generate a unique color for each global ID using a hash function.
    This will ensure that the same global ID gets the same color across all frames.

    Args:
        global_id: The global ID for which to generate the color.

    Returns:
        A tuple (B, G, R) representing the color.
    """
    # Generate a hash value from the global ID and use it to derive a unique color
    import hashlib

    hash_value = hashlib.md5(str(global_id).encode()).hexdigest()
    r = int(hash_value[0:2], 16)  # Red component
    g = int(hash_value[2:4], 16)  # Green component
    b = int(hash_value[4:6], 16)  # Blue component
    
    # Return as a BGR tuple (OpenCV uses BGR color format)
    return (b, g, r)  # OpenCV uses BGR format

def draw_bboxes_on_image(img, bboxes, global_ids, color_map, thickness=2):
    """
    Draw bounding boxes on the image, with local and global IDs below the boxes.
    
    Args:
        img: The image to draw on.
        bboxes: List of bounding boxes to draw (in [x1, y1, x2, y2] format).
        global_ids: List of global IDs corresponding to each bounding box.
        color_map: A dictionary mapping global IDs to unique colors.
        thickness: Thickness of the bounding box (default is 2).
        
    Returns:
        img: The image with bounding boxes drawn.
    """
    for i, bbox in enumerate(bboxes):
        x1, y1, x2, y2 = map(int, bbox)  # Convert to integers
        w = x2 - x1
        h = y2 - y1

        # Get the color for the current global ID
        color = color_map[global_ids[i]]

        # Draw the bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

        # Text positions: slightly below the bounding box
        text_y_offset = 10  # Space below the box
        
        # Draw global ID text below the box
        cv2.putText(img, f"G{global_ids[i]}", (x1, y1 + text_y_offset + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return img

def process_frame(predictions, frame_id, camera_id, frame_folder, output_folder, color_map):
    # Get the frame's prediction data for that frame
    frame_data = predictions.get(str(frame_id), [])
    
    # Initialize lists to store bounding boxes and global ids
    bboxes = []
    global_ids = []
    
    for obj in frame_data:
        # Check if the camera ID exists in this object's 2D bounding box data
        if camera_id in obj["2d bounding box visible"]:
            bbox = obj["2d bounding box visible"][camera_id][camera_id]  # Get the bounding box coordinates
            
            if bbox:
                bboxes.append(bbox)  # Add to list of bounding boxes for this frame
                global_ids.append(obj["object id"])  # Add the global ID

    # Path to the image for this frame
    img_path = os.path.join(frame_folder, f"{frame_id:06d}.jpg")
    
    if os.path.exists(img_path):
        # Read the image
        img = cv2.imread(img_path)
        
        # Draw the bounding boxes and IDs on the image
        img = draw_bboxes_on_image(img, bboxes, global_ids, color_map)
        
        # Output path
        output_path = os.path.join(output_folder, f"{frame_id:06d}.jpg")
        
        # Save the modified image
        cv2.imwrite(output_path, img)
    else:
        print(f"Image for frame {frame_id} does not exist.")

# Main function to process all frames and objects
def process_all_frames(predictions, base_video_path, output_vis_root, camera_ids, args):
    # Make sure the output folder exists
    os.makedirs(output_vis_root, exist_ok=True)

    # Create a color map for each global ID
    color_map = {}

    # Iterate through all the frames and cameras to generate the color map
    num_save = args.frame_num_save * 50 +1
    for camera_id in camera_ids:
        # Get the frame folder for this camera
        cam_folder = os.path.join(base_video_path, camera_id, "Frame")
        output_folder = os.path.join(output_vis_root, camera_id)
        os.makedirs(output_folder, exist_ok=True)
        
        # Iterate through the frames in the prediction data with intervals (e.g., every 50 frames)
        for frame_id in range(1, num_save, 50):  # Example: Process every 50th frame
            frame_data = predictions.get(str(frame_id), [])
            for obj in frame_data:
                global_id = obj["object id"]
                if global_id not in color_map:
                    color_map[global_id] = get_color_for_id(global_id)

    # Now process the selected frames for all cameras
    for camera_id in camera_ids:
        cam_folder = os.path.join(base_video_path, camera_id, "Frame")
        output_folder = os.path.join(output_vis_root, camera_id)
        os.makedirs(output_folder, exist_ok=True)
        
        # Process frames at regular intervals (every 50 frames)
        for frame_id in range(1, num_save, 50):  # Process every 50th frame
            process_frame(predictions, frame_id, camera_id, cam_folder, output_folder, color_map)

def convert_to_framewise_multicam_dict(assigned_global_map, max_frame=200):
    # Temporary storage per (frame, global_id)
    temp_data = {}

    for (cam_id, local_id), entry in assigned_global_map.items():
        obj = entry["object"]
        global_id = entry["global_id"]
        obj_type = obj.get("object type", "unknown")

        coords = obj.get("coords", [])  # list of (frame_id, [x,y,z])
        bbox_2d_seq = obj.get("2d bounding box visible_sequence", [])  # list of [x1,y1,x2,y2]
        scale_seq = obj.get("3d bounding box scale_sequence", [])
        rot_seq = obj.get("3d bounding box rotation_sequence", [])
        if cam_id == "Camera_11" and local_id ==4:
                z=2
        for idx, (frame_id, center) in enumerate(coords):
            if frame_id >= max_frame:
                continue  # Skip frames after max_frame
            
            if idx >= len(bbox_2d_seq) or idx >= len(scale_seq) or idx >= len(rot_seq):
                continue  # Skip if sequences are not of the same length

            # Get the bounding box, scale, and rotation for this frame
            bbox = bbox_2d_seq[idx]
            scale = scale_seq[idx]
            rot = rot_seq[idx]

            # Initialize frame data if it doesn't exist
            if frame_id not in temp_data:
                temp_data[frame_id] = {}

            # Store data for this global_id at this frame_id
            if cam_id == "Camera_11" and frame_id ==101:
                z=2
            if global_id not in temp_data[frame_id]:
                temp_data[frame_id][global_id] = {
                    "object type": obj_type,
                    "object id": global_id,
                    "centers": [],
                    "scales": [],
                    "rots": [],
                    "2d bounding box visible": {}
                }

            item = temp_data[frame_id][global_id]
            item["centers"].append(np.array(center, dtype=np.float32))
            item["scales"].append(np.array(scale, dtype=np.float32))
            item["rots"].append(np.array(rot, dtype=np.float32))
            item["2d bounding box visible"][cam_id] = to_serializable(bbox)

    # Now average and convert to final structure
    final_output = {}

    for frame_id, global_map in temp_data.items():
        frame_list = []
        if frame_id == 101:
            z=2
        for global_id, item in global_map.items():
            center_avg = np.mean(item["centers"], axis=0)
            scale_avg = np.mean(item["scales"], axis=0)
            rot_avg = np.mean(item["rots"], axis=0)

            out_obj = {
                "object type": item["object type"],
                "object id": item["object id"],
                "3d location": to_serializable(center_avg),
                "3d bounding box scale": to_serializable(scale_avg),
                "3d bounding box rotation": to_serializable(rot_avg),
                "2d bounding box visible": item["2d bounding box visible"]
            }
            frame_list.append(out_obj)

        final_output[str(frame_id)] = frame_list

    return final_output

def write_framewise_txt(scene_path, assigned_global_map, max_frame=200):
    from collections import defaultdict
    import numpy as np

    CLASS_NAME_TO_ID = {
        "Person": 0,
        "Forklift": 1,
        "NovaCarter": 2,
        "Transporter": 3,
        "FourierGR1T2": 4,
        "AgilityDigit": 5
    }

    temp_data = defaultdict(lambda: defaultdict(lambda: {
        "scene_id": None,
        "class_id": None,
        "global_id": None,
        "centers": [],
        "scales": [],
        "rots": [],
    }))

    for (cam_id, local_id), entry in assigned_global_map.items():
        obj = entry["object"]
        global_id = entry["global_id"]

        scene_str = obj.get("scene_id", "0")
        scene_id = int(scene_str.split('_')[-1])  # e.g., "Warehouse_016" → 16
        class_name = obj.get("object type", "Unknown")
        class_id = CLASS_NAME_TO_ID.get(class_name, -1)

        coords = obj.get("coords", [])  # list of (frame_id, [x,y,z])
        scale_seq = obj.get("3d bounding box scale_sequence", [])
        rot_seq = obj.get("3d bounding box rotation_sequence", [])
        if cam_id == "Camera_11" and local_id == 4:
            z=2
        for idx, (frame_id, center) in enumerate(coords):
            if frame_id >= max_frame:
                continue
            if idx >= len(scale_seq) or idx >= len(rot_seq):
                continue

            temp = temp_data[frame_id][global_id]
            temp["scene_id"] = scene_id
            temp["class_id"] = class_id
            temp["global_id"] = global_id
            temp["centers"].append(np.array(center, dtype=np.float32))
            temp["scales"].append(np.array(scale_seq[idx], dtype=np.float32))
            temp["rots"].append(np.array(rot_seq[idx], dtype=np.float32))

    # Write output
    output_dir = os.path.join(scene_path, "output_result")
    os.makedirs(output_dir, exist_ok=True)

    save_path = os.path.join(output_dir, f"final_track_{os.path.basename(scene_path)}.txt")
    with open(save_path, "w") as f:
        for frame_id in sorted(temp_data.keys()):
            for global_id, info in temp_data[frame_id].items():
                if len(info["centers"]) == 0:
                    continue
                avg_center = np.mean(info["centers"], axis=0)
                avg_scale = np.mean(info["scales"], axis=0)
                avg_rot = np.mean(info["rots"], axis=0)

                x, y, z = avg_center.tolist()
                w, l, h = avg_scale.tolist()
                yaw = float(avg_rot[2])  # Use yaw only

                line = f"{info['scene_id']} {info['class_id']} {global_id} {frame_id} {x:.6f} {y:.6f} {z:.6f} {w:.6f} {l:.6f} {h:.6f} {yaw:.6f}\n"
                f.write(line)

    print(f"✅ Saved track1-style .txt to: {save_path}")

def write_framewise_txt_from_json(scene_path, args, framewise_json):
    """
    Convert framewise_json (post-processed multi-camera tracking results) into a .txt file.

    Args:
        scene_path (str): The path to store the output file.
        framewise_json (dict): The framewise multi-camera tracking data (output of post-processing).
        max_frame (int): The maximum number of frames to process.
    """
    from collections import defaultdict
    import numpy as np

    # You might need a class mapping (this depends on your object types)
    CLASS_NAME_TO_ID = {
        "Person": 0,
        "Forklift": 1,
        "NovaCarter": 2,
        "Transporter": 3,
        "FourierGR1T2": 4,
        "AgilityDigit": 5
    }

    # Temporary storage for frame-wise data (global_id -> frame_id -> object details)
    temp_data = defaultdict(lambda: defaultdict(lambda: {
        "scene_id": None,
        "class_id": None,
        "global_id": None,
        "centers": [],
        "scales": [],
        "rots": [],
    }))

    # Loop through the framewise_json data and process it
    for frame_id, objects in framewise_json.items():
        for obj in objects:
            global_id = obj["object id"]
            scene_str = obj.get("scene_id", "0")
            scene_id = int(scene_str.split('_')[-1])  # Extract scene_id (e.g., Warehouse_016 -> 16)
            class_name = obj.get("object type", "Unknown")
            class_id = CLASS_NAME_TO_ID.get(class_name, -1)

            coords = obj.get("3d location", [])  # List of [x, y, z]
            scale_seq = obj.get("3d bounding box scale", [])
            rot_seq = obj.get("3d bounding box rotation", [])



            # Add object data to temporary storage
            temp = temp_data[int(frame_id)][global_id]
            temp["scene_id"] = scene_id
            temp["class_id"] = class_id
            temp["global_id"] = global_id
            temp["centers"].append(np.array(coords, dtype=np.float32))
            temp["scales"].append(np.array(scale_seq, dtype=np.float32))
            temp["rots"].append(np.array(rot_seq, dtype=np.float32))

    # Write output to a .txt file
    output_dir = os.path.join(scene_path, "output_result")
    os.makedirs(output_dir, exist_ok=True)

    unique_global_ids = set()
    save_path = os.path.join(output_dir, f"final_track_{args.exp_path}_fix.txt")
    with open(save_path, "w") as f:
        for frame_id in sorted(temp_data.keys()):
            for global_id, info in temp_data[frame_id].items():
                if len(info["centers"]) == 0:
                    continue
                # Filter out empty arrays before calculating the mean
                valid_centers = [center for center in info["centers"] if center.size > 0]
                valid_scales = [scale for scale in info["scales"] if scale.size > 0]
                valid_rots = [rot for rot in info["rots"] if rot.size > 0]
                # Check if there are valid centers left after filtering
                if valid_centers:
                    avg_center = np.mean(valid_centers, axis=0)
                else:
                    avg_center = np.zeros(3)  # or some other default value, depending on your use case
                if valid_scales:
                    avg_scale = np.mean(valid_scales, axis=0)
                else:
                    avg_scale = np.zeros(3)  # or some other default value

                if valid_rots:
                    avg_rot = np.mean(valid_rots, axis=0)
                else:
                    avg_rot = np.zeros(3)  # or some other default value

                x, y, z = avg_center.tolist()
                w, l, h = avg_scale.tolist()
                yaw = float(avg_rot[2])  # Use yaw only

                line = f"{info['scene_id']} {info['class_id']} {global_id} {frame_id} {x:.6f} {y:.6f} {z:.6f} {w:.6f} {l:.6f} {h:.6f} {yaw:.6f}\n"
                f.write(line)
                unique_global_ids.add(global_id)
    print(f"[INFO] Total number of unique global IDs: {len(unique_global_ids)}")
    print(f"✅ Saved track1-style .txt to: {save_path}")

def write_ground_truth_txt_from_json(scene_path, args, ground_truth_json):
    """
    Convert ground truth data into a .txt file (similar to framewise JSON).

    Args:
        scene_path (str): The path to store the output file.
        ground_truth_json (dict): The ground truth data in a similar format to framewise_json.
        max_frame (int): The maximum number of frames to process.
    """
    from collections import defaultdict
    import numpy as np

    # You might need a class mapping (this depends on your object types)
    CLASS_NAME_TO_ID = {
        "Person": 0,
        "Forklift": 1,
        "NovaCarter": 2,
        "Transporter": 3,
        "FourierGR1T2": 4,
        "AgilityDigit": 5
    }

    # Temporary storage for frame-wise data (global_id -> frame_id -> object details)
    temp_data = defaultdict(lambda: defaultdict(lambda: {
        "scene_id": None,
        "class_id": None,
        "global_id": None,
        "centers": [],
        "scales": [],
        "rots": [],
    }))

    # Loop through the ground_truth_json data and process it
    for frame_id, objects in ground_truth_json.items():
        for obj in objects:
            global_id = obj["object id"]
            scene_str = obj.get("scene_id", "0")
            scene_id = int(scene_str.split('_')[-1])  # Extract scene_id (e.g., Warehouse_016 -> 16)
            class_name = obj.get("object type", "Unknown")
            class_id = CLASS_NAME_TO_ID.get(class_name, -1)

            coords = obj.get("3d location", [])  # List of [x, y, z]
            scale_seq = obj.get("3d bounding box scale", [])
            rot_seq = obj.get("3d bounding box rotation", [])

            # Add object data to temporary storage
            temp = temp_data[int(frame_id)][global_id]
            temp["scene_id"] = scene_id
            temp["class_id"] = class_id
            temp["global_id"] = global_id
            temp["centers"].append(np.array(coords, dtype=np.float32))
            temp["scales"].append(np.array(scale_seq, dtype=np.float32))
            temp["rots"].append(np.array(rot_seq, dtype=np.float32))

    # Write output to a .txt file (ground_truth.txt)
    output_dir = os.path.join(scene_path, "output_result")
    os.makedirs(output_dir, exist_ok=True)

    unique_global_ids = set()
    save_path = os.path.join(output_dir, f"ground_truth_{args.scene_id}.txt")
    with open(save_path, "w") as f:
        for frame_id in sorted(temp_data.keys()):
            for global_id, info in temp_data[frame_id].items():
                if len(info["centers"]) == 0:
                    continue
                # Filter out empty arrays before calculating the mean
                valid_centers = [center for center in info["centers"] if center.size > 0]
                valid_scales = [scale for scale in info["scales"] if scale.size > 0]
                valid_rots = [rot for rot in info["rots"] if rot.size > 0]
                # Check if there are valid centers left after filtering
                if valid_centers:
                    avg_center = np.mean(valid_centers, axis=0)
                else:
                    avg_center = np.zeros(3)  # or some other default value, depending on your use case
                if valid_scales:
                    avg_scale = np.mean(valid_scales, axis=0)
                else:
                    avg_scale = np.zeros(3)  # or some other default value

                if valid_rots:
                    avg_rot = np.mean(valid_rots, axis=0)
                else:
                    avg_rot = np.zeros(3)  # or some other default value

                x, y, z = avg_center.tolist()
                w, l, h = avg_scale.tolist()
                yaw = float(avg_rot[2])  # Use yaw only

                line = f"{info['scene_id']} {info['class_id']} {global_id} {frame_id} {x:.6f} {y:.6f} {0.0:.6f} {w:.6f} {l:.6f} {h:.6f} {yaw:.6f}\n"
                f.write(line)
                unique_global_ids.add(global_id)
    print(f"[INFO] Total number of unique global IDs: {len(unique_global_ids)}")

    print(f"✅ Saved ground_truth-style .txt to: {save_path}")  

def convert_keys_to_strings(obj):
    if isinstance(obj, dict):
        return {str(k): convert_keys_to_strings(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_keys_to_strings(i) for i in obj]
    else:
        return obj
    

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

def visualize_fixed_tracklets(data, image_root, camera_id, save_root, max_frames=500):
    os.makedirs(save_root, exist_ok=True)
    frame_ids = sorted(map(int, data.keys()))[:max_frames]

    for frame_id in frame_ids:
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
                # print(f"[WARN] {cam_folder} sc_id={sc_id} has no dense cluster → fallback to first feature")
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
    global_id_map = {}  # Map of global ID to list of (camera_id, local_id)

    for cam_index, tracklets in enumerate(all_tracklets_by_camera):
        local_used_gids = set()  # Prevent duplicate global IDs in the same camera
        for tracklet in tracklets:
            matched_gid = None
            best_traj_dist = float('inf')
            best_feat_sim = -1
            best_camera_id = ""
            best_local_id = -1

            # Track the best match based on trajectory similarity
            best_traj_match = None

            for global_tr in global_tracklets:
                g_id = global_tr["global_id"]
                
                # Check if the global_id has already been assigned to the same camera view
                if any(cam_id == tracklet["camera_id"] for cam_id, _ in global_id_map.get(g_id, [])):
                    print(f"[DEBUG] Skipping global ID {g_id} for (cam={tracklet['camera_id']}, local_id={tracklet['local_id']}) because it is already assigned to this camera.")
                    continue  # Skip if the global_id is already assigned to the same camera view


                if is_trajectory_similar(tracklet["coords"], global_tr["coords"], threshold=args.trajectory_dist_thresh_init):
                    # If the features are also similar, immediately assign the global ID
                    if is_feature_similar(tracklet["features"], global_tr["features"], sim_thresh=args.feature_similarity_thresh_init, metric=args.metric):
                        matched_gid = g_id
                        print(f"[DEBUG] Feature match found as well, assigning global ID {g_id} to (cam={tracklet['camera_id']}, local_id={tracklet['local_id']}) "
                              f"based on both feature and trajectory similarity.")
                        break  # Exit if both conditions are satisfied

                    # If only trajectory is similar, store this match as a potential best match
                    if matched_gid is None:
                        if best_traj_dist > args.trajectory_dist_thresh_init:
                            best_traj_dist = args.trajectory_dist_thresh_init
                            best_feat_sim = -1  # Not considering feature similarity yet
                            best_traj_match = g_id
                            best_camera_id = global_tr['camera_id']
                            best_local_id = global_tr['local_id']

            # If a match was found where only trajectory similarity holds
            if matched_gid is None and best_traj_match is not None:
                matched_gid = best_traj_match  # Assign the best trajectory match
                print(f"[DEBUG] No feature match found, assigning global ID {matched_gid} of (cam={best_camera_id}, local_id ={best_local_id}, global_id={g_id}) to (cam={tracklet['camera_id']}, local_id={tracklet['local_id']}) "
                      f"based on trajectory similarity only.")

            if matched_gid is None:  # If no matches were found, assign a new global ID
                matched_gid = global_id_counter
                global_id_counter += 1
                print(f"[DEBUG] No match found, assigning new global ID {matched_gid} to (cam={tracklet['camera_id']}, local_id={tracklet['local_id']}).")

            # Assign the global ID to the tracklet
            tracklet["global_id"] = matched_gid
            local_used_gids.add(matched_gid)
            global_tracklets.append(tracklet)
             # Update the global_id_map with the new match
            if matched_gid not in global_id_map:
                global_id_map[matched_gid] = []
            global_id_map[matched_gid].append((tracklet["camera_id"], tracklet["local_id"]))

    unique_gids = set(t["global_id"] for t in global_tracklets)
    print(f"[INFO] Assigned initial global IDs: {len(global_tracklets)} tracklets → {len(unique_gids)} unique global IDs")
    # Expressively print the global_id_map with (camera_id, local_id) pairs
    print("[INFO] Global ID Map (Global ID -> Assigned (camera_id, local_id)):")
    for global_id, assigned_ids in global_id_map.items():
        print(f"Global ID {global_id}:")
        for cam_id, local_id in assigned_ids:
            print(f"  - Camera: {cam_id}, Local ID: {local_id}")
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
    Extend global tracklets by copying the entire local tracklet (all frames) for matched (camera_id, local_id).
    """
    global_lookup = {(tr["camera_id"], tr["local_id"]): tr for tr in global_tracklets}
    local_tracklets = {}

    for frame_str, objs in data.items():
        frame_id = int(frame_str)
        for obj in objs:
            sc_id = obj.get("object sc id")
            if sc_id is None:
                continue
            local_tracklets.setdefault(sc_id, []).append((frame_id, obj))

    extended_count = 0
    for sc_id, frame_obj_list in local_tracklets.items():
        key = (cam_folder, sc_id)
        if key not in global_lookup:
            continue

        tr = global_lookup[key]
        added_coords = []
        feature_paths = []
        scale_seq, rot_seq, vis_seq = [], [], []

        for frame_id, obj in frame_obj_list:
            # Coordinates
            if "3d location" in obj:
                coord = obj["3d location"]
                added_coords.append((frame_id, coord))

            # Feature paths
            if "feature path" in obj:
                feature_paths.append(obj["feature path"])

            # Sequences
            if "3d bounding box scale" in obj:
                scale_seq.append(obj["3d bounding box scale"])
            if "3d bounding box rotation" in obj:
                rot_seq.append(obj["3d bounding box rotation"])
            if "2d bounding box visible" in obj:
                vis_seq.append(obj["2d bounding box visible"])

            # Object type (once)
            if "object type" in obj and "object type" not in tr:
                tr["object type"] = obj["object type"]

        # Initialize or extend
        tr.setdefault("coords", [])
        if tr["coords"] and not isinstance(tr["coords"][0], tuple):
            tr["coords"] = [(fid, c) for fid, c in zip(range(1, args.end_glance_frame + 1), tr["coords"])]

        tr["coords"].extend(added_coords)
        tr["coords"] = sorted({fid: c for fid, c in tr["coords"]}.items())  # Dedup by frame_id

        tr.setdefault("feature_paths", []).extend(feature_paths)
        tr.setdefault("3d bounding box scale_sequence", []).extend(scale_seq)
        tr.setdefault("3d bounding box rotation_sequence", []).extend(rot_seq)
        tr.setdefault("2d bounding box visible_sequence", []).extend(vis_seq)

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

def trajectory_dict_distance(coords1, coords2):
    """
    Computes the average Euclidean distance between two trajectories.
    Each coords input is a list of (frame_id, coord), where coord is a 3D point [x, y, z].
    """
    if not coords1 or not coords2:
        return float('inf')

    # Ensure that coords1 and coords2 are lists of (frame_id, [x, y, z]) tuples
    # If coords1 and coords2 are just lists of [x, y, z], associate them with frame_ids
    if isinstance(coords1[0], list):  # If coords1 is a list of coordinates [x, y, z]
        coords1 = [(i, c) for i, c in enumerate(coords1)]  # Assign each coordinate a frame_id
    if isinstance(coords2[0], list):  # If coords2 is a list of coordinates [x, y, z]
        coords2 = [(i, c) for i, c in enumerate(coords2)]  # Assign each coordinate a frame_id

    # Convert the list of (frame_id, coord) tuples to a dict where key = frame_id, value = coord
    dict1 = dict(coords1)  # coords1 should be [(frame_id, [x, y, z])]
    dict2 = dict(coords2)  # coords2 should be [(frame_id, [x, y, z])]

    # Find common frames
    common_frames = sorted(set(dict1.keys()) & set(dict2.keys()))
    if not common_frames:
        return float('inf')

    # Compute the average Euclidean distance between corresponding coordinates in common frames
    distances = []
    for frame in common_frames:
        coord1 = dict1[frame]
        coord2 = dict2[frame]
        distance = np.linalg.norm(np.array(coord1) - np.array(coord2))
        distances.append(distance)

    # Return the average distance over all common frames
    return np.mean(distances)


def trajectory_distance(coords1, coords2):
    """
    Computes the average Euclidean distance between two trajectories.
    Each coords input is a list of (frame_id, coord), where coord is a 3D point [x, y, z].
    """
    if not coords1 or not coords2:
        return float('inf')

    # Convert to dict for frame-wise alignment
    dict1 = dict(coords1)
    dict2 = dict(coords2)

    # Find common frames
    common_frames = sorted(set(dict1.keys()) & set(dict2.keys()))
    if not common_frames:
        return float('inf')

    # Compute average distance over common frames
    distances = [
        np.linalg.norm(np.array(dict1[f]) - np.array(dict2[f]))
        for f in common_frames
    ]
    return np.mean(distances)

def feature_similarity(f1, f2, metric="cosine", topk=10):
    """
    Computes similarity score between two sets of features.
    Returns average of top-k pairwise similarities.

    f1, f2: list of 1D feature vectors
    metric: 'cosine' or 'euclidean'
    topk: number of best-matching pairs to average
    """
    if len(f1) == 0 or len(f2) == 0:
        return 0.0

    f1 = np.stack(f1)
    f2 = np.stack(f2)

    # Normalize for cosine similarity
    if metric == "cosine":
        f1 = normalize(f1, norm='l2')
        f2 = normalize(f2, norm='l2')
        sim_matrix = cosine_similarity(f1, f2)
    elif metric == "euclidean":
        dist_matrix = euclidean_distances(f1, f2)
        sim_matrix = -dist_matrix  # negative = similarity proxy
    else:
        raise ValueError("Unsupported metric. Use 'cosine' or 'euclidean'.")

    # Flatten and select top-k similarities
    sim_values = sim_matrix.flatten()
    if len(sim_values) == 0:
        return 0.0

    topk = min(topk, len(sim_values))
    top_sim = np.partition(sim_values, -topk)[-topk:]
    return float(np.mean(top_sim))


def load_and_get_representative_features(feature_paths, sc_id, cam_folder, args):
    """
    Load features from file paths and return a representative subset using clustering or similarity.
    """
    features = []
    for path in feature_paths:
        if not path:
            # print(f"[WARN] Empty or None feature path encountered. Skipping.")
            continue

        full_path = os.path.join("EmbedFeature", path)
        if not os.path.exists(full_path):
            # print(f"[WARN] Feature file not found: {full_path}")
            continue

        try:
            feat = np.load(full_path)
            features.append(feat)
        except Exception as e:
            # print(f"[WARN] Failed to load feature from {full_path}: {e}")
            continue

    if len(features) == 0:
        return []

    feats = np.stack(features)
    representitive_feature_len = args.representitive_feature_len
    metric = args.metric

    if len(features) == 1:
        return features

    if metric == "cosine":
        dot_products = np.einsum('ij,ij->i', feats, feats)
        is_normalized = np.allclose(dot_products, 1.0, atol=1e-3)

        if not is_normalized:
            feats_norm = normalize(feats, norm='l2')
        else:
            feats_norm = feats

        clustering = DBSCAN(eps=0.2, min_samples=3, metric='cosine').fit(feats_norm)
        labels = clustering.labels_

        unique_labels, counts = np.unique(labels[labels >= 0], return_counts=True)
        if len(unique_labels) == 0:
            # print(f"[WARN] {cam_folder} sc_id={sc_id} has no dense cluster → fallback to first feature")
            return [features[0]]

        main_cluster = unique_labels[np.argmax(counts)]
        inlier_indices = np.where(labels == main_cluster)[0]
        selected_indices = inlier_indices[:representitive_feature_len]

        return [features[i] for i in selected_indices]

    elif metric == "euclidean":
        sim_matrix = -euclidean_distances(feats)
        mean_sim = np.sum(sim_matrix, axis=1) - np.diag(sim_matrix)
        mean_sim /= (len(features) - 1)

        sorted_indices = np.argsort(-mean_sim)
        top_indices = sorted_indices[:min(representitive_feature_len, len(features))]
        return [features[i] for i in top_indices]

    else:
        raise ValueError("Unsupported metric. Use 'cosine' or 'euclidean'.")

def split_global_map_by_completion(assigned_global_map, args):
    """
    Split global tracklets into completed and active based on whether the global ID spans to the last frame.
    """
  

    global_id_to_coords = defaultdict(list)

    # Step 1: Group full objects by global_id
    global_id_to_objects = defaultdict(list)
    for key, gtrack in assigned_global_map.items():
        global_id = gtrack["global_id"]
        global_id_to_objects[global_id].append(gtrack["object"])

    # Step 2: For each global_id, find max frame over all objects that share this global ID
    global_id_to_max_frame = {}
    for global_id, obj_list in global_id_to_objects.items():
        max_frame = -1
        for obj in obj_list:
            coords = obj.get("coords", [])
            frames = [fid for fid, _ in coords]
            if frames:
                max_frame = max(max_frame, max(frames))
        global_id_to_max_frame[global_id] = max_frame

    # Step 3: Split the assigned_global_map into completed and active
    completed = {}
    active = {}
    for key, gtrack in assigned_global_map.items():
        global_id = gtrack["global_id"]
        if global_id_to_max_frame.get(global_id, 0) >= args.total_frames:
            completed[key] = gtrack
        else:
            active[key] = gtrack

    print(f"[SPLIT] Completed global tracklets: {len(completed)}")
    print(f"[SPLIT] Active (incomplete) global tracklets: {len(active)}")
    return completed, active




def post_process_multiple_camera_tracking(args):
    # Load the previously saved multi-camera tracking JSON
    import time
    input_json_path = f"Tracking/Multicamera/{args.scene_id}/{args.exp_path}/output_result/final_track_{args.scene_id}.json"
    
    if not os.path.exists(input_json_path):
        print(f"[Error] File not found: {input_json_path}")
        return
    
    # Load the JSON data
    with open(input_json_path, "r") as f:
        framewise_json = json.load(f)

    # Apply NMS (Non-Maximum Suppression) to merge tracklets based on temporal and spatial overlap
    print(f"[INFO] Running NMS for multi-camera tracking...")
    start_time = time.time()
    
    # Apply NMS for multi-camera tracking
    framewise_json = multi_camera_tracklets_non_maximum_suppression(framewise_json, args)
    
    elapsed = time.time() - start_time
    print(f"[NMS PROCESS] Completed for multi-camera tracking.")
    print(f"[NMS PROCESS] Time taken: {elapsed:.2f} seconds")

    # After applying NMS, save the results in a new JSON file
    output_result_dir = os.path.join(args.exp_path, "output_result")
    os.makedirs(output_result_dir, exist_ok=True)

    save_path =  f"Tracking/Multicamera/{args.scene_id}/{args.exp_path}/output_result/final_track_{args.scene_id}_fix.json"
    with open(save_path, "w") as f:
        json.dump(framewise_json, f, indent=2)
    print(f"[INFO] Saved NMS processed multi-camera tracklets to: {save_path}")

    return framewise_json

def get_final_new_id(new_id, merged):
    """
    Recursively find the final new_id for a given old_id by following the merge mapping.
    This ensures that all references to old_id are replaced with the final new_id.
    If the old_id is a value in merged, replace it with the most recent new_id.
    """
    # Iterate through the merged dictionary to check if any value is equal to old_id
    for key, value in merged.items():
        if value == new_id:
            # Update the merged dictionary to reflect the change in key-value pair
            merged[key] = merged.get(value, value)
            # Recurse until we find the final new_id
            # get_final_new_id(key, merged)
            
def multi_camera_tracklets_non_maximum_suppression(framewise_json, args):
    """
    Apply NMS to multi-camera tracking results.
    The method merges tracklets based on both spatial and temporal overlap.
    """
    temporal_iou_thresh = args.temporal_iou_thresh
    spatial_iou_thresh = args.spatial_iou_thresh
    remove_short_tracks = args.remove_short_tracks
    min_track_len = args.min_track_len

    track_by_id = defaultdict(list)

    # Group tracklets by global ID and camera
    for frame_id, objects in framewise_json.items():
        for obj in objects:
            global_id = obj["object id"]
            track_by_id[global_id].append((int(frame_id), obj))

    print(f"[INFO] Number of tracklets before NMS: {len(track_by_id)}")

    # Remove short tracks
    if remove_short_tracks:
        track_by_id = {tid: track for tid, track in track_by_id.items() if len(track) >= min_track_len}

    track_ids = sorted(track_by_id.keys())
    merged = {}

    # Compare tracklets for overlap
    for id1, id2 in combinations(track_ids, 2):
        track1 = track_by_id[id1]
        track2 = track_by_id[id2]

        f1 = set(f for f, _ in track1)
        f2 = set(f for f, _ in track2)
        common_frames = f1 & f2

        if not common_frames:
            continue
        
        # Debug
        _, obj1 = track1[0]
        _, obj2 = track2[0]
        if obj1["object id"] == 4 and obj2["object id"] == 13 :
            z = 2
        #
        temporal_overlap = len(common_frames) / min(len(f1), len(f2))
        if temporal_overlap < temporal_iou_thresh:
            continue
        

        # Check trajectory similarity instead of IoU
        traj_dist = trajectory_dict_distance([obj["3d location"] for _, obj in track1], [obj["3d location"] for _, obj in track2])
        
        # If trajectory distance is low enough, consider them as similar
        if traj_dist < args.trajectory_dist_thresh:  # You can adjust this threshold based on your data
            if obj1["object id"] == 13 or obj2["object id"] == 13 :
                z = 2
            # Keep the longer tracklet
            if len(track2) > len(track1):
                id1, id2 = id2, id1
                track1, track2 = track2, track1
            # Merge id2 into id1
            merged[id2] = id1
            get_final_new_id(id1, merged)

    # Apply merging to the tracklets
    for old_id, new_id in merged.items():
        if old_id in track_by_id:
            # Pop the old track and get its frames
            old_track = track_by_id.pop(old_id)
            
            # Create a dictionary for fast access to frames in the new tracklet
            new_track_dict = {f: obj for f, obj in track_by_id[new_id]}
            
            # Add frames from the old track to the new track (both shared and non-shared frames)
            for f, obj in old_track:
                # If the frame is not already in the new tracklet
                if f not in new_track_dict:
                    track_by_id[new_id].append((f, obj))
                
                # Now merge the visibility data from the old tracklet into the new tracklet
                if "2d bounding box visible" in obj:
                    # Merge visibility information from the old tracklet into the new tracklet
                    for cam_id, bbox in obj["2d bounding box visible"].items():
                        if f in new_track_dict:
                            # If the frame is in the new tracklet, update its visibility data
                            new_frame_obj = new_track_dict[f]
                            if "2d bounding box visible" not in new_frame_obj:
                                new_frame_obj["2d bounding box visible"] = {}
                            new_frame_obj["2d bounding box visible"][cam_id] = bbox
                        else:
                            # If the frame doesn't exist in the new tracklet, we need to add it
                            new_frame_obj = { 
                                "2d bounding box visible": {cam_id: bbox} 
                            }
                            track_by_id[new_id].append((f, new_frame_obj))

    # Rebuild the data with new merged IDs
    new_data = defaultdict(list)
    for new_id, track in enumerate(track_by_id.values(), 1):
        for frame_id, obj in track:
            obj["object id"] = new_id
            new_data[str(frame_id)].append(obj)

    print(f"[INFO] Number of tracklets after NMS: {len(track_by_id)}")
    
    return new_data


def global_matching(
    assigned_global_map,
    unassigned_local_map,
    args,
    force_match=False,
    relax_directional_check=False,
):
    from collections import defaultdict
    from tqdm import tqdm

    # Build global frame ranges
    global_frame_ranges = {}
    for (cam_id, local_id), gtrack in assigned_global_map.items():
        gcoords = gtrack["object"]["coords"]
        if not gcoords:
            continue
        global_frame_ranges[(cam_id, local_id)] = (gcoords[0][0], gcoords[-1][0])

    global_id_to_frames = defaultdict(list)
    for gtrack in assigned_global_map.values():
        global_id = gtrack["global_id"]
        coords = gtrack["object"].get("coords", [])
        frame_ids = [fid for fid, _ in coords]
        global_id_to_frames[global_id].extend(frame_ids)

    for g_id, frames in global_id_to_frames.items():
        if frames:
            min_f = min(frames)
            max_f = max(frames)
            print(f"  🔹 global_id={g_id}: covers frames {min_f} → {max_f} ({len(frames)} total frames)")
        else:
            print(f"  ⚠️  global_id={g_id}: has no coordinates")
    
    if not assigned_global_map:
        print("[INFO] No assigned global tracklets found. Skipping matching.")
        return assigned_global_map, unassigned_local_map


    # Filter unassigned local tracklets by overlap
    # 🔹 Step 1: Build the frame range for assigned global tracklets
    global_frame_ranges = {}
    for (cam_id, local_id), gtrack in assigned_global_map.items():
        gcoords = gtrack["object"]["coords"]
        if not gcoords:
            continue

        # Get the first and last frame of the global tracklet
        first_frame = gcoords[0][0]
        last_frame = gcoords[-1][0]
        global_frame_ranges[(cam_id, local_id)] = (first_frame, last_frame)
    
    # 🔹 Step 2: Find unassigned local tracklets that overlap with assigned tracklets based on frame ranges
    common_frame_unassigned_local_map = {}
    for (cam_id, local_id), frame_obj_list in list(unassigned_local_map.items()):
        local_frames = [f for f, _ in frame_obj_list]
        local_first_frame = min(local_frames)
        local_last_frame = max(local_frames)

        # Check for overlap with any of the global tracklet frame ranges
        for (assigned_cam_id, assigned_local_id), (global_first_frame, global_last_frame) in global_frame_ranges.items():
            if cam_id != assigned_cam_id:
                overlap_start = max(local_first_frame, global_first_frame)
                overlap_end = min(local_last_frame, global_last_frame)
                overlap_length = max(0, overlap_end - overlap_start + 1)

                # 🔸 Additional directional checks
                if not relax_directional_check:
                    if overlap_length >= args.min_shared_frames:
                # if not relax_directional_check:
                #     if overlap_length >= args.min_shared_frames and \
                #     local_first_frame >= global_first_frame and \
                #     local_last_frame > global_last_frame:
                        if (cam_id, local_id) not in common_frame_unassigned_local_map:
                            common_frame_unassigned_local_map[(cam_id, local_id)] = []
                        common_frame_unassigned_local_map[(cam_id, local_id)] = frame_obj_list 
                if relax_directional_check:
                    if overlap_length >= args.min_shared_frames:
                        if (cam_id, local_id) not in common_frame_unassigned_local_map:
                            common_frame_unassigned_local_map[(cam_id, local_id)] = []
                        common_frame_unassigned_local_map[(cam_id, local_id)] = frame_obj_list 
                        
    if not common_frame_unassigned_local_map:
        print("[INFO] No unassigned tracklets with sufficient frame overlap. Ending early.")
        return assigned_global_map, unassigned_local_map

    remaining_unassigned = dict(common_frame_unassigned_local_map)
    match_round = 0
    # remaining_unassigned = {
    #         k: v for k, v in remaining_unassigned.items()
    #         if k not in assigned_global_map
    #     }
    while match_round < 3 and remaining_unassigned:
        newly_assigned = []
        matched_global_ids_this_round = set()
        round_type = ["Full match (traj + feature)", "Fallback (traj only)", "Force match"][match_round]
        print(f"\n[ROUND {match_round + 1}] {round_type}")
        print(f"Remaining to match: {len(remaining_unassigned)}")

        if match_round == 2 and force_match ==False:
            match_round = match_round + 1
            continue

        for (cam_id, local_id), frame_obj_list in tqdm(list(remaining_unassigned.items()), desc=f"Matching round {match_round + 1}"):
            local_frame_ids = [fid for fid, _ in frame_obj_list]
            if not local_frame_ids:
                continue  # Skip if no frames
            local_first_frame = min(local_frame_ids)
            local_last_frame = max(local_frame_ids)

            coords = [(f, obj["3d location"]) for f, obj in frame_obj_list if "3d location" in obj]
            # Extract 2D bounding boxes (list of [x1, y1, x2, y2] for each camera)
            bbox_2d_seq = [
                (f, obj["2d bounding box visible"]) for f, obj in frame_obj_list if "2d bounding box visible" in obj
            ]
            
            # Extract 3D bounding box scale (list of scale for each frame)
            scale_seq = [obj.get("3d bounding box scale") for _, obj in frame_obj_list if "3d bounding box scale" in obj]

            # Extract 3D bounding box rotation (list of rotation for each frame)
            rot_seq = [obj.get("3d bounding box rotation") for _, obj in frame_obj_list if "3d bounding box rotation" in obj]

            if match_round == 0 or match_round ==2:
                features_path = [obj["feature path"] for _, obj in frame_obj_list if "feature path" in obj]
                features_path = features_path[:min(30, len(features_path))] # Only last 30 features considered
                features = load_and_get_representative_features(features_path, local_id, cam_id, args)

            best_match = None
            best_traj_dist = float('inf')
            best_feat_sim = -1

            ###

            for (a_cam_id, a_local_id), gtrack in assigned_global_map.items():
                g_id = gtrack["global_id"]
                gcoords = gtrack["object"]["coords"]
                g_frame_ids = [fid for fid, _ in gcoords]
                if not g_frame_ids:
                    continue
                
                if a_cam_id == cam_id: # an overlapping frame of unassigned local id of a camera view cannot match a global id of same camera view
                    continue
                
                g_first = min(g_frame_ids)
                g_last = max(g_frame_ids)

                overlap_start = max(local_first_frame, g_first)
                overlap_end = min(local_last_frame, g_last)
                overlap_len = max(0, overlap_end - overlap_start + 1)
                if overlap_len < args.min_shared_frames:
                    continue
                
                if not force_match:
                    if g_id in [gtrack["global_id"] for (g_cam_id, _) , gtrack in assigned_global_map.items() if g_cam_id == cam_id]:

                
                traj_dist = trajectory_distance(coords, gcoords)
                if match_round ==0 or match_round ==2:
                    gfeatures = gtrack["object"]["features"][:min(30, len(gtrack["object"]["features"]))]
                    feat_sim = feature_similarity(features, gfeatures, metric=args.metric, topk=10)

                if match_round == 0:
                    match_condition = traj_dist < args.trajectory_dist_thresh_rest and feat_sim > args.feature_similarity_thresh_rest
                elif match_round == 1:
                    match_condition = traj_dist < args.trajectory_dist_thresh_rest
                # elif match_round == 2:
                #     match_condition = feat_sim > args.feature_similarity_thresh_rest
                else:
                    match_condition = True if force_match else False

                if match_condition:
                    is_better = (traj_dist < best_traj_dist) or (traj_dist < best_traj_dist +1.5 and feat_sim > best_feat_sim)
                    if is_better:
                        best_match = (a_cam_id, a_local_id, g_id)
                        best_traj_dist = traj_dist
                        best_feat_sim = feat_sim

            if best_match:
                g_id = best_match[2]
                # Assign object with full data to global map
                assigned_global_map[(cam_id, local_id)] = {
                    "global_id": g_id,
                    "object": {
                        "scene_id": args.scene_id,
                        "camera_id": cam_id,
                        "local_id": local_id,
                        "coords": coords,  # Store coordinates (frame_id, [x, y, z])
                        "features": features,  # Store the extracted features
                        "global_id": g_id,
                        "object type": frame_obj_list[0][1].get("object type", "Unknown"),
                        
                        # Include the 2D bounding box sequence (if available)
                        "2d bounding box visible_sequence": [
                            to_serializable(bbox) for _, bbox in bbox_2d_seq
                        ],
                        
                        # Include the 3D bounding box scale sequence (if available)
                        "3d bounding box scale_sequence": scale_seq,  # Adding the 3D scale sequence here
                        
                        # Include the 3D bounding box rotation sequence (if available)
                        "3d bounding box rotation_sequence": rot_seq,  # Adding the 3D rotation sequence here
                    },
                }
                newly_assigned.append((cam_id, local_id))

                if match_round == 0:
                    print(f"[FEAT-TRAJ-MATCH] (cam={cam_id}, local_id={local_id}) → global_id={g_id} (sim={best_feat_sim:.3f}, dist={best_traj_dist:.3f})")
                elif match_round == 1:
                    print(f"[ONLY-TRAJ-MATCH] (cam={cam_id}, local_id={local_id}) → global_id={g_id} (dist={best_traj_dist:.3f})")
                # elif match_round == 2:
                #     print(f"[ONLY-FEAT-MATCH] (cam={cam_id}, local_id={local_id}) → global_id={g_id} (dist={best_traj_dist:.3f})")
                elif force_match:
                    print(f"[FORCE-MATCH WARNING] Forced (cam={cam_id}, local_id={local_id}) → global_id={g_id} (dist={best_traj_dist:.3f})")
                    
        for key in newly_assigned:
            remaining_unassigned.pop(key, None)


        match_round += 1

    return assigned_global_map, remaining_unassigned
    
def match_unassigned_to_assigned(unassigned_local_map, assigned_global_map, args):
    phase = 0  # 0 = strict, 1 = relax directional check, 2 = force
    stagnant_rounds = 0
    max_stagnant_rounds = args.max_stagnant_rounds

    initial_len = len(unassigned_local_map)
    unassigned_progress_bar = tqdm(total=initial_len, desc="Unassigned Tracklets Progress", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} tracklets assigned", colour='blue')

    completed_global_tracklets = {}
    while True:
        newly_assigned = []

        # Step 0: Refresh active vs completed tracklets
        completed_global_map, active_global_map = split_global_map_by_completion(assigned_global_map, args)
        completed_global_tracklets.update(completed_global_map)
        # assigned_global_map = active_global_map  # Only use active tracklets for matching


        print("[DEBUG] Active global tracklets after filtering:")
        print("[DEBUG] Global ID → Frame Range Summary (Active Tracklets):")

        
        print(f"\n[PHASE {phase}] Active tracklets: {len(assigned_global_map)}, Completed: {len(completed_global_map)}")

        unassigned_progress_bar.set_postfix({"Remaining": len(unassigned_local_map)}, refresh=True)
        # Step 1: Try assigning
        assigned_global_map, remaining_unassigned = global_matching(
            assigned_global_map,
            unassigned_local_map,
            args,
            force_match=(phase == 1 and args.force_match),
            relax_directional_check=(phase >= 1),
        )

         # Step 2: Check stagnation
        newly_completed_global_map, newly_active_global_map = split_global_map_by_completion(assigned_global_map, args)
        completed_global_tracklets.update(newly_completed_global_map)
        print(f"[INFO] Total completed global tracklets: {len(completed_global_tracklets)}")
        
        # Exclude already assigned (either completed or active)
        assigned_keys = set(newly_completed_global_map.keys()) | set(newly_active_global_map.keys())
        unassigned_local_map = {
            k: v for k, v in unassigned_local_map.items()
            if k not in assigned_keys
        }

        if len(newly_active_global_map) == len(active_global_map):  # no new progress
            stagnant_rounds += 1
            print(f"[PHASE {phase}] No change in active global map. Stagnant round {stagnant_rounds}/{max_stagnant_rounds}")
        else:
            stagnant_rounds = 0  # reset if progress is made

        if stagnant_rounds >= max_stagnant_rounds:
            if args.force_match:
                phase += 1
                stagnant_rounds = 0
                
                if phase > 1 and len(unassigned_local_map) == 0:
                    print(f"\n[ADVANCING TO PHASE {phase}]")
                    print("[STOP] All matching attempts completed.")
                    break
                elif phase > 1 and len(unassigned_local_map) > 0:
                    print("[REPEAT MATCH] Still some unmatching tracklets.")
                    phase = 0
                    stagnant_rounds = 0
                    
            else:
                # If force_match is False, pick the best unassigned tracklet and assign a new global ID
                print("[INFO] No progress. Assigning new global IDs to unassigned tracklets.")
                
                # Sort unassigned tracklets by the first frame, pick the earliest one
                best_unassigned = min(unassigned_local_map.items(), key=lambda x: min(f for f, _ in x[1]))
                cam_id, local_id = best_unassigned[0]
                frame_obj_list = best_unassigned[1]
                
                # Assign a new global ID (+1)
                new_global_id = max(gtrack["global_id"] for gtrack in assigned_global_map.values()) + 1

                # Move it from unassigned to assigned
                assigned_global_map[(cam_id, local_id)] = {
                    "global_id": new_global_id,
                    "object": {
                        "scene_id": args.scene_id,
                        "camera_id": cam_id,
                        "local_id": local_id,
                        "coords": [(f, obj["3d location"]) for f, obj in frame_obj_list],
                        # "features": [(obj["feature path"]) for _, obj in frame_obj_list if "feature path" in obj],
                        "features": [],
                        "global_id": new_global_id,
                        "object type": frame_obj_list[0][1].get("object type", "Unknown"),
                        "2d bounding box visible_sequence": [
                            to_serializable(obj["2d bounding box visible"]) for _, obj in frame_obj_list if "2d bounding box visible" in obj
                        ],
                        "3d bounding box scale_sequence": [
                            obj.get("3d bounding box scale") for _, obj in frame_obj_list if "3d bounding box scale" in obj
                        ],
                        "3d bounding box rotation_sequence": [
                            obj.get("3d bounding box rotation") for _, obj in frame_obj_list if "3d bounding box rotation" in obj
                        ],
                    },
                }
                # Now load the features from the file paths and get representative features as before
                features_path = [obj["feature path"] for _, obj in frame_obj_list if "feature path" in obj]
                features_path = features_path[:min(30, len(features_path))]  # Only last 30 features considered
                features = load_and_get_representative_features(features_path, local_id, cam_id, args)
                assigned_global_map[(cam_id, local_id)]["object"]["features"] = features
                # Remove it from unassigned_local_map
                del unassigned_local_map[(cam_id, local_id)]

                print(f"[INFO] Assigned new global ID {new_global_id} to (cam={cam_id}, local_id={local_id})")

                # Recheck the phases 0 and 1 after assigning the new global ID
                phase = 0
                stagnant_rounds = 0  # Reset stagnant rounds

        if not unassigned_local_map:
            print("[INFO] All unassigned tracklets have been assigned.")
            break  # Exit the loop when all unassigned tracklets are processed

    final_assigned_global_map = {}
    final_assigned_global_map.update(completed_global_tracklets)
    return final_assigned_global_map, assigned_global_map
