import os
import json
import argparse
from datetime import datetime

from tracking import Tracker
from utils import DetectedObjects
import cv2

import numpy as np

def run_scpt(feature_data_root, out_dir="outdir", tracking_params={}):
    # Load and generate "detected object list"
    tracking_results = {}
    if not os.path.isdir(feature_data_root):
        raise Exception(f"No such directory: {feature_data_root}")
    if os.path.basename(feature_data_root).startswith("Camera"):
        camera_ids = [os.path.basename(feature_data_root)]
        feature_data_root = os.path.dirname(feature_data_root)
        is_multi = False
    else:
        camera_ids = [cam_id for cam_id in os.listdir(feature_data_root) if cam_id[:7] == "camera_"]
        is_multi = True

    # loading detections
    for camera_id in camera_ids:
        data_dir = os.path.join(feature_data_root, camera_id)
        # camera_id = int(camera_id[7:])
        detected_objects = load_detections(data_dir)
        visualize_detections_with_gt(detected_objects, camera_id)
        tracking_results[camera_id] = detected_objects.to_trackingdict()
        del detected_objects
    
    # Run SCT on all detections of all cameras
    for camera_id in tracking_results:
        tracking_dict = tracking_results[camera_id]
        start_time = datetime.now()
        tracker = Tracker(tracking_params)
        if tracking_params['split_cls']:
            tracking_results[camera_id] = tracker.scpt_cls(tracking_dict) # tracking returns tracking_dict
        else:
            tracking_results[camera_id] = tracker.scpt(tracking_dict) # tracking returns tracking_dict
        end_time = datetime.now()
        print(f"Camera{camera_id} elapsed time: {end_time - start_time}")

        # Dump the result
        out_json = os.path.join(out_dir, f'{camera_id}_tracking_results.json')
        os.makedirs(os.path.dirname(out_json), exist_ok=True)
        with open(out_json, mode='w') as f:
            json.dump(tracking_results[camera_id], f)        

def run_mcpt(scene_id, json_dir,out_dir="outdir", tracking_params={}):
    start_time = datetime.now()
    tracker = Tracker(tracking_params)
    whole_tracking_result = tracker.mcpt(scene_id, json_dir,out_dir)
    
    # Dump the result
    out_file = os.path.join(out_dir, 'whole_tracking_results.json')
    with open(out_file, mode='w') as f:
        json.dump(whole_tracking_result, f)
    end_time = datetime.now()
    print(f"Elapsed_time: {end_time - start_time}")


def correct_scpt_result(scene_id, json_dir, out_dir=None, tracking_params={}):
    if not os.path.isdir(json_dir):
        raise Exception(f"The directory '{json_dir}' does not exist.")
    if out_dir == None:
        out_dir = json_dir
    
    json_files = [f for f in os.listdir(json_dir) if os.path.splitext(f)[1].lower() == ".json" and f.startswith("Camera")]
    json_files = sorted(json_files)
    for json_file in json_files:
        # camera_id = int(json_file.split("_")[0][6:])
        with open(os.path.join(json_dir, json_file)) as f:
            tracking_dict = json.load(f)
        tracker = Tracker(tracking_params)
        tracking_dict = tracker.correcting_scpt_result(tracking_dict) 
        out_file = os.path.join(out_dir, "fixed_"+os.path.basename(json_file))
        with open(out_file, mode='w') as f:
            json.dump(tracking_dict, f)

def correct_mcpt_result(scene_id,json_dir,out_dir,tracking_params={}):
    with open(os.path.join(json_dir, 'whole_tracking_results.json')) as f:
        tracking_results = json.load(f)
    with open(os.path.join(json_dir, f"representative_nodes_{scene_id}.json")) as f:
        representative_nodes = json.load(f)
    tracker = Tracker(tracking_params)
    tracking_resuluts = tracker.correcting_mcpt_result(scene_id,tracking_results,representative_nodes)
    out_file = os.path.join(out_dir, "fixed_whole_tracking_results.json")
    with open(out_file, mode='w') as f:
        json.dump(tracking_resuluts, f)


def load_detections(data_root, debug=False):
    print(f"Loading detections from {data_root}.")
    detected_objects = DetectedObjects()
    detected_objects.load_from_directory(feature_root=data_root)
    print(f"Found {len(detected_objects.objects)} frames, and {detected_objects.num_objects} objects.")
    if debug:
        frames = sorted(detected_objects.objects)
        min_num_obj = 9999999
        max_num_obj = 0
        for frame in frames:
            obj = detected_objects[frame]
            num = len(obj)
            min_num_obj = min(min_num_obj, num)
            max_num_obj = max(max_num_obj, num)
        print(f"###  MIN num detections: {min_num_obj},  MAX num detections: {max_num_obj} ###\n")

    return detected_objects

def visualize_detections(detected_objects, camera_id, output_dir="Detection/Warehouse_016/visualized", max_frames=100):
    os.makedirs(os.path.join(output_dir, camera_id), exist_ok=True)

    frame_ids = sorted(detected_objects.objects.keys())[:max_frames]

    for frame_id in frame_ids:
        objects_in_frame = detected_objects.objects.get(frame_id, [])
        if not objects_in_frame:
            continue

        # Use the first object in the frame to find image path
        img_path = objects_in_frame[0].image_path
        if not img_path or not os.path.exists(img_path):
            print(f"[WARN] Missing image for frame {frame_id}: {img_path}")
            continue

        img = cv2.imread(img_path)
        if img is None:
            print(f"[WARN] Failed to load image: {img_path}")
            continue

        for obj in objects_in_frame:
            x1, y1, x2, y2 = obj.coordinate.x1, obj.coordinate.y1, obj.coordinate.x2, obj.coordinate.y2
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            cls = getattr(obj, 'cls', -1)  # Default to -1 if class is missing

            # Draw bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Prepare label text
            label = f"cls{cls} | 2D:({cx},{cy})"
            if obj.worldcoordinate and hasattr(obj.worldcoordinate, 'z'):
                wx, wy, wz = obj.worldcoordinate.x, obj.worldcoordinate.y, obj.worldcoordinate.z
                label += f" | 3D:({wx:.2f},{wy:.2f},{wz:.2f})"

            # Draw label text
            cv2.putText(img, label, (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, (0, 0, 255), 1, cv2.LINE_AA)

        out_path = os.path.join(output_dir, camera_id, f"frame_{frame_id:05d}.jpg")
        cv2.imwrite(out_path, img)

def visualize_detections_with_gt(detected_objects, camera_id,
                                 gt_path="AIC25_Track1/Val/Warehouse_016/ground_truth.json",
                                 output_dir="Detection/Warehouse_016/visualized_gt", max_frames=100):
    os.makedirs(os.path.join(output_dir, camera_id), exist_ok=True)

    with open(gt_path, "r") as f:
        gt_data = json.load(f)

    frame_ids = sorted(detected_objects.objects.keys())[:max_frames]

    for frame_id in frame_ids:
        objects_in_frame = detected_objects.objects.get(frame_id, [])
        if not objects_in_frame:
            continue

        # Image path
        img_path = objects_in_frame[0].image_path
        if not img_path or not os.path.exists(img_path):
            print(f"[WARN] Missing image for frame {frame_id}: {img_path}")
            continue

        img = cv2.imread(img_path)
        if img is None:
            print(f"[WARN] Failed to load image: {img_path}")
            continue

        gt_objs = gt_data.get(str(frame_id), [])
        gt_bboxes = []
        for obj in gt_objs:
            if camera_id in obj.get("2d bounding box visible", {}):
                bbox = obj["2d bounding box visible"][camera_id]
                label = obj.get("object type", "unknown")
                gt3d = obj.get("3d location", None)
                gt_bboxes.append((bbox, label, gt3d))

        for obj in objects_in_frame:
            x1, y1, x2, y2 = obj.coordinate.x1, obj.coordinate.y1, obj.coordinate.x2, obj.coordinate.y2
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            cls = getattr(obj, 'cls', -1)

            # Draw predicted bbox
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"cls{cls} | 2D:({cx},{cy})"

            if obj.worldcoordinate and hasattr(obj.worldcoordinate, 'z'):
                wx, wy, wz = obj.worldcoordinate.x, obj.worldcoordinate.y, obj.worldcoordinate.z
                label += f" | 3D:({wx:.2f},{wy:.2f},{wz:.2f})"

            # Find closest GT bbox
            closest_gt = None
            min_dist = float('inf')
            for gt_bbox, gt_label, gt3d in gt_bboxes:
                gx1, gy1, gx2, gy2 = gt_bbox
                gcx, gcy = (gx1 + gx2) / 2, (gy1 + gy2) / 2
                dist = np.sqrt((gcx - cx)**2 + (gcy - cy)**2)
                if dist < min_dist:
                    min_dist = dist
                    closest_gt = (gt_bbox, gt_label, gt3d)

            # Draw GT bbox (blue) and annotate
            if closest_gt:
                gt_bbox, gt_label, gt3d = closest_gt
                gx1, gy1, gx2, gy2 = map(int, gt_bbox)
                cv2.rectangle(img, (gx1, gy1), (gx2, gy2), (255, 0, 0), 1)  # blue box
                if gt3d:
                    label += f" | GT3D:({gt3d[0]:.2f},{gt3d[1]:.2f},{gt3d[2]:.2f})"

            # Draw label
            cv2.putText(img, label, (x1, max(0, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)

        out_path = os.path.join(output_dir, camera_id, f"frame_{frame_id:05d}.jpg")
        cv2.imwrite(out_path, img)

def get_args():
    parser = argparse.ArgumentParser(description='Offline Tracker sample app.')
    parser.add_argument('-d', '--data', default='EmbedFeature/scene_001', type=str)
    parser.add_argument('-o', '--outdir', default='output', type=str)

    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()

    run(feature_data_root=args.data, out_dir=args.outdir, tracking_params={})
