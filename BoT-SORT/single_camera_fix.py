import json
import os
import argparse
from collections import defaultdict
from itertools import combinations
import numpy as np
from tools.utils_25 import plot_tracking_with_world, compute_iou, assign_features_to_tracklets, visualize_fixed_tracklets,\
load_detection_json, fix_zero_yaws_by_tracklet, remove_invalid_and_short_tracklets, print_tracklet_coords_by_frame,\
interpolate_missing_tracklet_frames, compute_iou_intersection

import time



def get_final_new_id(old_id, merged):
    """
    Recursively find the final new_id for a given old_id by following the merge mapping.
    This ensures that all references to old_id are replaced with the final new_id.
    If the old_id is a value in merged, replace it with the most recent new_id.
    """
    # Iterate through the merged dictionary to check if any value is equal to old_id
    for key, value in merged.items():
        if value == old_id:
            # Update the merged dictionary to reflect the change in key-value pair
            merged[key] = merged.get(value, value)
            # Recurse until we find the final new_id
            get_final_new_id(key, merged)

def tracklets_non_maximum_suppression(data, fix_cfg, args):
    """
    Merge tracklets based on temporal and spatial overlap, and remove short tracks.
    """
    temporal_iou_thresh = fix_cfg.get("temporal_iou_thresh", 0.7)
    spatial_iou_thresh = fix_cfg.get("spatial_iou_thresh", 0.7)
    remove_short_tracks = fix_cfg.get("remove_short_tracks", True)
    min_track_len = fix_cfg.get("min_track_len", 20)

    track_by_id = defaultdict(list)

    for frame_id, objects in data.items():
        for obj in objects:
            tid = obj["object sc id"]
            track_by_id[tid].append((int(frame_id), obj))
    print(f"[INFO] Number of tracklets before NMS: {len(track_by_id)}")

    # Remove short tracks
    if remove_short_tracks:
        track_by_id = {tid: track for tid, track in track_by_id.items() if len(track) >= min_track_len}

    track_ids = sorted(track_by_id.keys())
    merged = {}

    for id1, id2 in combinations(track_ids, 2):
        track1 = track_by_id[id1]
        track2 = track_by_id[id2]

        f1 = set(f for f, _ in track1)
        f2 = set(f for f, _ in track2)
        common_frames = f1 & f2

        # Debug
        _, obj1 = track1[0]
        _, obj2 = track2[0]
        if obj1["object sc id"] == 61 and obj2["object sc id"] == 67 :
            z = 2
        #
        if obj1["object sc id"] == 67 and obj2["object sc id"] == 61 :
            z = 2
        #
        if not common_frames:
            continue

        temporal_overlap = len(common_frames) / min(len(f1), len(f2))
        if temporal_overlap < temporal_iou_thresh:
            continue

        spatial_ious = []
        for frame_id in common_frames:
            camera_key = args.camera_id  # e.g., "Camera_01"
            bbox1 = next(obj["2d bounding box visible"].get(camera_key, [0, 0, 0, 0]) for f, obj in track1 if f == frame_id)
            bbox2 = next(obj["2d bounding box visible"].get(camera_key, [0, 0, 0, 0]) for f, obj in track2 if f == frame_id)
            iou = compute_iou_intersection(bbox1, bbox2)
            spatial_ious.append(iou)

        if np.mean(spatial_ious) >= spatial_iou_thresh:
            # Keep longer tracklet
            if len(track2) > len(track1):
                id1, id2 = id2, id1
                track1, track2 = track2, track1
            # Merge id2 into id1
            merged[id2] = id1

    # Apply merging
    for old_id, new_id in merged.items():
        if old_id in track_by_id:
            # Get the final new_id for the old_id (if it has already been replaced in previous merges)
            if old_id == 925:
                z=2
            get_final_new_id(old_id, merged)  # Update the merge map in place

            # After updating, the final new_id for the old_id is in merged[old_id]
            final_new_id = merged[old_id]

            # Pop the old track
            old_track = track_by_id.pop(old_id)
            existing_frames = set(f for f, _ in track_by_id[final_new_id])

            # Get dominant class from the longer tracklet (final_new_id)
            dominant_class = None
            for _, obj in track_by_id[final_new_id]:
                dominant_class = obj["object type"]
                break  # Assume all entries in the longer tracklet have the same class

            # Only add frames not already present in final_new_id
            for f, obj in old_track:
                if f not in existing_frames:
                    # Overwrite class with dominant class
                    obj["object type"] = dominant_class
                    track_by_id[final_new_id].append((f, obj))

            # Update the merge map: replace the old_id with the most recent new_id
            merged[old_id] = final_new_id
            


    # Re-assign new IDs and reconstruct JSON
    new_data = defaultdict(list)
    for new_id, track in enumerate(track_by_id.values(), 1):
        for frame_id, obj in track:
            obj["object sc id"] = new_id
            new_data[str(frame_id)].append(obj)
    print(f"[INFO] Number of tracklets after NMS: {len(track_by_id)}")

    return new_data

def parse_args():
    parser = argparse.ArgumentParser(description="Post-process and fix single-camera tracklets with NMS.")

    parser.add_argument(
        "-s", "--scene_id", type=str, required=True,
        help="Scene ID (e.g., Warehouse_016)"
    )
    parser.add_argument(
        "-c", "--camera_id", type=str, required=True,
        help="Camera ID (e.g., Camera_01)"
    )
    parser.add_argument(
        "--base_dir", type=str, default="Tracking/Singlecamera",
        help="Base directory of tracking results"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Optional path to save output JSON"
    )
    parser.add_argument(
        "--limit_frame", type=int, default=-1,
        help="Limit number of frames to process (for debugging). Use -1 to process all frames."
    )
    parser.add_argument(
        "--nms", action="store_true",
        help="Enable NMS-based tracklet merging"
    )
    parser.add_argument(
        "--dataset", type=str, default="Val",
        help="Dataset type ,E.g Val or Test"
    )
    parser.add_argument(
        "--frame_num_save", type=int, default=200,
        help="Save num of frames for visualization"
    )
    


    # Fixing configs
    parser.add_argument("--temporal_iou_thresh", type=float, default=0.7, help="Temporal IoU threshold for merging tracks")
    parser.add_argument("--spatial_iou_thresh", type=float, default=0.7, help="Spatial IoU threshold for merging tracks")
    parser.add_argument("--remove_short_tracks", action="store_true", help="Remove short-length tracklets")
    parser.add_argument("--min_track_len", type=int, default=20, help="Minimum length of track to keep")
    parser.add_argument("--iou_features", type=float, default=0.8, help="IoU threshold for finding features of tracklets in each frame")
    parser.add_argument("--overlap_block_thresh", type=float, default=0.4, help="IoU threshold for finding interference with other bbox to reject overlapping features")

    
    return parser.parse_args()





def main():
    args = parse_args()
    fix_cfg = {
    "temporal_iou_thresh": args.temporal_iou_thresh,
    "spatial_iou_thresh": args.spatial_iou_thresh,
    "remove_short_tracks": args.remove_short_tracks,
    "min_track_len": args.min_track_len,
    "iou_features": args.iou_features,
    "overlap_block_thresh": args.overlap_block_thresh
    }
    
    input_path = os.path.join(args.base_dir, args.scene_id, args.camera_id, args.camera_id + ".json")
    if not os.path.exists(input_path):
        print(f"[Error] File not found: {input_path}")
        return

    with open(input_path, "r") as f:
        data = json.load(f)

    # Limit data for debugging
    if args.limit_frame > 0:
        limited_data = {k: v for k, v in data.items() if int(k) <= args.limit_frame}
    else:
        limited_data = data

    # Apply NMS (merge overlapping tracklets)
    if args.nms:
        start_time = time.time()
        data = tracklets_non_maximum_suppression(limited_data, fix_cfg, args)
        elapsed = time.time() - start_time
        print(f"[NMS PROCESS] completed for {args.dataset} dataset, {args.scene_id}-{args.camera_id}")
        print(f"[NMS PROCESS] time taken: {elapsed:.2f} seconds")
    else:
        data = limited_data
    

    # Find non-overlapping features
    detection_json_path = f"Detection/{args.scene_id}/{args.camera_id}.json"
    det_by_frame = load_detection_json(detection_json_path)
    data = assign_features_to_tracklets(data, det_by_frame, args.camera_id, args.iou_features, args.overlap_block_thresh)

    # Fix zero yaws
    data = fix_zero_yaws_by_tracklet(data)

    # Fix remove none type objects
    data = remove_invalid_and_short_tracklets(data, fix_cfg.get("min_track_len", 20))


    # Debug purpose
    # print_tracklet_coords_by_frame(data)

    # Interpolate missing frames in valid tracklets
    data = interpolate_missing_tracklet_frames(data, camera_key=args.camera_id)

    # Visualize fixed tracklets
    print(f"[WARN] visualization is for {args.dataset} dataset, {args.scene_id}-{args.camera_id}")     
    image_root = os.path.join("AIC25_Track1", args.dataset, args.scene_id, "videos", args.camera_id, "Frame")
    save_dir = os.path.join("Tracking", "Singlecamera", args.scene_id, args.camera_id, "fixed_Frames")
    visualize_fixed_tracklets(data, image_root, args.camera_id, save_dir, max_frames=args.frame_num_save)


    if args.output:
        output_path = args.output
    else:
        output_dir = os.path.dirname(input_path)
        output_filename = f"fixed_{args.camera_id}.json"
        output_path = os.path.join(output_dir, output_filename)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[Done] Saved fixed tracklets to: {output_path}")


if __name__ == "__main__":
    main()