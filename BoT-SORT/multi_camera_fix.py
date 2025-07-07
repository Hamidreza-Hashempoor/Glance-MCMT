import json
import os
import argparse
from collections import defaultdict
from itertools import combinations
import numpy as np
from tools.utils_25_revised import post_process_multiple_camera_tracking, load_predictions, process_all_frames, write_ground_truth_txt_from_json,  write_framewise_txt_from_json

import time



def parse_args():
    parser = argparse.ArgumentParser(description="Post-process and fix single-camera tracklets with NMS.")

    parser.add_argument(
        "-s", "--scene_id", type=str, required=True,
        help="Scene ID (e.g., Warehouse_016)"
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
        "--frame_num_save", type=int, default=100,
        help="Save num of frames for visualization"
    )
    


    # Fixing configs
    parser.add_argument("--temporal_iou_thresh", type=float, default=0.5, help="Temporal IoU threshold for merging tracks")
    parser.add_argument("--spatial_iou_thresh", type=float, default=0.7, help="Spatial IoU threshold for merging tracks")
    parser.add_argument("--remove_short_tracks", action="store_true", help="Remove short-length tracklets")
    parser.add_argument("--min_track_len", type=int, default=200, help="Minimum length of track to keep")
    parser.add_argument("--iou_features", type=float, default=0.8, help="IoU threshold for finding features of tracklets in each frame")
    parser.add_argument("--overlap_block_thresh", type=float, default=0.5, help="IoU threshold for finding interference with other bbox to reject overlapping features")
    parser.add_argument("--exp_path", type=str, default="exp55", help="Experiment")
    parser.add_argument("--trajectory_dist_thresh", type=float, default=2.0, help="traj overlap")
    

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
    
    framewise_json = post_process_multiple_camera_tracking(args)

    
    # Load the JSON data
    with open("AIC25_Track1/Val/Warehouse_016/ground_truth.json", "r") as f:
        ground_truth_json = json.load(f)
    # with open(f"Tracking/Multicamera/Warehouse_016/{args.exp_path}/output_result/final_track_Warehouse_016_fix.json", "r") as f:
    #     framewise_json = json.load(f)

    ## Save into text
    scene_path = f"Tracking/Multicamera/{args.scene_id}/{args.exp_path}"
    write_framewise_txt_from_json(scene_path, args, framewise_json)
    write_ground_truth_txt_from_json(scene_path, args, ground_truth_json)

    ## Visualization (draw and save images with boxes)
    predictions = framewise_json
    base_video_path = f"AIC25_Track1/{args.dataset}/{args.scene_id}/videos"  # Path to your base video folder
    output_vis_root = os.path.join("Tracking", "Multicamera", args.scene_id, args.exp_path, "output_vis_fix") # Path to save output images
    camera_ids = [d for d in os.listdir(f"AIC25_Track1/{args.dataset}/{args.scene_id}/videos") if os.path.isdir(os.path.join(f"AIC25_Track1/{args.dataset}/{args.scene_id}/videos", d))]

    # Process all frames for all cameras
    print(f"Start drawing a few samples on : {output_vis_root}")
    process_all_frames(predictions, base_video_path, output_vis_root, camera_ids, args)


if __name__ == "__main__":
    main()