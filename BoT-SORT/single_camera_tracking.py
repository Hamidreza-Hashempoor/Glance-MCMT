import argparse
import json
from collections import defaultdict, OrderedDict
from pathlib import Path
import os
import numpy as np
import cv2
import re
from tqdm import tqdm
import sys
try:
    sys.path.append('/data2/Hamid/AI_city_challenge_2025/AIC24_Track1_YACHIYO_RIIPS/BoT-SORT')
except:
    print( "bot sort already in path")

from tracker.bot_sort import BoTSORT  # Make sure this import path is correct
from tools.utils_25 import plot_tracking, load_full_detections_per_frame, preview_frames, print_nth_frame_detections, load_depth_map,\
unload_depth_map, plot_tracking_with_world, get_gt_coords_for_tracks, compute_average_3d_scales, get_yaw_for_tracks,\
     analyze_gt_rotations_by_class, CameraCalibration
# from tracker.utils.visualize import plot_tracking  # Optional
from collections import OrderedDict
from pathlib import Path

# Names
CLASS_ID_TO_NAME = {
    0: "Person",
    1: "Forklift",
    2: "NovaCarter",
    3: "Transporter",
    4: "FourierGR1T2",
    5: "AgilityDigit"
}
CLASS_NAME_TO_ID = {
    "Person": 0,
    "Forklift": 1,
    "NovaCarter": 2,
    "Transporter": 3,
    "FourierGR1T2": 4,
    "AgilityDigit": 5
}

FIXED_SCALES_BY_CLASS = {
    "Transporter": [1.430852, 0.654056, 0.208093],
    "AgilityDigit": [0.537891, 0.795680, 1.712401],
    "FourierGR1T2": [0.600571, 0.503744, 1.643690],
    "Person": [0.602972, 0.459467, 1.899607],
    "Forklift": [1.157524, 2.019892, 2.563527],     # from earlier values
    "NovaCarter": [0.705297, 0.447695, 0.425750]     # from earlier values
}

###
def run_botsort_tracking(frame_detections: OrderedDict, args, save_dir=None):

    """
    Run BoT-SORT tracker on loaded detection data.

    Args:
        frame_detections (OrderedDict): {frame_id: [list of detection dicts]}
        args (argparse.Namespace): Arguments required by BoT-SORT
        save_dir (str or Path): Optional output directory to save visualizations/results
    """
    tracker = BoTSORT(args, frame_rate=args.fps)
    results = []

    
    print(f"[WARN] depth map and calibration are for {args.dataset} dataset")
    
    # Load depth map
    depth_h5 = load_depth_map(f"AIC25_Track1/{args.dataset}/{args.scene_id}/depth_map/{args.camera_id}.h5")

    # Load calibration
    calib = CameraCalibration()
    calib.load_calibration(f"AIC25_Track1/{args.dataset}/{args.scene_id}/calibration.json", args.camera_id)

    


    if args.dataset == "Val" or args.dataset == "Train":
        # Load gt (for visualization of boxes and 3D points)
        with open(f"AIC25_Track1/{args.dataset}/{args.scene_id}/ground_truth.json", "r") as f:
            gt_data = json.load(f)
        scales_by_class = compute_average_3d_scales(gt_data)
        for cls, stat in scales_by_class.items():
            print(f"{cls}: mean={stat['mean']}, std={stat['std']}")
        CLASS_ID_TO_SCALE = {
            CLASS_NAME_TO_ID[class_name]: stats['mean']
            for class_name, stats in scales_by_class.items()
            if class_name in CLASS_NAME_TO_ID
        }
    else:
        # Use fixed values from val statistics
        CLASS_ID_TO_SCALE = {
            CLASS_NAME_TO_ID[class_name]: scale
            for class_name, scale in FIXED_SCALES_BY_CLASS.items()
            if class_name in CLASS_NAME_TO_ID
        }

    # Analyze rotation 
    # rotation_stats = analyze_gt_rotations_by_class(gt_data)

    # Main Loop
    all_frame_outputs = defaultdict(list)
    for frame_idx, (frame_id, detections) in enumerate(tqdm(frame_detections.items(), desc="Tracking frames"), 1):
        if len(detections) == 0:
            continue
        if args.limit_frames >0 and frame_idx> args.limit_frames:
            print(f"[Warning] Only {args.limit_frames} are process for debug reason. Remove limit_frames element in terminal for full process.")
            break

        # Load image (needed for tracking + optional visualization)
        raw_path = detections[0]["ImgPath"]
        img_path = raw_path.lstrip("../") if raw_path.startswith("../") else raw_path
        img = cv2.imread(img_path)
        if img is None:
            print(f"[Warning] Cannot read image {img_path}")
            continue

        # Prepare detection array: [x1, y1, x2, y2, score, class]
        det_arr = []
        features = []
        world_coords = []
        for det in detections:
            coord = det["Coordinate"]
            npy_full_path = os.path.join("EmbedFeature", det["NpyPath"])
            feat = np.load(npy_full_path)
            features.append(feat)

            x1, y1, x2, y2 = coord["x1"], coord["y1"], coord["x2"], coord["y2"]
            # Extract confidence from filename
            npy_filename = os.path.basename(det["NpyPath"])
            match = re.search(r"feature_\d+_\d+_\d+_\d+_\d+_\d+_(\d+)_cls\d+\.npy", npy_filename)
            if match:
                conf_raw = match.group(1)
                # Example: "09947138428688049" → 0.9947138428688049
                conf_score = float(conf_raw[0] + "." + conf_raw[1:])
            else:
                print(f"[Warning] Confidence not parsed from {npy_filename}, defaulting to 0.99")
                conf_score = 0.99
            det_arr.append([x1, y1, x2, y2, conf_score, det["ClassID"]])

            # Estimate world coordinate from bbox center-bottom
            cx = int((x1 + x2) / 2)
            cy = int(y2)  # bottom center
            if args.use_depth:    
                frame_key = f"distance_to_image_plane_{int(frame_id):05d}.png"
                if depth_h5 and frame_key in depth_h5:
                    depth_map = depth_h5[frame_key][:]
                    world_coord = calib.convert_coordinates_to_3d_world(cx, cy, depth_map)
                else:
                    print(f"[WARN] Depth map missing for frame {frame_key}, falling back to None")
                    world_coord = None
            else:
                world_coord = calib.convert_coordinates_to_3d_ground(cx, cy)

            world_coords.append(world_coord)

        det_arr = np.array(det_arr)
        features = np.stack(features, axis=0)

        # Feed to tracker
        online_targets = tracker.update(det_arr, img, features, world_coords)

        # Get yaw angles
        yaw_angles = get_yaw_for_tracks(online_targets)

        # Store (.txt format)
        # for t in online_targets:
        #     tlwh = t.tlwh
        #     tid = t.track_id
        #     results.append(f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n")

        # Store (.json format)
        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id
            bbox_2d = [int(tlwh[0]), int(tlwh[1]), int(tlwh[0] + tlwh[2]), int(tlwh[1] + tlwh[3])]

            obj_class_id = getattr(t, 'cls_id', -1)
            obj_class_name = CLASS_ID_TO_NAME.get(obj_class_id, "Unknown")

            world_coord = t.world_coord if t.world_coord is not None else [0.0, 0.0, 0.0]
            bbox_scale = CLASS_ID_TO_SCALE.get(obj_class_id, [0.5, 0.5, 0.5])  # fallback
            yaw = yaw_angles.get(tid, 0.0)

            all_frame_outputs[str(frame_id)].append({
                "object type": obj_class_name,
                "object sc id": tid,
                "3d location": [round(v, 6) for v in world_coord],
                "3d bounding box scale": [round(v, 6) for v in bbox_scale],
                "3d bounding box rotation": [0.0, 0.0, round(yaw, 6)],
                "2d bounding box visible": {
                    args.camera_id: bbox_2d
                }
            })

        # Optional: Visualization
        if args.save_result and save_dir and frame_idx < args.frame_num_save:
            os.makedirs(Path(save_dir) / "Frames", exist_ok=True)

            # out_img = plot_tracking(img, [t.tlwh for t in online_targets], [t.track_id for t in online_targets], frame_id=frame_id)
            gt_coords = None
            if args.dataset == "Val":
                gt_coords = get_gt_coords_for_tracks(
                                                        gt_data=gt_data,
                                                        camera_id=args.camera_id,
                                                        frame_id=frame_id,
                                                        tlwhs=[t.tlwh for t in online_targets]
                                                    )
            out_img = plot_tracking_with_world(
                                                img,
                                                [t.tlwh for t in online_targets],
                                                [t.track_id for t in online_targets],
                                                world_coords=[t.world_coord for t in online_targets],
                                                gt_coords=gt_coords,  # Optional: supply if available
                                                frame_id=frame_id
                                            )
            
            out_path = Path(save_dir) / "Frames"/ f"{frame_id:06d}.jpg"
            cv2.imwrite(str(out_path), out_img)


    # Save tracking results

    output_path = os.path.join(save_dir, f"{args.camera_id}.json")
    with open(output_path, "w") as f:
        json.dump(all_frame_outputs, f, indent=2)
    print(f"[BoTSORT] Results for scene {args.scene_id}-{args.camera_id} is saved to {output_path}")

    # Unload the depth map
    unload_depth_map(depth_h5)
    print(f"unload depth of {args.scene_id}-{args.camera_id}")

def parse_args():
    parser = argparse.ArgumentParser(description="Run BoT-SORT tracking with precomputed detections.")

    # Detection input
    parser.add_argument(
        "--preview", action="store_true", default=False,
        help="Whether to preview the first few frames"
    )
    parser.add_argument(
        "--max_preview", type=int, default=3,
        help="Number of frames to preview (if --preview is set)"
    )
    parser.add_argument(
        "-s", "--scene_id", type=str, default="Warehouse_016",
        help="Scene ID (e.g., Warehouse_016)"
    )
    parser.add_argument(
        "-c", "--camera_id", type=str, default="Camera_01",
        help="Camera ID (e.g., Camera_01)"
    )
    parser.add_argument(
        "--dataset", type=str, default="Val",
        help="Dataset type ,E.g Val or Test"
    )

    parser.add_argument(
        "--frame_num_save", type=int, default=100,
        help="number of print frames"
    )
    parser.add_argument(
        "--limit_frames", type=int, default=-1,
        help="only process portion of frames"
    )
    parser.add_argument(
        "--use_depth", type=bool, default=False,
        help="only process portion of frames"
    )

    
    

    

    # Tracking parameters (BoT-SORT)
    parser.add_argument("--track_high_thresh", type=float, default=0.5)
    parser.add_argument("--track_low_thresh", type=float, default=0.1)
    parser.add_argument("--new_track_thresh", type=float, default=0.6)
    parser.add_argument("--track_buffer", type=int, default=30)
    parser.add_argument("--proximity_thresh", type=float, default=0.5)
    parser.add_argument("--appearance_thresh", type=float, default=0.25)
    parser.add_argument("--match_thresh", type=float, default=0.8)

    # ReID and model config
    parser.add_argument("--with_reid", action="store_true", default=False)
    parser.add_argument("--fast_reid_config", type=str, default="path/to/config.yaml")
    parser.add_argument("--fast_reid_weights", type=str, default="path/to/model.pth")

    # Runtime
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--cmc_method", type=str, default="sparseOptFlow")
    parser.add_argument("--mot20", action="store_true", default=False)
    parser.add_argument("--name", type=str, default="botsort_run")
    parser.add_argument("--ablation", action="store_true", default=False)
    parser.add_argument("--save_result", action="store_true", default=True)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    json_path = f"Detection/{args.scene_id}/{args.camera_id}.json"
    frame_detections = load_full_detections_per_frame(json_path)
    print(f"Loaded detections for {len(frame_detections)} frames.")

    if args.scene_id == "Warehouse_016":
        args.dataset = "Val"
    elif args.scene_id == "Warehouse_017" or args.scene_id == "Warehouse_018" or args.scene_id == "Warehouse_019" or args.scene_id == "Warehouse_020":
        args.dataset = "Test"
    
    if args.preview:
        preview_frames(frame_detections, args.max_preview)
    
    save_dir = f"Tracking/Singlecamera/{args.scene_id}/{args.camera_id}"
    os.makedirs(save_dir, exist_ok=True)
    run_botsort_tracking(frame_detections, args, save_dir=save_dir)


