import os
import json
import argparse
import numpy as np
import cv2
import re
from sklearn.preprocessing import normalize
from tools.utils_25_revised import compute_feature_similarity, merge_redundant_global_ids, assign_initial_global_ids,\
ensure_all_coords_have_features, filter_expressive_features, extend_global_tracklets,\
find_best_matching_global_id, extract_unassigned_coords_features_with_frames, ensure_all_frame_coords_have_features,\
extend_global_tracklets_with_full_data_fixed, match_unassigned_to_assigned, convert_keys_to_strings,\
convert_to_framewise_multicam_dict, write_framewise_txt, load_predictions, process_all_frames


def save_args_as_json(args, output_dir):
    args_dict = vars(args)
    save_path = os.path.join(output_dir, "args.json")
    with open(save_path, "w") as f:
        json.dump(args_dict, f, indent=2)

def get_next_experiment_id(scene_path):
    existing = [d for d in os.listdir(scene_path) if os.path.isdir(os.path.join(scene_path, d))]
    exp_nums = [
        int(re.match(r"exp(\d+)", d).group(1))
        for d in existing
        if re.match(r"exp\d+", d)
    ]
    next_id = max(exp_nums, default=0) + 1
    return f"exp{next_id}"

def save_results_and_visualizations(scene_path, assigned_global_map, exp, args):
    exp_path = os.path.join(scene_path, exp)
    os.makedirs(exp_path, exist_ok=True)

    if not args.save_result:
        return

    ## 1. Save JSON result
    framewise_json = convert_to_framewise_multicam_dict(assigned_global_map, args.total_frames)
    output_result_dir = os.path.join(exp_path, "output_result")
    os.makedirs(output_result_dir, exist_ok=True)

    save_path = os.path.join(output_result_dir, f"final_track_{os.path.basename(scene_path)}.json")
    with open(save_path, "w") as f:
        json.dump(framewise_json, f, indent=2)

    ## 2. Save track1-style TXT
    write_framewise_txt(exp_path, assigned_global_map, args.total_frames)

    ## 3. Visualization (draw and save images with boxes)

    predictions = load_predictions(save_path)  # Load the prediction JSON
    base_video_path = f"AIC25_Track1/{args.dataset}/{args.scene_id}/videos"  # Path to your base video folder
    output_vis_root = os.path.join("Tracking", "Multicamera", args.scene_id, exp, "output_vis") # Path to save output images
    camera_ids = [d for d in os.listdir(f"AIC25_Track1/{args.dataset}/{args.scene_id}/videos") if os.path.isdir(os.path.join(f"AIC25_Track1/{args.dataset}/{args.scene_id}/videos", d))]

    # Process all frames for all cameras
    print(f"Start drawing a few samples on : {output_vis_root}")
    process_all_frames(predictions, base_video_path, output_vis_root, camera_ids, args)


def glance_tracklets_for_global_id(scene_path, args):
    frame_range = (args.start_glance_frame, args.end_glance_frame)

    per_camera_tracklets = []
    camera_folders = sorted(os.listdir(scene_path))
    

    for cam_folder in camera_folders:
        cam_file = f"{scene_path}/{cam_folder}/fixed_{cam_folder}.json"
        
        if not os.path.exists(cam_file):
            raise FileNotFoundError(f"Expected file not found: {cam_file}")

        with open(cam_file, 'r') as f:
            data = json.load(f)
        if args.limit_frame > 0:
            data = dict(sorted(data.items(), key=lambda x: int(x[0]))[:args.limit_frame])
        
        object_coords = {}        # sc_id → list of 3D coords
        object_features = {}      # sc_id → list of feature vectors

        for frame_str, objs in data.items():
            frame_id = int(frame_str)
            if frame_id < frame_range[0] or frame_id > frame_range[1]:
                continue

            for obj in objs:
                sc_id = obj["object sc id"]
                coord = obj["3d location"]
                feat_path = obj.get("feature path", None)

                # Add coordinates
                if sc_id not in object_coords:
                    object_coords[sc_id] = []
                object_coords[sc_id].append(coord)

                # Add feature vector if available
                if feat_path and feat_path.lower().endswith(".npy"):
                    if sc_id not in object_features:
                            object_features[sc_id] = []
                    try:
                        feat_full_path = os.path.join("EmbedFeature", feat_path)
                        feature = np.load(feat_full_path)

                        

                        object_features[sc_id].append(feature)
                    except Exception as e:
                        print(f"Warning: Failed to load feature for sc_id={sc_id} from {feat_path}: {e}")
                        
        # After collecting initial coords/features from frame range:
        ensure_all_coords_have_features(data, object_coords, object_features, cam_folder)
        object_features = filter_expressive_features(object_features, cam_folder, args)


        cam_tracklets = []
        for sc_id in object_coords:
            coords = object_coords[sc_id]
            avg_coord = np.mean(coords, axis=0)

            # Optionally average feature (or leave list if needed)
            features = object_features.get(sc_id, [])
            avg_feature = np.mean(features, axis=0) if features else None

            tracklet_data = {
                "scene_id": os.path.basename(scene_path),
                "camera_id": cam_folder,
                "local_id": sc_id,
                "coords": object_coords[sc_id],
                "features": object_features.get(sc_id, [])
            }

            cam_tracklets.append(tracklet_data)
        per_camera_tracklets.append(cam_tracklets)  # ← outside per-camera loop
    globalized_tracklets = assign_initial_global_ids(per_camera_tracklets, args)
    globalized_tracklets_merged = merge_redundant_global_ids(globalized_tracklets, args)
    return globalized_tracklets_merged

def assign_to_existing_global_ids(scene_path, global_tracklets, args):
    frame_start = args.end_glance_frame + 1
    camera_folders = sorted(os.listdir(scene_path))
    assigned_tracklets = []
    assigned_keys = set((tr["camera_id"], tr["local_id"]) for tr in global_tracklets)
    new_assignments = True

    stagnant_rounds = 0
    last_unassigned_count = float('inf')

    camera_data_cache = {}  # Cache for each camera

    # Cache each camera's full data once
    camera_data_cache = {}
    for cam_folder in camera_folders:
        cam_file = f"{scene_path}/{cam_folder}/fixed_{cam_folder}.json"
        if not os.path.exists(cam_file):
            raise FileNotFoundError(f"Expected file not found: {cam_file}")
        with open(cam_file, 'r') as f:
            data = camera_data_cache[cam_folder] = json.load(f)
        if args.limit_frame > 0:
            camera_data_cache[cam_folder] = dict(sorted(data.items(), key=lambda x: int(x[0]))[:args.limit_frame])

    # 🔹 Step 1 (initial): Extend ALL initial global tracklets before loop
    print(f"[INIT] Initial number of global tracklets: {len(global_tracklets)}")
    print("[INIT] Current global tracklets before extension:")

    # Track which global IDs were extended
    extended_keys = set()
    for tr in global_tracklets:
        print(f"  ↪ global_id={tr['global_id']} from {tr['camera_id']} - local_id={tr['local_id']}")
    for cam_folder in camera_folders:
        data = camera_data_cache[cam_folder]
        global_tracklets, count = extend_global_tracklets_with_full_data_fixed(
            global_tracklets, data, cam_folder, args
        )
        # Collect extended global_id, camera_id, local_id triplets
        for tr in global_tracklets:
            key = (tr['camera_id'], tr['local_id'])
            if key not in extended_keys and len(tr.get("coords", [])) > args.end_glance_frame:
                extended_keys.add(key)
       ###
    print("[INIT] Global tracklets after extension:")
    for tr in global_tracklets:
        if (tr['camera_id'], tr['local_id']) in extended_keys:
            coords = tr.get("coords", [])
            if coords:
                print(f"  ✅ global_id={tr['global_id']} ({tr['camera_id']} - local_id={tr['local_id']}) "
                    f"extended to {len(coords)} frames, from frame {coords[0][0]} to {coords[-1][0]}")
    
    # Make maps
    assigned_global_map = {}

    for tr in global_tracklets:
        cam_id = tr["camera_id"]
        local_id = tr["local_id"]
        global_id = tr["global_id"]
        coords = tr.get("coords", [])  # list of (frame_id, coord)

        assigned_global_map[(cam_id, local_id)] = {
            "global_id": global_id,
            # "frames": coords,
            "object": tr
        }

    unassigned_local_map = {}  # key: (camera_id, local_id), value: list of (frame_id, obj)

    for cam_folder in camera_folders:
        data = camera_data_cache[cam_folder]
        for frame_str, objs in data.items():
            frame_id = int(frame_str)
            for obj in objs:
                sc_id = obj.get("object sc id")
                if sc_id is None:
                    continue
                key = (cam_folder, sc_id)
                if key in assigned_global_map:
                    continue  # Already assigned to a global ID
                if key not in unassigned_local_map:
                    unassigned_local_map[key] = []
                unassigned_local_map[key].append((frame_id, obj))

    final_assigned_global_map, assigned_global_map = match_unassigned_to_assigned(
                                                                                unassigned_local_map,
                                                                                assigned_global_map,
                                                                                args
                                                                            )
    
    return final_assigned_global_map, assigned_global_map


def main():
    parser = argparse.ArgumentParser(description="Assign global IDs across multiple camera views using 3D coordinates.")
    parser.add_argument(
        "-s", "--scene_id", type=str, required=True,
        help="Scene ID (e.g., Warehouse_016)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Optional path to save output JSON"
    )
    parser.add_argument("--start_glance_frame", type=int, default=1, help="Start frame (inclusive)")
    parser.add_argument("--end_glance_frame", type=int, default=25, help="End frame (inclusive)")
    parser.add_argument("--total_frames", type=int, default=9000, help="End frame (inclusive)")
    parser.add_argument("--eps", type=float, default=1.0, help="DBSCAN distance threshold (in meters)")
    parser.add_argument("--min_samples", type=int, default=1, help="DBSCAN min_samples for clustering")
    parser.add_argument("--representitive_feature_len", type=int, default=20, help="Len of representitive features")
    parser.add_argument("--metric", type=str, default="cosine", help="Feature similarity metric")
    parser.add_argument("--trajectory_dist_thresh_init", type=float, default=1.5, help="Distance threshold for trajectories init G ID")
    parser.add_argument("--feature_similarity_thresh_init", type=float, default=0.65, help="Similarity threshold for trajectories init G ID")
    parser.add_argument("--trajectory_dist_thresh_merge", type=float, default=1.2, help="Distance threshold for merging init trajectories G ID")
    parser.add_argument("--feature_similarity_thresh_merge", type=float, default=0.55, help="Similarity threshold for merging init trajectories G ID")

    parser.add_argument("--trajectory_dist_thresh_rest", type=float, default=1.5, help="Distance threshold for trajectories init G ID")
    parser.add_argument("--feature_similarity_thresh_rest", type=float, default=0.70, help="Similarity threshold for merging init trajectories G ID")
    parser.add_argument("--min_shared_frames", type=int, default=6, help="Overlap between globals and locals")
    parser.add_argument("--max_stagnant_rounds", type=int, default=1, help="Iteration over the global matching")



    parser.add_argument(
        "--limit_frame", type=int, default=-1,
        help="Limit number of frames to process (for debugging). Use -1 to process all frames."
    )
    parser.add_argument(
        "--dataset", type=str, default="Val",
        help="Dataset type ,E.g Val or Test"
    )

    parser.add_argument(
        "--frame_num_save", type=int, default=100,
        help="Save num of frames for visualization"
    )

    parser.add_argument("--save_result", action="store_true", default=True)
    parser.add_argument("--force_match", action="store_true", default=False)
    
    
    args = parser.parse_args()
    # if args.save_result:
    #     base_path = os.path.join("Tracking", "Multicamera", args.scene_id)
    #     os.makedirs(base_path, exist_ok=True)
    #     exp = get_next_experiment_id(base_path)
    #     exp_path = os.path.join(base_path, exp)
    #     os.makedirs(exp_path, exist_ok=True)
    #     # Save args.json
    #     save_args_as_json(args, exp_path)


    print(f"Loading tracklets from: {args.scene_id}")
    scene_path = f"Tracking/Singlecamera/{args.scene_id}"
    tracklets = glance_tracklets_for_global_id(scene_path, args)
    
    print(f"Assign the rest of tracklets to the global tracklets")
    _, assigned_global_map = assign_to_existing_global_ids(scene_path, tracklets, args)
    
    # Save final global tracklets
    
    if args.save_result:
        base_path = os.path.join("Tracking", "Multicamera", args.scene_id)
        os.makedirs(base_path, exist_ok=True)
        exp = get_next_experiment_id(base_path)
        exp_path = os.path.join(base_path, exp)
        os.makedirs(exp_path, exist_ok=True)
        # Save args.json
        save_args_as_json(args, exp_path)

        # Save results and visualizations
        save_results_and_visualizations(base_path, assigned_global_map, exp, args)

    print("Done.")

if __name__ == "__main__":
    main()