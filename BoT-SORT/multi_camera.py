import os
import json
import argparse
import numpy as np

from sklearn.preprocessing import normalize
from tools.utils_25 import compute_feature_similarity, merge_redundant_global_ids, assign_initial_global_ids,\
ensure_all_coords_have_features, filter_expressive_features, extend_global_tracklets, extend_global_tracklets_with_full_data_fixed,\
find_best_matching_global_id, extract_unassigned_coords_features_with_frames, ensure_all_frame_coords_have_features





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

    while True:
        # Cache camera data if not already loaded
        for cam_folder in camera_folders:
            if cam_folder not in camera_data_cache:
                cam_file = f"{scene_path}/{cam_folder}/fixed_{cam_folder}.json"
                if not os.path.exists(cam_file):
                    raise FileNotFoundError(f"Expected file not found: {cam_file}")
                with open(cam_file, 'r') as f:
                    camera_data_cache[cam_folder] = json.load(f)

        # STEP 0: Check unassigned keys
        all_keys = set()
        for cam_folder in camera_folders:
            data = camera_data_cache[cam_folder]  # ✅ Use cached data
            for frame_objs in data.values():
                for obj in frame_objs:
                    sc_id = obj.get("object sc id")
                    if sc_id is not None:
                        all_keys.add((cam_folder, sc_id))

        assigned_keys = {(tr["camera_id"], tr["local_id"]) for tr in global_tracklets}
        unassigned_keys = all_keys - assigned_keys
        unassigned_count = len(unassigned_keys)

        print(f"[STATUS] {unassigned_count} unassigned local IDs remaining")

        if unassigned_count == 0:
            print("[STATUS] All local IDs have been assigned to global IDs. Exiting loop.")
            break

        # Track if progress has stalled
        if unassigned_count >= last_unassigned_count:
            stagnant_rounds += 1
        else:
            stagnant_rounds = 0
        last_unassigned_count = unassigned_count

        if stagnant_rounds >= 5:
            print("[NOTICE] Matching process has stalled for 5 iterations. Forcing fallback matching...")
            break

        # STEP 1: Extend all existing global IDs with full per-frame data
        for cam_folder in camera_folders:
            cam_file = f"{scene_path}/{cam_folder}/fixed_{cam_folder}.json"
            if not os.path.exists(cam_file):
                raise FileNotFoundError(f"Expected file not found: {cam_file}")
            # with open(cam_file, 'r') as f:
            #     data = json.load(f)
            data = camera_data_cache[cam_folder]

            global_tracklets, count = extend_global_tracklets_with_full_data_fixed(global_tracklets, data, cam_folder, args)
            print(f"[{cam_folder}] Extended {count} global tracklets")

        # STEP 2: Try to match new local IDs to existing global tracklets
        for cam_folder in camera_folders:
            cam_file = f"{scene_path}/{cam_folder}/fixed_{cam_folder}.json"
            # with open(cam_file, 'r') as f:
            #     data = json.load(f)
            data = camera_data_cache[cam_folder]

            # object_coords, object_features = extract_coords_features_from_frames(data, cam_folder, args.end_glance_frame)
            global_lookup_keys = {(tr["camera_id"], tr["local_id"]) for tr in global_tracklets}
            # object_coords, object_features = extract_unassigned_coords_features(
            #     data, cam_folder, args.end_glance_frame + 1, global_lookup_keys
            # )
            object_coords, object_features = extract_unassigned_coords_features_with_frames(
                data, cam_folder, args.end_glance_frame + 1, global_lookup_keys
            )
            ensure_all_frame_coords_have_features(data, object_coords, object_features, cam_folder)
            object_features = filter_expressive_features(object_features, cam_folder, args)

            for sc_id in object_coords:
                key = (cam_folder, sc_id)
                if key in assigned_keys:
                    continue  # Already handled in step 1

                coords = object_coords[sc_id]
                features = object_features.get(sc_id, [])

                best_gid, best_traj_dist, best_feat_sim = find_best_matching_global_id(
                    coords, features, cam_folder, global_tracklets, args
                )

                if best_gid is not None:
                    new_global_track = {
                        "scene_id": os.path.basename(scene_path),
                        "camera_id": cam_folder,
                        "local_id": sc_id,
                        "global_id": best_gid,
                        "coords": coords,               # assumed to be (frame_id, coord) list
                        "features": features,          # already extracted
                        "feature_paths": [],
                        "3d bounding box scale_sequence": [],
                        "3d bounding box rotation_sequence": [],
                        "2d bounding box visible_sequence": []
                    }

                    # Scan original data again to collect additional attributes
                    for frame_str, objs in data.items():
                        for obj in objs:
                            if obj.get("object sc id") != sc_id:
                                continue

                            # Optional: object type
                            if "object type" in obj and "object type" not in new_global_track:
                                new_global_track["object type"] = obj["object type"]

                            # Optional: feature path
                            if "feature path" in obj:
                                new_global_track["feature_paths"].append(obj["feature path"])

                            # Frame-based metadata sequences
                            for attr in ["3d bounding box scale", "3d bounding box rotation", "2d bounding box visible"]:
                                if attr in obj:
                                    new_global_track[f"{attr}_sequence"].append(obj[attr])

                    # Only once per matched tracklet
                    global_tracklets.append(new_global_track)
                    assigned_keys.add(key)
                    # new_assignments = True

    # FINAL FALLBACK: force assign all remaining local IDs
    print("[FALLBACK] Assigning all remaining unassigned local tracklets using fallback strategy...")

    for cam_folder in camera_folders:
        cam_file = f"{scene_path}/{cam_folder}/fixed_{cam_folder}.json"
        with open(cam_file, 'r') as f:
            data = json.load(f)

        # STEP 1: Re-extend all global_tracklets (ensures per-frame data is updated)
        global_tracklets, count = extend_global_tracklets_with_full_data_fixed(global_tracklets, data, cam_folder, args)
        print(f"[{cam_folder}] Extended {count} global tracklets (for fallback)")

    # STEP 2: Re-attempt matching all unassigned local IDs
    for cam_folder in camera_folders:
        cam_file = f"{scene_path}/{cam_folder}/fixed_{cam_folder}.json"
        with open(cam_file, 'r') as f:
            data = json.load(f)

        global_lookup_keys = {(tr["camera_id"], tr["local_id"]) for tr in global_tracklets}
        object_coords, object_features = extract_unassigned_coords_features_with_frames(
            data, cam_folder, frame_start, global_lookup_keys
        )
        ensure_all_frame_coords_have_features(data, object_coords, object_features, cam_folder)
        object_features = filter_expressive_features(object_features, cam_folder, args)

        for sc_id in object_coords:
            key = (cam_folder, sc_id)
            if key in global_lookup_keys:
                continue  # Already assigned

            coords = object_coords[sc_id]
            features = object_features.get(sc_id, [])

            best_gid, best_traj_dist, best_feat_sim = find_best_matching_global_id(
                coords, features, cam_folder, global_tracklets, args, force_assign=True  # <== Fallback activated
            )

            if best_gid is not None:
                print(f"[FALLBACK-ASSIGN] Local ID {sc_id} from camera '{cam_folder}' assigned to global ID {best_gid} "
                                    f"(trajectory distance = {best_traj_dist:.3f}, feature similarity = {best_feat_sim:.3f})")
                new_global_track = {
                    "scene_id": os.path.basename(scene_path),
                    "camera_id": cam_folder,
                    "local_id": sc_id,
                    "global_id": best_gid,
                    "coords": coords,
                    "features": features,
                    "feature_paths": [],
                    "3d bounding box scale_sequence": [],
                    "3d bounding box rotation_sequence": [],
                    "2d bounding box visible_sequence": []
                }

                for frame_str, objs in data.items():
                    for obj in objs:
                        if obj.get("object sc id") != sc_id:
                            continue
                        if "object type" in obj and "object type" not in new_global_track:
                            new_global_track["object type"] = obj["object type"]
                        if "feature path" in obj:
                            new_global_track["feature_paths"].append(obj["feature path"])
                        for attr in ["3d bounding box scale", "3d bounding box rotation", "2d bounding box visible"]:
                            if attr in obj:
                                new_global_track[f"{attr}_sequence"].append(obj[attr])

                global_tracklets.append(new_global_track)
                assigned_keys.add(key)

    print("[FALLBACK COMPLETED] All local IDs forcibly matched to global IDs.")
    return global_tracklets


# def assign_to_existing_global_ids(scene_path, global_tracklets, args):
#     frame_start = args.end_glance_frame + 1
#     camera_folders = sorted(os.listdir(scene_path))
#     assigned_tracklets = []
#     # Organize existing global tracklets by (camera_id, local_id)
#     global_lookup = {}
#     for tr in global_tracklets:
#         key = (tr["camera_id"], tr["local_id"])
#         if key not in global_lookup:
#             global_lookup[key] = tr

#     for cam_folder in camera_folders:
#         cam_file = f"{scene_path}/{cam_folder}/fixed_{cam_folder}.json"
#         if not os.path.exists(cam_file):
#             raise FileNotFoundError(f"Expected file not found: {cam_file}")

#         with open(cam_file, 'r') as f:
#             data = json.load(f)
#         global_tracklets, extended_count = extend_global_tracklets_with_full_data(global_tracklets, data, cam_folder, args)
#         print(f"[{cam_folder}] Extended {extended_count} global tracklets with full per-frame object data")


#         object_coords = {}
#         object_features = {}

#         for frame_str, objs in data.items():
#             frame_id = int(frame_str)
#             if frame_id < frame_start:
#                 continue  # Only process frames after the glance phase

#             for obj in objs:
#                 sc_id = obj["object sc id"]
#                 coord = obj["3d location"]
#                 feat_path = obj.get("feature path", None)

#                 if sc_id not in object_coords:
#                     object_coords[sc_id] = []
#                 object_coords[sc_id].append(coord)

#                 if feat_path and feat_path.lower().endswith(".npy"):
#                     if sc_id not in object_features:
#                         object_features[sc_id] = []
#                     try:
#                         feat_full_path = os.path.join("EmbedFeature", feat_path)
#                         feature = np.load(feat_full_path)
#                         object_features[sc_id].append(feature)
#                     except Exception as e:
#                         print(f"Warning: Failed to load feature for sc_id={sc_id} from {feat_path}: {e}")

#         ensure_all_coords_have_features(data, object_coords, object_features, cam_folder)
#         object_features = filter_expressive_features(object_features, cam_folder, args)

#         global_tracklets, extended_count = extend_global_tracklets(global_tracklets, object_coords, object_features, cam_folder)


#         for sc_id in object_coords:
#             coords = object_coords[sc_id]
#             features = object_features.get(sc_id, [])
#             key = (cam_folder, sc_id)

#             if key in global_lookup:
#                 # Case 1: Local ID seen in glance → extend its tracklet
#                 global_lookup[key]["coords"].extend(coords)
#                 global_lookup[key]["features"].extend(features)
#             else:
#                 # Case 2: New local ID → match to best global tracklet (not from same cam)
#                 best_gid = None
#                 best_traj_dist = float('inf')
#                 best_feat_sim = -1

#                 for global_tr in global_tracklets:
#                     if global_tr["camera_id"] == cam_folder:
#                         continue  # Skip same camera (new ID must come from another view)

#                     g_coords = global_tr["coords"]
#                     min_len = min(len(coords), len(g_coords))
#                     if min_len == 0:
#                         continue

#                     traj_sim = np.mean(np.linalg.norm(np.array(coords[:min_len]) - np.array(g_coords[:min_len]), axis=1))
#                     if traj_sim < args.trajectory_dist_thresh_rest:
#                         feat_sim = compute_feature_similarity(features, global_tr["features"], args)
#                         if feat_sim > args.feature_similarity_thresh_rest:
#                             if traj_sim < best_traj_dist or (traj_sim == best_traj_dist and feat_sim > best_feat_sim):
#                                 best_gid = global_tr["global_id"]
#                                 best_traj_dist = traj_sim
#                                 best_feat_sim = feat_sim

#                 if best_gid is not None:
#                     # Append new view of the same global ID
#                     global_tracklets.append({
#                         "scene_id": os.path.basename(scene_path),
#                         "camera_id": cam_folder,
#                         "local_id": sc_id,
#                         "coords": coords,
#                         "features": features,
#                         "global_id": best_gid
#                     })

#     return global_tracklets

def save_global_mapping(tracklets, output):
    mapping = {}
    for t in tracklets:
        key = f'{t["scene_id"]}_{t["camera_id"]}_{t["local_id"]}'
        mapping[key] = t["global_id"]

    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, 'w') as f:
        json.dump(mapping, f, indent=2)

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
    parser.add_argument("--end_glance_frame", type=int, default=80, help="End frame (inclusive)")
    parser.add_argument("--eps", type=float, default=1.0, help="DBSCAN distance threshold (in meters)")
    parser.add_argument("--min_samples", type=int, default=1, help="DBSCAN min_samples for clustering")
    parser.add_argument("--representitive_feature_len", type=int, default=20, help="Len of representitive features")
    parser.add_argument("--metric", type=str, default="cosine", help="Feature similarity metric")
    parser.add_argument("--trajectory_dist_thresh_init", type=float, default=2.5, help="Distance threshold for trajectories init G ID")
    parser.add_argument("--feature_similarity_thresh_init", type=float, default=0.65, help="Similarity threshold for trajectories init G ID")
    parser.add_argument("--trajectory_dist_thresh_merge", type=float, default=1.2, help="Distance threshold for merging init trajectories G ID")
    parser.add_argument("--feature_similarity_thresh_merge", type=float, default=0.55, help="Similarity threshold for merging init trajectories G ID")

    parser.add_argument("--trajectory_dist_thresh_rest", type=float, default=1.2, help="Distance threshold for trajectories init G ID")
    parser.add_argument("--feature_similarity_thresh_rest", type=float, default=0.60, help="Similarity threshold for merging init trajectories G ID")

    args = parser.parse_args()

    print(f"Loading tracklets from: {args.scene_id}")
    scene_path = f"Tracking/Singlecamera/{args.scene_id}"
    tracklets = glance_tracklets_for_global_id(scene_path, args)
    
    print(f"Assign the rest of tracklets to the global tracklets")
    tracklets = assign_to_existing_global_ids(scene_path, tracklets, args)
    # Save final global tracklets
    output_path = os.path.join("Tracking", "Multicamera", args.scene_id, f"final_track_{args.scene_id}.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(tracklets, f, indent=2)

    print(f"[DONE] Saved final global tracklets to: {output_path}")

    print("Clustering tracklets based on 3D coordinates...")


    # tracklets_with_global_ids = assign_global_ids(tracklets, eps=args.eps, min_samples=args.min_samples)

    # print(f"Saving global ID mapping to: {args.output}")
    # if args.output is not None:
    #     args.output = args.output
    # else:
    #     args.output = f"Tracking/Multicamera/{args.scene_id}"
    # save_global_mapping(tracklets_with_global_ids, args.output)
    print("Done.")

if __name__ == "__main__":
    main()