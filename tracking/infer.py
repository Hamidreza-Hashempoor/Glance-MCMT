import os
import sys
import json
from datetime import datetime
from multiprocessing import Pool
import subprocess
import glob
import tarfile
import argparse

sys.path.append("tracking")
sys.path.append("tracking/src")
import run

"""
This file contains functions to execute offline tracking.
"""

# Single camera people tracking
def scpt(tracking_params={}):
    # distributed SCPT processing by simply using multiprocessing pool.
    global scene_id
    global camera_ids
    global exp_root
    global tracking_parameters
    tracking_parameters = tracking_params

    # num_processes = 5  # Could be more than 5, but it depends on machine instance
    # p = Pool(num_processes)
    # result = p.map(single_tracking, camera_ids)

    # Debug 1 for the single thread
    # result = []
    # for cam_id in camera_ids:
    #     r = single_tracking(cam_id)
    #     result.append(r)

    result = []
    for cam_id in camera_ids:
        if cam_id == 0:
            json_path = f'Tracking/{scene_id}/Camera_tracking_results.json'
        else:
            json_path = f'Tracking/{scene_id}/Camera_{cam_id:02d}_tracking_results.json'

        if os.path.exists(json_path):
            print(f"[INFO] {json_path} exists — skipping single camera track")
        else:
            print(f"[INFO] {json_path} not found — running single camera track")
            r = single_tracking(cam_id)
            result.append(r)

    run.correct_scpt_result(scene_id=scene_id, json_dir=exp_root, out_dir=exp_root,
        tracking_params=tracking_params)

def single_tracking(cam_id):
    global scene_id
    global embed_root
    global exp_root
    global tracking_parameters


    # run.run_scpt(feature_data_root=f'{embed_root}/scene_{scene_id:03d}/camera_{cam_id:04d}', out_dir=exp_root,
    #             tracking_params=tracking_parameters)
    if cam_id == 0:
        print(f"Started a background process to {scene_id} Camera\n")
        run.run_scpt(feature_data_root=f'{embed_root}/{scene_id}/Camera', out_dir=exp_root,
                    tracking_params=tracking_parameters)
    else:
        print(f"Started a background process to {scene_id} Camera_{cam_id}\n")
        run.run_scpt(feature_data_root=f'{embed_root}/{scene_id}/Camera_{cam_id:02d}', out_dir=exp_root,
                tracking_params=tracking_parameters)
    return

def get_camera_ids(scene_id, json_f="tracking/config/scene_2_camera_id_file.json"):
    with open(json_f) as f:
        scene2camera = json.load(f)
    camera_ids = []
    for scene_camera in scene2camera:
        # if scene_camera["scene_name"] == f"scene_{scene_id:03d}":
        if scene_camera["scene_name"] == scene_id:
            camera_ids = scene_camera["camera_ids"]
            break
    if not camera_ids:
        raise ValueError(f"No camera IDs found for scene '{scene_id}' in {json_f}")
    return camera_ids


# Multi camera tracking, aka ReID
def mcpt(tracking_params={}):
    global scene_id
    global exp_root
    global tracking_parameters
    tracking_parameters = tracking_params

    run.run_mcpt(scene_id=scene_id, json_dir=exp_root, out_dir=exp_root, tracking_params=tracking_parameters)
    run.correct_mcpt_result(scene_id=scene_id, json_dir=exp_root, out_dir=exp_root, tracking_params=tracking_parameters)

def run_tracking(scene, embed, output, debug=True, tracking_params={}):
    """
    Main routine
    """
    global scene_id
    global embed_root
    global exp_root
    global output_root
    global camera_ids
    global exec_scpt
    global exec_mcpt

    if debug:
        print(f"### tracking parameters: {tracking_params}", flush=True)

    scene_id = scene
    embed_root = embed
    camera_ids = get_camera_ids(scene_id)
    print(f"Target scene ID: {scene_id}, camera IDs: {camera_ids}")

    # Configure output directory
    # exp_root = os.path.join(output, f"scene_{scene_id:03d}")
    exp_root = os.path.join(output, scene_id)
    output_root = exp_root

    # Execute SCPT (Single Camera People Tracking)
    if exec_scpt:
        scpt_started = datetime.now()
        print(f"Start SCPT: {scpt_started}", flush=True)
        scpt(tracking_params=tracking_params)
        print(f"SCPT finished. Elapsed: {datetime.now()-scpt_started}", flush=True)

    # Execute MCPT (Multi Camera People Tracking) aka ReID
    if exec_mcpt:
        mcpt_started = datetime.now()
        print(f"Start MCPT: {mcpt_started}")
        mcpt(tracking_params=tracking_params)
        print(f"MCPT finished. Elapsed: {datetime.now()-mcpt_started}", flush=True)


def get_parameters_to_scene(scene_id, param_file):
    if not os.path.isfile(param_file):
        print(f"'parameters_per_scene file does not exist. {param_file}")
        return {}

    sys.path.append("tracking/config")
    import parameters_per_scene as pps

    # scene = int(scene_id)
    scene = scene_id
    if scene in pps.parameters_per_scene:
        return pps.parameters_per_scene[scene]
    else:
        return {}

def get_args():
    parser = argparse.ArgumentParser(description='Offline Tracker Inferencing app.')
    parser.add_argument('-s', '--scene', type=str, default="Warehouse_016", required=True)
    parser.add_argument('-o', '--output', default="Tracking", type=str)
    parser.add_argument('-all', '--exec_all', action='store_true')
    parser.add_argument('-scpt', '--exec_scpt', action='store_true')
    parser.add_argument('-mcpt', '--exec_mcpt', action='store_true')

    return parser.parse_args()

if __name__ == "__main__":
    global exec_scpt
    global exec_mcpt

    args = get_args()

    if args.exec_all or (not (args.exec_scpt | args.exec_mcpt)):
        exec_scpt = exec_mcpt = True
    else:
        exec_scpt = exec_mcpt = False
        if args.exec_scpt:
            exec_scpt = True
        if args.exec_mcpt:
            exec_mcpt = True

    # Default tracking parameter
    default_tracking_parameters = {
        "epsilon_scpt": 0.10, "time_period":3,"epsilon_mcpt": 0.37, "short_track_th":120,
        "keypoint_condition_th":1, "replace_similarity_by_wcoordinate":True, "distance_type":"min",
        "distance_th":10, "sim_th":0.85, "delete_gid_th":5000, "split_cls":False,
        }

    scene = args.scene
    param_file = "tracking/config/parameters_per_scene.py"
    parameters = get_parameters_to_scene(scene, param_file)
    if len(parameters) > 0:
        tracking_parameters = parameters["tracking_parameters"]
    else:
        # Empty parameters to the scene, so use the default parameters.
        tracking_parameters = default_tracking_parameters
    embed_path = f"EmbedFeature"

    # Run offline tracking
    run_tracking(scene=scene, embed=embed_path, output=args.output, tracking_params=tracking_parameters)
