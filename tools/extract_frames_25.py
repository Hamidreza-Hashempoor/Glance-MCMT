import os
import json
import numpy as np
import PIL.Image as Image
import cv2
from multiprocessing import Pool
from sys import stdout
import argparse
import os.path as osp
import shutil

def make_parser():
    parser = argparse.ArgumentParser("extract frame")
    parser.add_argument("root_path", type=str, default=None)
    parser.add_argument("-s", "--scene", type=str, default=None)
    return parser

args = make_parser().parse_args()
# data_root = osp.join(args.root_path, "AIC25_Track1")
data_root = args.root_path
scene = args.scene

fprint, endl = stdout.write, "\n"

IMAGE_FORMAT = ".jpg"


def video2image(parameter_set):
    scenario, camera, camera_dir = parameter_set
    fprint(f"[Processing] {scenario} {camera}{endl}")
    imgs_dir = f"{camera_dir}/Frame"
    if os.path.exists(imgs_dir):
        # shutil.rmtree(imgs_dir)  # Delete the entire directory
        fprint(f"{camera_dir}/Frame already exists. If you want to regenrate, manually remove it.")
        return

    os.makedirs(imgs_dir)        # Create a fresh new one
    print("camera_dir:" + camera_dir)

    video_file = None
    for file in os.listdir(camera_dir):
        if file.endswith(".mp4"):
            video_file = os.path.join(camera_dir, file)
            break

    if video_file is None:
        fprint(f"[Error] No video found in {camera_dir}{endl}")
        return
    cap = cv2.VideoCapture(video_file)
    current_frame = 1
    ret, frame = cap.read()
    while ret:
        frame_file_name = f"{str(current_frame).zfill(6)}{IMAGE_FORMAT}"
        cv2.imwrite(f"{imgs_dir}/{frame_file_name}", frame)
        ret, frame = cap.read()
        current_frame += 1
    fprint(f"[Done] {scenario} {camera}{endl}")


def main():
    parameter_sets = []
    scenario_dir = osp.join(data_root, scene +"/videos")
    if not osp.exists(scenario_dir):
        print("No such scenario_dir.")
    else:
        for fname in os.listdir(scenario_dir):
            if not fname.endswith(".mp4"):
                continue

            video_name = osp.splitext(fname)[0]  # e.g., "1.mp4" → "1"
            video_path = osp.join(scenario_dir, fname)
            target_dir = osp.join(scenario_dir, video_name)
            target_path = osp.join(target_dir, fname)

            if not osp.exists(target_dir):
                os.makedirs(target_dir)


            # Move the video into the folder
            shutil.move(video_path, target_path)
            print(f"Moved {fname} → {target_path}")

    cameras = os.listdir(scenario_dir)
    for each_camera in cameras:
        cam = each_camera
        if "map" in each_camera:
            continue
        camera_dir = f"{scenario_dir}/{each_camera}"                
        parameter_sets.append(
            [scene, each_camera, camera_dir]
        )

    pool = Pool(processes=len(parameter_sets))
    pool.map(video2image, parameter_sets)
    pool.close()


if __name__ == "__main__":
    main()

