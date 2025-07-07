import os
import os.path as osp
import json
import argparse
import os.path as osp
import shutil

CLASS_NAME_TO_ID = {
    "Person": 1,
    "Forklift": 2,
    "NovaCarter": 3,
    "Transporter": 4,
    "FourierGR1T2": 5,
    "AgilityDigit": 6,
    # Add more if needed
}

def make_parser():
    parser = argparse.ArgumentParser("extract gt")
    parser.add_argument("root_path", type=str, default=None)
    parser.add_argument("-s", "--scene", type=str, default=None)
    return parser

args = make_parser().parse_args()
# data_root = osp.join(args.root_path, "AIC25_Track1")
data_root = args.root_path
scene = args.scene

def create_det_txt_from_groundtruth(base_dir, scene, gt_file_name='ground_truth.json'):
    gt_path = osp.join(base_dir, scene, gt_file_name)
    video_root = osp.join(base_dir, scene, "videos")

    with open(gt_path, 'r') as f:
        gt_data = json.load(f)

    for camera in os.listdir(video_root):
        if not camera.startswith("Camera"):
            continue
        camera_dir = osp.join(video_root, camera)
        det_dir = osp.join(camera_dir, "det")
        os.makedirs(det_dir, exist_ok=True)
        det_txt_path = osp.join(det_dir, "det.txt")

        with open(det_txt_path, 'w') as det_file:
            for frame_id_str, objects in gt_data.items():
                frame_id = int(frame_id_str)
                for obj in objects:
                    bboxes = obj.get("2d bounding box visible", {})
                    if camera in bboxes:
                        x1, y1, x2, y2 = bboxes[camera]
                        w = x2 - x1
                        h = y2 - y1
                        # line = f"{frame_id},-1,{x1},{y1},{w},{h},1,-1,-1,-1\n"
                        class_name = obj.get("object type")
                        class_id = CLASS_NAME_TO_ID.get(class_name, -1)  # default to -1 if unknown
                        line = f"{frame_id},-1,{x1},{y1},{w},{h},1,{class_id},-1,-1\n"
                        det_file.write(line)

        print(f"[✓] Created: {det_txt_path}")

def main():
    parameter_sets = []
    create_det_txt_from_groundtruth(data_root, scene, gt_file_name='ground_truth.json')


if __name__ == "__main__":
    main()