import os
import os.path as osp
import json
import glob
from tqdm import tqdm
import cv2
import argparse



# Define class mapping if needed
CLASS_NAME_TO_ID = {
    "Person": 1,
    "Forklift": 2,
    "NovaCarter": 3,
    "Transporter": 4,
    "FourierGR1T2": 5,
    "AgilityDigit": 6,
    # Add more if needed
}

NUM_CLASSES = len(CLASS_NAME_TO_ID)

def convert_det_to_coco(split_dir, output_json):
    image_id = 1
    ann_id = 1
    video_id = 1

    coco_output = {
        "images": [],
        "annotations": [],
        "categories": [{"id": cid, "name": name} for name, cid in CLASS_NAME_TO_ID.items()],
        "videos": []
    }

    if os.path.exists(output_json):
        print(f"[✗] File already exists: {output_json}")
        exit(0)

    warehouses = os.listdir(split_dir)
    for warehouse in tqdm(sorted(warehouses), desc="Warehouses"):
        video_root = osp.join(split_dir, warehouse, "videos")
        if not osp.isdir(video_root):
            continue

        for camera in tqdm(sorted(os.listdir(video_root)), desc=f"{warehouse} Cameras", leave=False):
            if not camera.startswith("Camera_"):
                continue

            camera_dir = osp.join(video_root, camera)
            frame_dir = osp.join(camera_dir, "Frame")
            det_path = osp.join(camera_dir, "det", "det.txt")
            if not osp.exists(det_path):
                continue

            coco_output["videos"].append({
                "id": video_id,
                "file_name": f"{warehouse}/{camera}"
            })

            # Build image_id map
            frame_files = sorted(glob.glob(osp.join(frame_dir, "*.jpg")))
            frame_id_to_image_id = {}

            for idx, frame_file in enumerate(tqdm(frame_files, desc=f"{camera} Frames", leave=False)):
                filename = osp.relpath(frame_file, split_dir)
                img = cv2.imread(frame_file)
                h, w = img.shape[:2]
                image_info = {
                    "id": image_id,
                    "file_name": filename.replace("\\", "/"),
                    "frame_id": idx + 1,
                    "video_id": video_id,
                    "height": h,
                    "width": w
                }
                frame_number = idx + 1
                frame_id_to_image_id[frame_number] = image_id
                coco_output["images"].append(image_info)
                image_id += 1

            # Parse det.txt
            with open(det_path, 'r') as f:
                for line in tqdm(f, desc=f"{camera} Detections", leave=False):
                    parts = line.strip().split(',')
                    if len(parts) < 7:
                        continue
                    frame, _, x, y, w, h, conf = parts[:7]
                    class_id = int(parts[7]) if len(parts) > 7 else 1  # Default to 1 if no class

                    frame = int(frame)
                    x, y, w, h = map(float, [x, y, w, h])
                    conf = float(conf)

                    image_id_ref = frame_id_to_image_id.get(frame)
                    if not image_id_ref:
                        continue

                    annotation = {
                        "id": ann_id,
                        "image_id": image_id_ref,
                        "category_id": class_id,
                        "bbox": [x, y, w, h],
                        "area": w * h,
                        "iscrowd": 0,
                        "track_id": -1,
                        "conf": conf
                    }
                    coco_output["annotations"].append(annotation)
                    ann_id += 1

            video_id += 1

    os.makedirs(osp.dirname(output_json), exist_ok=True)
    with open(output_json, 'w') as f:
        json.dump(coco_output, f)
    print(f"[✓] COCO-style {SPLIT} saved to: {output_json}")




def make_parser():
    parser = argparse.ArgumentParser("make coco")
    parser.add_argument("--BASE_DIR", type=str, default='./AIC25_Track1')
    parser.add_argument("-s", "--SPLIT", type=str, default='Train')
    return parser

args = make_parser().parse_args()
BASE_DIR = args.BASE_DIR
SPLIT = args.SPLIT
OUTPUT_PATH = f'./AIC25_Track1/{SPLIT}/annotations/{SPLIT.lower()}.json'

if __name__ == '__main__':
    convert_det_to_coco(osp.join(BASE_DIR, SPLIT), OUTPUT_PATH)