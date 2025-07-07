import os
import json
import cv2

# === Configuration ===
COCO_JSON = "coco_annotations/train.json"  # COCO format JSON
IMAGE_ROOT = "AIC25_Track1/Train"  # root folder where actual images are located
SAVE_DIR = "AIC25_Track1/train_annotation_check"      # output folder

# === Setup ===
os.makedirs(SAVE_DIR, exist_ok=True)

with open(COCO_JSON, "r") as f:
    coco = json.load(f)

# Build image_id → image info dict
image_id_map = {img["id"]: img for img in coco["images"][:100]}

# Build image_id → list of annotations
from collections import defaultdict
ann_map = defaultdict(list)
for ann in coco["annotations"]:
    img_id = ann["image_id"]
    if img_id in image_id_map:
        ann_map[img_id].append(ann)

# Build category_id → name
category_map = {cat["id"]: cat["name"] for cat in coco["categories"]}

# === Process and draw boxes ===
for img_id, img_info in image_id_map.items():
    img_path = os.path.join(IMAGE_ROOT, img_info["file_name"])
    img = cv2.imread(img_path)
    if img is None:
        print(f"Warning: Could not read {img_path}")
        continue

    anns = ann_map[img_id]
    for ann in anns:
        x, y, w, h = map(int, ann["bbox"])
        category_id = ann["category_id"]
        label = category_map.get(category_id, "Unknown")
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 1, cv2.LINE_AA)

    out_path = os.path.join(SAVE_DIR, os.path.basename(img_info["file_name"]))
    cv2.imwrite(out_path, img)
    print(f"Saved {out_path}")