from pycocotools.coco import COCO
import os

coco = COCO("AIC25_Track1/Train/annotations/train_aicity_25.json")
print("Number of images:", len(coco.imgs))
print("Number of annotations:", len(coco.anns))
print("Number of categories:", len(coco.cats))

# Try to access all image paths
for img_id, img_info in coco.imgs.items():
    path = os.path.join("AIC25_Track1/Train", img_info["file_name"])
    if not os.path.exists(path):
        print(f"Missing file: {path}")