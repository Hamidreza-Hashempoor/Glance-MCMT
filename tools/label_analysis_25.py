import json
from collections import defaultdict

# === Paths ===
COCO_JSON = "coco_annotations/train.json"

# === Define your label mapping (same as used in training) ===
CLASS_NAME_TO_ID = {
    "Person": 1,
    "Forklift": 2,
    "NovaCarter": 3,
    "Transporter": 4,
    "FourierGR1T2": 5,
    "AgilityDigit": 6,
}

ID_TO_CLASS_NAME = {v: k for k, v in CLASS_NAME_TO_ID.items()}

# === Load COCO annotations ===
with open(COCO_JSON, "r") as f:
    coco = json.load(f)

# === Count annotations per category ===
label_counts = defaultdict(int)
for ann in coco["annotations"]:
    cat_id = ann["category_id"]
    label_counts[cat_id] += 1

# === Print results ===
print("Annotation counts per label:")
for cat_id in sorted(ID_TO_CLASS_NAME.keys()):
    name = ID_TO_CLASS_NAME[cat_id]
    count = label_counts.get(cat_id, 0)
    print(f"{name:15}: {count}")



# Output
# Annotation counts per label:
# Person         : 7904240
# Forklift       : 179980
# NovaCarter     : 385026
# Transporter    : 360237
# FourierGR1T2   : 33584
# AgilityDigit   : 62953