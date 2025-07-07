import json
import cv2
import os
from collections import defaultdict

# Load the JSON file
with open("Detection/Warehouse_016/Camera_01.json", "r") as f:
    data = json.load(f)

# Output directory
output_dir = "Detection/detection_from_json"
os.makedirs(output_dir, exist_ok=True)

# Group detections by Frame
detections_by_frame = defaultdict(list)
for det in data.values():
    detections_by_frame[det["Frame"]].append(det)

# Only draw for Frame 1 and Frame 2
for frame_id in [1, 2,3,4,5,6,7,8,9,10]:
    if frame_id not in detections_by_frame:
        continue

    detections = detections_by_frame[frame_id]
    # img_path = detections[0]["ImgPath"]
    img_path = detections[0]["ImgPath"].replace("../", "./")  # Correct the path
    img = cv2.imread(img_path)

    if img is None:
        print(f"Could not read image: {img_path}")
        continue

    for det in detections:
        coord = det["Coordinate"]
        cls_id = det["ClassID"]
        x1, y1, x2, y2 = coord["x1"], coord["y1"], coord["x2"], coord["y2"]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f'Cls {cls_id}', (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    # Save the image to output directory
    filename = f"frame_{frame_id:04d}_bbox.jpg"
    cv2.imwrite(os.path.join(output_dir, filename), img)
    print(f"Saved: {os.path.join(output_dir, filename)}")