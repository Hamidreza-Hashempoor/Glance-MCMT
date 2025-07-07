from yolox.exp import get_exp
from yolox.utils import fuse_model, postprocess
from yolox.data.data_augment import ValTransform
from yolox.utils.visualize import vis
import torch
import cv2
import os
import random
from pycocotools.coco import COCO
import numpy as np
from tqdm import tqdm

# ---- Load model ----
ckpt_path = "YOLOX_outputs/yolox_x_AI_City_25/best_ckpt.pth.tar"
exp = get_exp("BoT-SORT/yolox/exps/example/mot/yolox_x_AI_City_25.py", None)
model = exp.get_model()
model.eval()

# Load checkpoint
ckpt = torch.load(ckpt_path, map_location="cpu")
model.load_state_dict(ckpt["model"])
model = model.cuda()
model = fuse_model(model)

coco = COCO("AIC25_Track1/Train/annotations/train_aicity_25.json")
img_ids = coco.getImgIds()
random.seed(42)
sampled_ids = random.sample(img_ids, 200)

transform = ValTransform()
out_dir = "AIC25_Track1/check_detection"
os.makedirs(out_dir, exist_ok=True)

for img_id in tqdm(sampled_ids):
    info = coco.loadImgs(img_id)[0]
    img_path = os.path.join("AIC25_Track1/Train", info["file_name"])
    img = cv2.imread(img_path)

    if img is None:
        print(f"Warning: could not read {img_path}")
        continue

    h, w = img.shape[:2]
    img_input, _ = transform(img, np.zeros((0, 5)), exp.test_size)
    img_input = torch.from_numpy(img_input).unsqueeze(0).float().cuda()

    with torch.no_grad():
        outputs = model(img_input)
        outputs = postprocess(outputs, exp.num_classes, exp.test_conf, exp.nmsthre)

    output = outputs[0]
    if output is None:
        continue

    output = output.cpu().numpy()
    bboxes = output[:, 0:4]
    scores = output[:, 4] * output[:, 5]
    cls_ids = output[:, 6]

    # Filter by confidence > 0.5
    conf_thresh = 0.1
    keep = scores > conf_thresh
    bboxes = bboxes[keep]
    scores = scores[keep]
    cls_ids = cls_ids[keep]

    # Rescale bboxes
    scale = min(exp.test_size[0] / h, exp.test_size[1] / w)
    bboxes /= scale

    # Draw and save
    vis_img = vis(img, bboxes, scores, cls_ids, conf=conf_thresh, class_names=exp.class_names)
    cv2.imwrite(os.path.join(out_dir, f"{img_id}.jpg"), vis_img)