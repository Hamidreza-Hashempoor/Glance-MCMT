import sys
import argparse
import os
import os.path as osp
import time
import cv2
import torch
import numpy as np
import json

from loguru import logger

sys.path.append('.')

try:
    sys.path.append('BoT-SORT')
except:
    print( "bot sort already in path")

from yolox.data.data_augment import preproc
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess
from yolox.utils.visualize import plot_tracking
from tracker.bot_sort import BoTSORT
from tracker.tracking_utils.timer import Timer


IMAGE_EXT = [".jpg"]
def make_parser():
    parser = argparse.ArgumentParser("BoT-SORT Demo!")
    parser.add_argument("root_path", type=str, default=None)
    parser.add_argument("-s","--scene", default=None, type=str)
    parser.add_argument("demo", default="video", help="demo type, eg. image, video and webcam")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")
    parser.add_argument("--path", default="AIC25_Track1/Val/Warehouse_016/videos/Camera/Camera.mp4", help="path to images or video")
    parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")
    parser.add_argument("--save_result", action="store_true",help="whether to save the inference result of image/video")
    parser.add_argument("-f", "--exp_file", default="BoT-SORT/yolox/exps/example/mot/yolox_x_AI_City_25.py", type=str, help="pls input your expriment description file")
    parser.add_argument("-c", "--ckpt", default="BoT-SORT/ai_city_ckpt.pth.tar", type=str, help="ckpt for eval")
    parser.add_argument("--device", default="gpu", type=str, help="device to run our model, can either be cpu or gpu")
    parser.add_argument("--conf", default=None, type=float, help="test conf")
    parser.add_argument("--nms", default=None, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument("--fps", default=30, type=int, help="frame rate (fps)")
    parser.add_argument("--fp16", dest="fp16", default=False, action="store_true",help="Adopting mix precision evaluating.")
    parser.add_argument("--fuse", dest="fuse", default=False, action="store_true", help="Fuse conv and bn for testing.")
    parser.add_argument("--trt", dest="trt", default=False, action="store_true", help="Using TensorRT model for testing.")

    # tracking args
    parser.add_argument("--track_high_thresh", type=float, default=0.6, help="tracking confidence threshold")
    parser.add_argument("--track_low_thresh", default=0.1, type=float, help="lowest detection threshold")
    parser.add_argument("--new_track_thresh", default=0.7, type=float, help="new track thresh")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")
    parser.add_argument("--aspect_ratio_thresh", type=float, default=1.6, help="threshold for filtering out boxes of which aspect ratio are above the given value.")
    parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')
    parser.add_argument("--fuse-score", dest="fuse_score", default=False, action="store_true", help="fuse score and iou for association")

    # CMC
    parser.add_argument("--cmc-method", default="orb", type=str, help="cmc method: files (Vidstab GMC) | orb | ecc")

    # ReID
    parser.add_argument("--with-reid", dest="with_reid", default=False, action="store_true", help="test mot20.")
    parser.add_argument("--fast-reid-config", dest="fast_reid_config", default=r"fast_reid/configs/MOT17/sbs_S50.yml", type=str, help="reid config file path")
    parser.add_argument("--fast-reid-weights", dest="fast_reid_weights", default=r"pretrained/mot17_sbs_S50.pth", type=str,help="reid config file path")
    parser.add_argument('--proximity_thresh', type=float, default=0.5, help='threshold for rejecting low overlap reid matches')
    parser.add_argument('--appearance_thresh', type=float, default=0.25, help='threshold for rejecting low appearance similarity reid matches')
    return parser


def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = osp.join(maindir, filename)
            ext = osp.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names


def write_results(filename, results):
    save_format = '{frame},{id},{x1},{y1},{w},{h},{s},-1,-1,{cls_id}\n'
    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids, scores, cls_ids in results:
            for tlwh, track_id, score, cls_id in zip(tlwhs, track_ids, scores, cls_ids):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                line = save_format.format(
                    frame=frame_id,
                    id=track_id,
                    x1=round(x1, 1),
                    y1=round(y1, 1),
                    w=round(w, 1),
                    h=round(h, 1),
                    s=round(score, 2),
                    cls_id=int(cls_id)
                )
                f.write(line)
    logger.info('save results to {}'.format(filename))


class Predictor(object):
    def __init__(
        self,
        model,
        exp,
        trt_file=None,
        decoder=None,
        device=torch.device("cpu"),
        fp16=False
    ):
        self.model = model
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones((1, 3, exp.test_size[0], exp.test_size[1]), device=device)
            self.model(x)
            self.model = model_trt
        self.rgb_means = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

    def inference(self, img, timer):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = osp.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        img, ratio = preproc(img, self.test_size, self.rgb_means, self.std)
        img_info["ratio"] = ratio
        img = torch.from_numpy(img).unsqueeze(0).float().to(self.device)
        if self.fp16:
            img = img.half()  # to FP16

        with torch.no_grad():
            timer.tic()
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(outputs, self.num_classes, self.confthre, self.nmsthre)
        return outputs, img_info


def image_demo(predictor, vis_folder, current_time, args):

    root_path = args.root_path
    scene = args.scene
    # input = osp.join(root_path, "Original", scene)
    input = osp.join(root_path, "AIC25_Track1/Val", scene, "videos")
    cameras = []
    for f in os.listdir(input):
        if os.path.isdir(os.path.join(input, f)):
            cameras.append(f)
    cameras = sorted(cameras)
    scale = min(800/1080,1440/1920)
    for cam in cameras:
        imgs = sorted(os.listdir(osp.join(input, cam, 'Frame')))
        timer = Timer()
        output = osp.join(root_path,'Detection', '{}.txt'.format(osp.join(scene, cam)))
        outjson = osp.join(root_path,'Detection', '{}.json'.format(osp.join(scene, cam)))
        if not os.path.isdir(osp.join(root_path,'Detection',scene)):
            os.makedirs(osp.join(root_path,'Detection',scene))
        u_num = 0
        ret_json = {}
        results = []
        for frame_id, img_path in enumerate(imgs, 1):
            img_path = osp.join(input, cam, 'Frame',img_path)

            # Detect objects
            outputs, img_info = predictor.inference(img_path, timer)
            
            detections = []
            if outputs[0] is not None:
                outputs = outputs[0].cpu().numpy()
                detections = outputs[:, :7]
                detections[:, :4] /= scale
                detections = detections[detections[:,4]>0.1]
                timer.toc()
            else:
                timer.toc()

            for det in detections:
                x1, y1, x2, y2, score, _, cls_id = det
                x1 = max(0,x1)
                y1 = max(0,y1)
                x2 = min(1920,x2)
                y2 = min(1080,y2)
                results.append([cam, frame_id, int(cls_id), int(x1), int(y1), int(x2), int(y2), float(score)])
                det_json = {}
                det_json['Frame'] = frame_id
                det_json['ImgPath'] = img_path.replace(root_path + '/','')
                det_json['NpyPath'] = ''
                Coordinate = {'x1':int(x1), 'y1':int(y1), 'x2': int(x2), 'y2': int(y2)}
                det_json['Coordinate'] = Coordinate
                det_json['ClusterID'] = None
                det_json['OfflineID'] = None
                det_json['ClassID'] =  int(cls_id)  # NEW FIELD
                ret_json[str(u_num).zfill(8)] = det_json
                u_num += 1

            if frame_id % 1000 == 0:
                logger.info('Processing cam {} frame {} ({:.2f} fps)'.format(cam, frame_id, 1. / max(1e-5, timer.average_time)))

        with open(output,'a') as f:
            for cam,frame_id,cls,x1,y1,x2,y2,score in results:
                f.write('{},{},{},{},{},{},{},{}\n'.format(cam,frame_id,cls,x1,y1,x2,y2,score))
        with open(outjson, 'a') as f:
            json.dump(ret_json, f, ensure_ascii=False)

def imageflow_demo(predictor, vis_folder, current_time, args):
    cap = cv2.VideoCapture(args.path if args.demo == "video" else args.camid)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)
    timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
    save_folder = osp.join(vis_folder, timestamp)
    os.makedirs(save_folder, exist_ok=True)
    if args.demo == "video":
        save_path = osp.join(save_folder, args.path.split("/")[-1])
    else:
        save_path = osp.join(save_folder, "camera.mp4")
    logger.info(f"video save_path is {save_path}")
    vid_writer = cv2.VideoWriter(
        save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
    )
    tracker = BoTSORT(args, frame_rate=args.fps)
    timer = Timer()
    frame_id = 0
    results = []
    while True:
        if frame_id % 20 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))
        ret_val, frame = cap.read()
        if ret_val:
            # Detect objects
            outputs, img_info = predictor.inference(frame, timer)
            scale = min(exp.test_size[0] / float(img_info['height'], ), exp.test_size[1] / float(img_info['width']))

            if outputs[0] is not None:
                outputs = outputs[0].cpu().numpy()
                #
                # detections = outputs[:, :7]
                # detections[:, :4] /= scale

                # Extract bboxes, score, and class_id
                bboxes = outputs[:, :4] / scale
                obj_conf = outputs[:, 4]
                cls_conf = outputs[:, 5]
                cls_ids = outputs[:, 6].astype(int)
                detections = np.concatenate((bboxes, obj_conf[:, None], cls_conf[:, None], cls_ids[:, None]), axis=1)

                # Run tracker
                online_targets = tracker.update(detections, img_info["raw_img"])

                online_tlwhs = []
                online_ids = []
                online_scores = []
                online_cls_ids = []  # New list
                for t in online_targets:
                    tlwh = t.tlwh
                    tid = t.track_id
                    cls_id = t.cls_id  # <— you must store this in STrack
                    vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
                    if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                        online_tlwhs.append(tlwh)
                        online_ids.append(tid)
                        online_scores.append(t.score)
                        online_cls_ids.append(cls_id)
                        results.append(
                            f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,{cls_id}\n"
                        )
                timer.toc()
                online_im = plot_tracking(
                    img_info['raw_img'], online_tlwhs, online_ids, frame_id=frame_id + 1, fps=1. / timer.average_time
                )
            else:
                timer.toc()
                online_im = img_info['raw_img']
            if args.save_result:
                vid_writer.write(online_im)

            # Not suppurted in the server side (NO GUI)
            # cv2.imshow("Online Tracking", online_im) 
            # ch = cv2.waitKey(1)
            # if ch == 27 or ch == ord("q") or ch == ord("Q"):
            #     break
        else:
            break
        frame_id += 1

    if args.save_result:
        res_file = osp.join(vis_folder, f"{timestamp}.txt")
        with open(res_file, 'w') as f:
            f.writelines(results)
        logger.info(f"save results to {res_file}")


def main(exp, args):
    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    output_dir = osp.join(exp.output_dir, args.experiment_name)
    os.makedirs(output_dir, exist_ok=True)

    if args.save_result:
        vis_folder = osp.join(output_dir, "track_vis")
        os.makedirs(vis_folder, exist_ok=True)

    if args.trt:
        args.device = "gpu"
    args.device = torch.device("cuda" if args.device == "gpu" else "cpu")

    logger.info("Args: {}".format(args))

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    model = exp.get_model().to(args.device)
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))
    model.eval()

    if not args.trt:
        if args.ckpt is None:
            ckpt_file = osp.join(output_dir, "best_ckpt.pth.tar")
        else:
            ckpt_file = args.ckpt
        logger.info("loading checkpoint")
        ckpt = torch.load(ckpt_file, map_location="cpu")
        # load the model state dict
        model.load_state_dict(ckpt["model"])
        logger.info("loaded checkpoint done.")

    if args.fuse:
        logger.info("\tFusing model...")
        model = fuse_model(model)

    if args.fp16:
        model = model.half()  # to FP16

    if args.trt:
        assert not args.fuse, "TensorRT model is not support model fusing!"
        trt_file = osp.join(output_dir, "model_trt.pth")
        assert osp.exists(
            trt_file
        ), "TensorRT model is not found!\n Run python3 tools/trt.py first!"
        model.head.decode_in_inference = False
        decoder = model.head.decode_outputs
        logger.info("Using TensorRT to inference")
    else:
        trt_file = None
        decoder = None

    predictor = Predictor(model, exp, trt_file, decoder, args.device, args.fp16)
    current_time = time.localtime()
    
    # image_demo(predictor, None, current_time, args)


    imageflow_demo(predictor, vis_folder, current_time, args)



if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)

    args.ablation = False
    args.mot20 = not args.fuse_score

    main(exp, args)
