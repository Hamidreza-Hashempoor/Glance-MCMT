'''
extract ReID features from testing data.
'''
import os
import argparse
import os.path as osp
import sys
try:
    sys.path.append('deep-person-reid')
except:
    print( "reid already in path")

import numpy as np
import torch
import time
import torchvision.transforms as T
from PIL import Image
import sys
from utils import FeatureExtractor
import torchreid
import json



def make_parser():
    parser = argparse.ArgumentParser("reid")
    parser.add_argument("root_path", type=str, default=None)
    parser.add_argument("-s", "--scene", type=str, default=None)
    parser.add_argument("-c", "--camera", type=str, default=None)
    return parser

if __name__ == "__main__":

    args = make_parser().parse_args()
    data_root = args.root_path
    scene = args.scene
    camera = args.camera

    sys.path.append(data_root+'/deep-person-reid')

    # img_dir = os.path.join(data_root,'Original')
    # img_dir = os.path.join(data_root, "AIC25_Track1/Val", scene, "videos")
    img_dir = os.path.join(data_root, "AIC25_Track1/Test", scene, "videos")
    
    det_dir = os.path.join(data_root,'Detection')
    out_dir = os.path.join(data_root,'EmbedFeature')

    models = {
              'osnet_x1_0':data_root+'/deep-person-reid/checkpoints/osnet_ms_m_c.pth.tar'
             }
    
    
    model_names = ['osnet_x1_0']
    

    val_transforms = T.Compose([
        T.Resize([256, 128]),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    for model_idx,name in enumerate(models):
        
        model_p = models[name]
        model_name = model_names[model_idx]

        print('Using model {}'.format(name))

        extractor = FeatureExtractor(
            model_name=model_name,
            model_path=model_p,
            device='cuda'
        )   


        base, ext = os.path.splitext(camera + ".txt")
        if ext == '.txt':
            print('processing file {}{}'.format(base,ext))
            det_path = os.path.join(det_dir,scene,'{}.txt'.format(base))
            json_path = os.path.join(det_dir,scene,'{}.json'.format(base))
            dets = np.genfromtxt(det_path,dtype=str,delimiter=',')
            with open(json_path) as f:
                jf = json.load(f)
            cur_frame = 0
            u_num = 0
            emb = np.array([None]*len(dets))
            start = time.time()
            print('processing scene {} cam {} with {} detections'.format(scene,base,len(dets)))
            for idx,(cam,frame,_,x1,y1,x2,y2,conf) in enumerate(dets):
                u_num += 1
                x1,y1,x2,y2 = map(float,[x1,y1,x2,y2])
                if idx%1000 == 0:
                    if idx !=0:
                        end = time.time()
                        print('processing time :',end-start)
                    start = time.time()
                    print('process {}/{}'.format(idx,len(dets)))
                if cur_frame != int(frame):
                    cur_frame = int(frame)
                if not os.path.isdir(osp.join(out_dir,scene,cam)):
                    os.makedirs(osp.join(out_dir,scene,cam))
                
                # 🟡 Extract ClassID from loaded JSON
                class_id = jf[str(idx).zfill(8)].get('ClassID', -1)
                fname = f"feature_{cur_frame}_{u_num}_{int(x1)}_{int(x2)}_{int(y1)}_{int(y2)}_{str(conf).replace('.', '')}_cls{class_id}.npy"
                save_fn = os.path.join(out_dir, scene, cam, fname)
                jf[str(idx).zfill(8)]['NpyPath'] = os.path.join(scene, cam, fname)
                # save_fn = os.path.join(out_dir,scene,cam,'feature_{}_{}_{}_{}_{}_{}_{}.npy'.format(cur_frame,u_num,str(int(x1)),str(int(x2)),str(int(y1)),str(int(y2)),str(conf).replace(".","")))
                # jf[str(idx).zfill(8)]['NpyPath'] = os.path.join(scene,cam,'feature_{}_{}_{}_{}_{}_{}_{}.npy'.format(cur_frame,u_num,str(int(x1)),str(int(x2)),str(int(y1)),str(int(y2)),str(conf).replace(".","")))
                img_path = os.path.join(img_dir,cam,'Frame',frame.zfill(6)+'.jpg')
                img = Image.open(img_path)

                ### guard
                img_w, img_h = img.size  # ✅ Get image dimensions

                # Clamp bounding box to image boundaries
                x1 = max(0, min(x1, img_w - 1))
                x2 = max(0, min(x2, img_w - 1))
                y1 = max(0, min(y1, img_h - 1))
                y2 = max(0, min(y2, img_h - 1))

                # Ensure coordinates are in the correct order
                if x2 <= x1:
                    x2 = x1 + 1
                if y2 <= y1:
                    y2 = y1 + 1
                img_crop = img.crop((x1,y1,x2,y2))
                img_crop = val_transforms(img_crop.convert('RGB')).unsqueeze(0)
                feature = extractor(img_crop).cpu().detach().numpy()[0]

                np.save(save_fn,feature)
            end = time.time()
            print('processing time :',end-start)
            start = time.time()
            print('process {}/{}'.format(idx+1,len(dets)))
            with open(json_path, 'w') as f:
                json.dump(jf, f, ensure_ascii=False)
