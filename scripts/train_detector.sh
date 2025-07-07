conda activate botsort_env

## train from scratch
python BoT-SORT/yolox/train.py -f BoT-SORT/yolox/exps/example/mot/yolox_x_AI_City_25.py -d 1 -b 4 --fp16 -c BoT-SORT/pretrained/bytetrack_x_mot17.pth.tar

## resume already started training
python BoT-SORT/yolox/train.py -f BoT-SORT/yolox/exps/example/mot/yolox_x_AI_City_25.py -d 1 -b 4 --fp16 -c YOLOX_outputs/yolox_x_AI_City_25/latest_ckpt.pth.tar --resume
