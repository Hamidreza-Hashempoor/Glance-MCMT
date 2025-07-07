cp ./poser/load_tracking_result.py ./mmpose/demo/
cp ./poser/top_down_video_demo_with_track_file.py ./mmpose/demo/
cd ./mmpose
conda activate openmmlab

files=("Warehouse_016")

for SCENE in "${files[@]}"
do
    F_SCENE=$(printf "%03d" "$SCENE")
    echo "Processing scene-$SCENE"

    find "../Detection/$SCENE" -maxdepth 1 -type f -name "*.txt" | while read -r file;
    do
        CAMERA=$(basename "$file")
        CAMERA_NAME=${CAMERA%.txt}
        number=$(echo "$CAMERA" | sed 's/Camera\([0-9]\+\).txt/\1/')

        if [ -z "$number" ]; then
            number=""  # Leave blank if it's just "Camera.txt"
        fi

        echo "  Processing $CAMERA (camera number: $number)"
        echo "  Getting ../Detection/${SCENE}/${CAMERA}"
        echo "  Video is ../AIC25_Track1/Val/${SCENE}/videos/${CAMERA_NAME}/${CAMERA_NAME}.mp4"

        python3 demo/top_down_video_demo_with_track_file.py \
            ../Detection/${SCENE}/${CAMERA} \
            ./configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w48_coco_256x192.py \
            https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth \
            --video-path ../AIC25_Track1/Val/${SCENE}/videos/${CAMERA_NAME}/${CAMERA_NAME}.mp4 \
            --out-file ../Pose/${SCENE}/${CAMERA_NAME}/${CAMERA_NAME}_out_keypoint.json

    done
done