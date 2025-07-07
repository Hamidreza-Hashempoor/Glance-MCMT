conda activate botsort_env

files=("Warehouse_016")
for SCENE in "${files[@]}"
do
    echo "Processing $SCENE"
    python3 tools/extract_frames_25.py ./AIC25_Track1/Val -s $SCENE
    # python3 tools/extract_gt_25.py ./AIC25_Track1/Train -s $SCENE
done


