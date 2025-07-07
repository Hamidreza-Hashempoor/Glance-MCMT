conda activate botsort_env

files=("Warehouse_000" "Warehouse_011" "Warehouse_014")
for SCENE in "${files[@]}"
do
    echo "Processing $SCENE"
    python3 tools/extract_frames_25.py ./AIC25_Track1/Train -s $SCENE
    python3 tools/extract_gt_25.py ./AIC25_Track1/Train -s $SCENE
done

# Convert all to COCO format
python3 tools/convert_to_coco_25.py --BASE_DIR ./AIC25_Track1 -s Train