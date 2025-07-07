cp ./detector/aic25_get_detection.py ./BoT-SORT/tools/
cd ./BoT-SORT
conda activate botsort_env

files=("Warehouse_016")
for SCENE in "${files[@]}"
do
    F_SCENE=$(printf "%03d" "$SCENE")
    echo Procssing scene-$F_SCENE
    python3 tools/aic25_get_detection.py -s $SCENE ../
done