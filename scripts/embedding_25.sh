cp ./embedder/aic25_extract.py ./deep-person-reid/torchreid/
cd ./deep-person-reid
conda activate torchreid

files=("Warehouse_016")
for SCENE in "${files[@]}"
do
    F_SCENE=$(printf "%03d" "$SCENE")
    echo Procssing scene-$F_SCENE
    python3 torchreid/aic25_extract.py -s $SCENE ../
done