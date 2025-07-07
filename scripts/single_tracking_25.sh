

conda activate botsort_env

files=("Warehouse_016")
cameras=("Camera" "Camera_01" "Camera_02" "Camera_03" "Camera_04" "Camera_05" "Camera_06" "Camera_07" "Camera_08" "Camera_09" "Camera_10" "Camera_11")
for SCENE in "${cameras[@]}"
do
    echo Procssing scene-$SCENE
    python BoT-SORT/single_camera_tracking.py -s Warehouse_016 -c $SCENE 
done

#files=("Warehouse_017")
# cameras=("Camera" "Camera_01" "Camera_02" "Camera_03" "Camera_04" "Camera_05" "Camera_06" "Camera_07")
# for SCENE in "${cameras[@]}"
# do
#     echo Procssing scene-$SCENE
#     python BoT-SORT/single_camera_tracking.py -s Warehouse_017 -c $SCENE 
# done

# #files=("Warehouse_018")
# cameras=("Camera" "Camera_01" "Camera_02" "Camera_03" "Camera_04" "Camera_05" "Camera_06" "Camera_07" "Camera_08")
# for SCENE in "${cameras[@]}"
# do
#     echo Procssing scene-$SCENE
#     python BoT-SORT/single_camera_tracking.py -s Warehouse_018 -c $SCENE 
# done

# #files=("Warehouse_019")
# cameras=("Camera" "Camera_01" "Camera_02" "Camera_03" "Camera_04" "Camera_05" "Camera_06" "Camera_07")
# for SCENE in "${cameras[@]}"
# do
#     echo Procssing scene-$SCENE
#     python BoT-SORT/single_camera_tracking.py -s Warehouse_019 -c $SCENE 
# done

# #files=("Warehouse_020")
# cameras=("Camera" "Camera_01" "Camera_02" "Camera_03" "Camera_04" "Camera_05" "Camera_06" "Camera_07" "Camera_08")
# for SCENE in "${cameras[@]}"
# do
#     echo Procssing scene-$SCENE
#     python BoT-SORT/single_camera_tracking.py -s Warehouse_020 -c $SCENE 
# done
