import os
import numpy as np
import json
import glob
import h5py


class DetectedObjects:
    """
    Represents whole detected objects to track.
    Object dict is built by frame_id as a key and its entity contains a list of all Detected objects of the frame. 
    """
    def __init__(self):
        self.num_objects = 0
        self.objects = {}
        self._objects_registered = {}
        #self.scene_id = scene_id
        #self.camera_id = -1
        self.camera_projection_matrix = None
        self.homography_matrix = None

    def __str__(self):
        return f"DetectedObjects: scene_id:{self.scene_id}, camera_id:{self.camera_id}, num_objects:{self.num_objects}"

    def verify_estimated_3d_with_ground_truth(self, gt_path, camera_id, frame_id, estimated_3d, estimated_bbox):
        """
        Compare estimated 3D world coordinates with GT 3D based on closest 2D bbox in the same frame.
        
        Args:
            gt_path (str): Path to ground_truth.json
            camera_id (str): e.g., "Camera_0005"
            frame_id (int or str): e.g., 25
            estimated_3d (tuple): (x, y, z) in meters
            estimated_bbox (tuple): (x1, y1, x2, y2) in image pixels
        """
        if isinstance(frame_id, int):
            frame_id = str(frame_id)

        with open(gt_path, "r") as f:
            gt_data = json.load(f)

        if frame_id not in gt_data:
            print(f"[WARN] Frame {frame_id} not found in GT data.")
            return

        objects = gt_data[frame_id]
        closest_obj = None
        min_dist = float('inf')
        est_cx = (estimated_bbox[0] + estimated_bbox[2]) / 2
        est_cy = (estimated_bbox[1] + estimated_bbox[3]) / 2

        for obj in objects:
            bbox_dict = obj.get("2d bounding box visible", {})
            if camera_id in bbox_dict:
                gt_bbox = bbox_dict[camera_id]
                gt_cx = (gt_bbox[0] + gt_bbox[2]) / 2
                gt_cy = (gt_bbox[1] + gt_bbox[3]) / 2
                dist = np.sqrt((est_cx - gt_cx) ** 2 + (est_cy - gt_cy) ** 2)
                if dist < min_dist:
                    min_dist = dist
                    closest_obj = obj

        if closest_obj is None:
            print(f"[WARN] No GT bbox found for {camera_id} in frame {frame_id}.")
            return

        gt_3d = closest_obj.get("3d location")
        if gt_3d:
            gt_3d_np = np.array(gt_3d)
            est_3d_np = np.array(estimated_3d)
            diff = np.linalg.norm(gt_3d_np - est_3d_np)
            print(f"✅ Verification Result for frame {frame_id}, camera {camera_id}")
            print(f"  - Closest GT bbox center distance: {min_dist:.2f} pixels")
            print(f"  - Estimated 3D: {est_3d_np}")
            print(f"  - GT 3D       : {gt_3d_np}")
            print(f"  - 3D Euclidean error: {diff:.3f} m\n")
        else:
            print(f"[WARN] No GT 3D location found in matched object.")

    def load_from_directory(self, feature_root, calibration_path="Calibration"):
        if not os.path.isdir(feature_root):
            raise Exception(f'There is no directory to read from. {feature_root}')
        npys = sorted(glob.glob(os.path.join(feature_root, "**/*.npy"), recursive=True))
        scene_id = None
        camera_id = None
        path_list = feature_root.split("/")
        for dir in path_list:  
            if dir.startswith("Warehouse_"):  
                # scene_id = int(dir.replace("scene_",""))
                scene_id = dir
            if dir.startswith("Camera"):  
                # camera_id = int(dir.replace("camera_",""))
                camera_id = dir
        if scene_id is not None and camera_id is not None:
            calibration_path = f"AIC25_Track1/Val/{scene_id}/calibration.json"
            self.load_calibration(calibration_path, camera_id)

            depth_map_path = os.path.join("AIC25_Track1", "Val", scene_id, "depth_map", f"{camera_id}.h5")
            self.load_depth_map(depth_map_path)  
        else:
            print(f'\033[33mwarning\033[0m : failed to get scene_id and camera_id from feature path.')
            print(f'\033[33mwarning\033[0m : world coordinate calculations are ignored.')


        # for f in npys:
        #     self.add_object_from_image_path(f)

        #### Debug 1
        DEBUG_MAX_FRAMES = 1000
        frame_ids_seen = set()
        print("in debug mode only 1000 frames.")
        print(f"in debug mode: only load objects from {DEBUG_MAX_FRAMES} unique frames.")

        for f in npys:
            fname = os.path.basename(f)
            if fname.startswith("feature_"):
                _, frame_id, *_ = os.path.splitext(fname)[0].split("_")
            else:
                _, frame_id, *_ = os.path.splitext(fname)[0].split("_")

            frame_id_int = int(frame_id)
            if frame_id_int not in frame_ids_seen:
                if len(frame_ids_seen) >= DEBUG_MAX_FRAMES:
                    break
                frame_ids_seen.add(frame_id_int)

            image_pth = f"AIC25_Track1/Val/{scene_id}/videos/{camera_id}/Frame/{frame_id_int:06d}.jpg"
            self.add_object_from_image_path(f, image_pth)

        #### Debug 2
        # DEBUG_MAX_OBJECTS = 2000
        # object_count = 0
        # print("in debug mode only 2000 objects.")
        # for f in npys:
        #     if object_count >= DEBUG_MAX_OBJECTS:
        #         break

        #     fname = os.path.basename(f)
        #     if fname.startswith("feature_"):
        #         _, frame_id, *_ = os.path.splitext(fname)[0].split("_")
        #     else:
        #         _, frame_id, *_ = os.path.splitext(fname)[0].split("_")

        #     image_pth = f"AIC25_Track1/Val/{scene_id}/videos/{camera_id}/Frame/{int(frame_id):06d}.jpg"
        #     # Track number of objects before and after
        #     num_before = self.num_objects
        #     self.add_object_from_image_path(f, image_pth)
        #     num_after = self.num_objects
        #     object_count += (num_after - num_before)

        # Unload depth map to free memory
        if hasattr(self, "depth_h5") and self.depth_h5 is not None:
            self.depth_h5.close()  # Optional: explicitly close HDF5 file
            self.depth_h5 = None
            print(f"✔ Depth map {camera_id} unloaded from memory.")

    def add_object(self, frame_id, coordinate, world_coordinate, confidence, feature_path, image_path=None, cls=None):
        if isinstance(frame_id, str):
            frame_id = int(frame_id)

        # Check if coordinate is reasonable
        if coordinate.x1 >= coordinate.x2 or coordinate.y1 >= coordinate.y2:
            print(f"Unnatural coordinate found in frame {frame_id}: {coordinate}")
            return

        detected_obj = DetectedObject(object_id=self.num_objects, frame_id=frame_id, coordinate=coordinate, worldcoordinate=world_coordinate,
                                      confidence=confidence, feature_path=feature_path, image_path= image_path, cls=cls)
        key = f"{coordinate.x1}_{coordinate.y1}_{coordinate.x2}_{coordinate.y2}"
        if frame_id in self.objects:
            if not key in self._objects_registered[frame_id]:
                objects_per_frame = self.objects[frame_id].append(detected_obj)
                self._objects_registered[frame_id].append(key)
            else:
                print(f"Duplicate coord found in frame {frame_id}: {coordinate}")
                return
        else:
            objects_per_frame = self.objects[frame_id] = [detected_obj]
            self._objects_registered[frame_id] = [key]
        self.num_objects += 1

    def add_object_from_image_path(self, feature_path, image_path=None, calibration_path="Calibration"):
        file_path = os.path.basename(feature_path)
        if file_path.startswith("feature_"):
            _, frame_id, serial_no, x1, x2, y1, y2, conf, cls = os.path.splitext(file_path)[0].split("_")
            cls = int(cls[3:])
            conf = conf if len(conf) == 1 else conf[0]+"."+conf[1:]
        else:
            serial_no, frame_id, x1, x2, y1, y2 = os.path.splitext(file_path)[0].split("_")
            x1, x2, y1, y2 = int(x1.replace("x","")), int(x2), int(y1.replace("y","")), int(y2)
            conf = 0.98765 # Dummy
            cls = 0

        
        cx, cy = int((float(x1) + float(x2)) / 2), int(float(y2))  # bbox center bottom
        World_coordinate = None
        # Estimate 3D world coordinate
        frame_key = f"distance_to_image_plane_{int(frame_id):05d}.png"
        if self.depth_h5 and frame_key in self.depth_h5 and self.intrinsic_matrix is not None and self.cam_to_world_matrix is not None:
            depth_frame = self.depth_h5[frame_key][:]
            world_xyz = self.convert_coordinates_to_3d_world(cx, cy, depth_frame)


            if world_xyz is not None:
                World_coordinate = WorldCoordinate3D(*world_xyz)
                # Optional verification

                # gt_path = "AIC25_Track1/Val/Warehouse_016/ground_truth.json"
                # bbox = (int(x1), int(y1), int(x2), int(y2))
                # camera_id = "Camera"
                # self.verify_estimated_3d_with_ground_truth(gt_path, camera_id, frame_id, world_xyz, bbox)

        # Fallback: use 2D homography if depth or calibration missing
        elif self.homography_matrix is not None:
            w_x, w_y = self.convert_coordinates_2world(cx, cy)
            World_coordinate = WorldCoordinate(w_x, w_y)

        self.add_object(frame_id=int(frame_id), coordinate=Coordinate(x1, y1, x2, y2), world_coordinate=World_coordinate,
                        confidence=float(conf), feature_path=feature_path, image_path=image_path, cls=cls)

    def get_objects_of_frames(self, start_frame, end_frame):
        if start_frame > self.num_frames() or end_frame > self.num_frames():
            return None
        object_dict = {}
        for frame_id in range(start_frame, end_frame):
            if frame_id in self.objects:
                object_dict[frame_id] = self[frame_id]
            #else:
            #    print(f"There is no such frame in the DetectedObjects, will be ignored. frame_id: {frame_id}")
        return object_dict

    def get_object_ids_of_frames(self, start_frame, end_frame):
        """
        Returns a list of detected object IDs that appeared within the specified frame window.
        """
        if start_frame > self.num_frames() or end_frame > self.num_frames():
            return None
        object_ids = []
        for frame_id in range(start_frame, end_frame):
            if frame_id in self.objects:
                for det in self[frame_id]:
                    object_ids.append(det.object_id)
        return sorted(object_ids)

    def __getitem__(self, frame_id):
        if frame_id in self.objects:
            return self.objects[frame_id]
        else:
            return None

    def num_frames(self):
        """
        Returns number of frames that currently holding.
        """
        return len(self.objects)

    def last_frame_id(self):
        """
        Returns the last frame id.
        """
        return max(self.objects.keys())

    def to_trackingdict(self):
        """
        Compatibility function to convert detections in TrackingDict format.
        """
        track_dict = {}
        for frame_id in self.objects:
            for detected_object in self.objects[frame_id]:
                serial_no = detected_object.object_id
                coordinate = json.loads(detected_object.coordinate.__str__())
                if detected_object.worldcoordinate.__str__() != "None":
                    world_coordinate = json.loads(detected_object.worldcoordinate.__str__())
                else:
                    world_coordinate = None
                new_object = { "Frame": frame_id, "NpyPath": detected_object.feature_path,
                               "Coordinate": coordinate, "WorldCoordinate": world_coordinate,  "OfflineID": -1, #"ClusterID": None,
                                "Confidence": detected_object.confidence,
                                "Class": getattr(detected_object, "cls", None),
                                "ImagePath": detected_object.image_path, } 
                track_dict[serial_no] = new_object
        return track_dict

    # def load_calibration(self, calib_path):
    #     if os.path.isfile(calib_path):
    #         with open(calib_path, 'r') as file:
    #             data = json.load(file)
    #             self.camera_projection_matrix = np.array(data["camera projection matrix"])
    #             self.homography_matrix =  np.array(data["homography matrix"])
    #     else:
    #         print(f'\033[33mwarning\033[0m : not found Calibration File.')
    #         print(f'\033[33mwarning\033[0m : world coordinate calculations are ignored.')

    def load_depth_map(self, depth_map_path):
        
        if os.path.isfile(depth_map_path):
            self.depth_h5 = h5py.File(depth_map_path, "r")
            print(f"✔ Loaded depth map: {depth_map_path}")
        else:
            self.depth_h5 = None
            print(f'\033[33mwarning\033[0m : Depth map file not found: {depth_map_path}')
            print(f'\033[33mwarning\033[0m : 3D coordinate estimation will be skipped.')

    def load_calibration(self, calib_path, camera_id):
        if not os.path.isfile(calib_path):
            print(f'\033[33mwarning\033[0m : Calibration file not found at: {calib_path}')
            print(f'\033[33mwarning\033[0m : World coordinate calculations are ignored.')
            return

        with open(calib_path, 'r') as file:
            data = json.load(file)

        for sensor in data.get("sensors", []):
            if sensor.get("type") == "camera" and sensor.get("id") == camera_id:
                try:
                    # Load camera projection matrix (3x4)
                    self.camera_projection_matrix = np.array(sensor["cameraMatrix"], dtype=np.float32)

                    # Load homography matrix (3x3)
                    self.homography_matrix = np.array(sensor["homography"], dtype=np.float32)

                    # Load intrinsic matrix (3x3)
                    self.intrinsic_matrix = np.array(sensor["intrinsicMatrix"], dtype=np.float32)

                    # Load extrinsic matrix (3x4) and convert to 4x4
                    extrinsic_3x4 = np.array(sensor["extrinsicMatrix"], dtype=np.float32)
                    extrinsic_4x4 = np.eye(4, dtype=np.float32)
                    extrinsic_4x4[:3, :] = extrinsic_3x4
                    self.extrinsic_matrix = extrinsic_4x4

                    # Compute camera-to-world transformation
                    self.cam_to_world_matrix = np.linalg.inv(self.extrinsic_matrix)

                    # (Optional) Store origin and direction info
                    self.camera_coordinates = sensor.get("coordinates", {})
                    self.scale_factor = sensor.get("scaleFactor", None)
                    self.translation_to_global = sensor.get("translationToGlobalCoordinates", {})

                    print(f"✔ Loaded calibration for camera: {camera_id}")
                    return
                except Exception as e:
                    print(f'\033[31merror\033[0m : Failed to parse calibration for {camera_id}: {e}')
                    return

        print(f'\033[33mwarning\033[0m : Camera ID "{camera_id}" not found in calibration file.')

    def convert_coordinates_2world(self, x, y):
        vector_xyz = np.array([x, y, 1]) # z=1
        vector_xyz_3d = np.dot(np.linalg.inv(self.homography_matrix), vector_xyz.T)
        return vector_xyz_3d[0] / vector_xyz_3d[2], vector_xyz_3d[1] / vector_xyz_3d[2]
    
def convert_coordinates_to_3d_world(cx, cy, depth_frame, intrinsic_matrix, cam_to_world_matrix):
    """
    Convert 2D image coordinates (cx, cy) and depth to 3D world coordinates.
    Applies clamping to ensure valid pixel access.

    Args:
        cx (int): X coordinate (center of bbox).
        cy (int): Y coordinate (bottom of bbox).
        depth_frame (ndarray): 2D numpy array with depth values.
        intrinsic_matrix (ndarray): 3x3 camera intrinsics.
        cam_to_world_matrix (ndarray): 4x4 camera-to-world extrinsics.

    Returns:
        np.ndarray or None: 3D world coordinates [x, y, z] or None if invalid.
    """
    h, w = depth_frame.shape
    clamped = False

    if cx < 0:
        print(f"[WARN] cx={cx} < 0 — clamped to 0")
        cx = 0
        clamped = True
    elif cx >= w:
        print(f"[WARN] cx={cx} >= width={w} — clamped to {w - 1}")
        cx = w - 1
        clamped = True

    if cy < 0:
        print(f"[WARN] cy={cy} < 0 — clamped to 0")
        cy = 0
        clamped = True
    elif cy >= h:
        print(f"[WARN] cy={cy} >= height={h} — clamped to {h - 1}")
        cy = h - 1
        clamped = True

    if clamped:
        print(f"[WARN] Pixel ({cx},{cy}) was out of bounds and has been clamped.")

    depth = depth_frame[cy, cx] / 1000.0  # convert to meters
    if depth == 0:
        print(f"[WARN] No depth at ({cx},{cy})")
        return None

    pixel = np.array([cx, cy, 1.0])
    cam_coords = np.linalg.inv(intrinsic_matrix) @ (pixel * depth)
    cam_coords_h = np.append(cam_coords, 1.0)
    world_coords = cam_to_world_matrix @ cam_coords_h

    return world_coords[:3]
        
# class DetectedObject:
#     """
#     Represents individual detected object to track.
#     """
#     def __init__(self, object_id, frame_id, coordinate, confidence, worldcoordinate, feature_path, image_path=None):
#         self.object_id = f"{object_id:08d}" # AKA serial number
#         self.frame_id = frame_id
#         self.feature_path = feature_path
#         self.confidence = confidence
#         self.image_path = image_path
#         if isinstance(coordinate, Coordinate):
#             self.coordinate = coordinate
#         elif isinstance(coordinate, (list, tuple)) and len(coordinate) == 4:
#             self.coordinate = Coordinate(*coordinate)
#         else:
#             raise Exception(f"Unknown coordinate format: {coordinate}")

#         if isinstance(worldcoordinate, WorldCoordinate):
#             self.worldcoordinate = worldcoordinate
#         elif isinstance(worldcoordinate, (list, tuple)) and len(worldcoordinate) == 4:
#             self.worldcoordinate = WorldCoordinate(*worldcoordinate)
#         else:
#             self.worldcoordinate = None
    
class DetectedObject:
    """
    Represents individual detected object to track.
    """
    def __init__(self, object_id, frame_id, coordinate, confidence, worldcoordinate, feature_path, image_path=None, cls=None):
        self.object_id = f"{object_id:08d}"  # AKA serial number
        self.frame_id = frame_id
        self.feature_path = feature_path
        self.confidence = confidence
        self.image_path = image_path
        self.cls = cls  # New: class label

        # Coordinate handling
        if isinstance(coordinate, Coordinate):
            self.coordinate = coordinate
        elif isinstance(coordinate, (list, tuple)) and len(coordinate) == 4:
            self.coordinate = Coordinate(*coordinate)
        else:
            raise Exception(f"Unknown coordinate format: {coordinate}")

        # World coordinate handling (2D or 3D)
        if isinstance(worldcoordinate, (WorldCoordinate, WorldCoordinate3D)):
            self.worldcoordinate = worldcoordinate
        elif isinstance(worldcoordinate, (list, tuple)) and len(worldcoordinate) in (2, 3):
            if len(worldcoordinate) == 2:
                self.worldcoordinate = WorldCoordinate(*worldcoordinate)
            else:
                self.worldcoordinate = WorldCoordinate3D(*worldcoordinate)
        else:
            self.worldcoordinate = None

class Coordinate:
    def __init__(self, x1, y1, x2, y2):
        self.x1 = int(float(x1))
        self.y1 = int(float(y1))
        self.x2 = int(float(x2))
        self.y2 = int(float(y2))

    def __str__(self):
        return(f'{{"x1":{self.x1}, "y1":{self.y1}, "x2":{self.x2}, "y2":{self.y2}}}')

class WorldCoordinate:
    def __init__(self, x, y):
        self.x = float(x)
        self.y = float(y)

    def __str__(self):
        return(f'{{"x":{self.x}, "y":{self.y}}}')

class WorldCoordinate3D:
    def __init__(self, x, y, z):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    def __str__(self):
        return f'{{"x":{self.x}, "y":{self.y}, "z":{self.z}}}'
    
class TrackingCluster:
    def __init__(self, camera_id, offline_id):
        self.camera_id = camera_id
        self.offline_id = 0
        self.global_offline_id = -1
        self.clusters = {}
        self.serials = []

    def add(self, serial):
        if serial in self.serials:
            raise Exception("DUP!")
        self.serials.append(serial)
        

class TrackingClusters:
    def __init__(self, camera_id):
        self.camera_id = camera_id
        self.clusters = []
        self.offline_ids = []

    def add(self, cluster: TrackingCluster):
        cl_id = cluster.offline_id
        if cl_id in self.offline_ids:
            raise Exception("DUP!")
        else:
            self.clusters.append(cluster)

    def get(self, cluster_id):
        if not cluster_id in self.offline_ids:
            raise Exception("No cluster_id registered. {cluster_id}")
        else:
            return self.clusters[offline_ids.index(cluster_id)]

class feature_vector_shed:
    def __init__(self):
        self.features = {}

    def add_vector(self, camera_id, serial_no, npy_path):
        key = camera_id + "_" + serial_no
        if key in self.features:
            print(f"Feature vector of camera ID '{camera_id}' and serial no '{serial_no}' is already exist. ")
            return
            
        if not os.path.isfile(npy_path):
            print(f"The feature vector file '{npy_path}' does not exist. ")
            return
        feature = np.load(npy_path)
        self.features[key] = feature

    def get(self, camera_id, serial_no):
        key = camera_id + "_" + serial_no
        return self.features[key]
