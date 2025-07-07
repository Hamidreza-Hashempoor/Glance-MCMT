import os
from trackeval.datasets._base_dataset import _BaseDataset as BaseDataset
import numpy as np

class AICity3D(BaseDataset):
    def __init__(self, config=None):
        super().__init__()
        self.config = self.get_default_dataset_config()
        if config is not None:
            self.config.update(config)
        self.tracker_list = self.config['TRACKERS_TO_EVAL']
        self.seq_list = self.config['SEQ_TO_EVAL']
        self.class_list = self.config['CLASSES_TO_EVAL']

        self.should_classes_combine = self.config.get('OUTPUT_EMPTY_CLASSES', False)
        self.use_super_categories = False

        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        self.output_fol = os.path.join(root_dir, 'data/AIC25_Track1/outputs')  # or your actual output path
        self.output_sub_fol = ''

    def get_output_fol(self, tracker):
        return os.path.join(self.output_fol, tracker)

    def get_default_dataset_config(self):
        # this_dir = os.path.dirname(os.path.realpath(__file__))
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        default_config = {
            'GT_FOLDER': os.path.join(root_dir, 'data/AIC25_Track1/gt'),
            'TRACKERS_FOLDER': os.path.join(root_dir, 'data/AIC25_Track1/tracker'),
            'SEQ_TO_EVAL': ['exp57'],
            'TRACKERS_TO_EVAL': ['exp57'],
            'SKIP_SPLIT_FOL': True,
            'DO_PREPROC': False,
            'GT_LOC_FORMAT': '{gt_folder}/{seq}.txt',
            'TRACKER_LOC_FORMAT': '{tracker_folder}/{tracker}/{seq}.txt',
            'CLASSES_TO_EVAL': ['pedestrian'],  # not used, but required
            'OUTPUT_EMPTY_CLASSES': False,
            'TRACKER_SUB_FOLDER': '',  # set if tracker name = subfolder
        }
        return default_config

    # def get_preprocessed_seq_data(self, raw_data, cls):
    #     # No class-specific filtering needed, so return all data
    #     return raw_data

    def get_preprocessed_seq_data(self, raw_data, cls):
        cls_id = 1  # assuming 'pedestrian' is class 1

        data = {}
        data['num_timesteps'] = raw_data['num_timesteps']
        data['similarity_scores'] = raw_data['similarity_scores']

        # Filter by class
        gt_ids = []
        tracker_ids = []
        gt_dets = []
        tracker_dets = []
        tracker_confidences = []

        num_gt_dets = 0
        num_tracker_dets = 0

        for t in range(raw_data['num_timesteps']):
            cls_mask_gt = raw_data['gt_classes'][t] == cls_id
            cls_mask_tracker = raw_data['tracker_classes'][t] == cls_id

            gt_ids.append(raw_data['gt_ids'][t][cls_mask_gt])
            tracker_ids.append(raw_data['tracker_ids'][t][cls_mask_tracker])
            gt_dets.append(raw_data['gt_dets'][t][cls_mask_gt])
            tracker_dets.append(raw_data['tracker_dets'][t][cls_mask_tracker])
            tracker_confidences.append(raw_data['tracker_confidences'][t][cls_mask_tracker])

            num_gt_dets += cls_mask_gt.sum()
            num_tracker_dets += cls_mask_tracker.sum()

        data['gt_ids'] = gt_ids
        data['tracker_ids'] = tracker_ids
        data['gt_dets'] = gt_dets
        data['tracker_dets'] = tracker_dets
        data['tracker_confidences'] = tracker_confidences
        data['num_gt_dets'] = num_gt_dets
        data['num_tracker_dets'] = num_tracker_dets

        if gt_ids and np.any([len(g) > 0 for g in gt_ids]):
            data['num_gt_ids'] = int(np.max(np.concatenate(gt_ids)) + 1)
        else:
            data['num_gt_ids'] = 0
        if tracker_ids and np.any([len(t) > 0 for t in tracker_ids]):
            data['num_tracker_ids'] = int(np.max(np.concatenate(tracker_ids)) + 1)
        else:
            data['num_tracker_ids'] = 0
        
        data['dataset'] = self
        return data
    
    def _load_raw_file(self, tracker, seq, is_gt):
        file_path = self.config['GT_LOC_FORMAT'].format(
            gt_folder=self.config['GT_FOLDER'], seq=seq) if is_gt else \
            self.config['TRACKER_LOC_FORMAT'].format(
                tracker_folder=self.config['TRACKERS_FOLDER'], tracker=tracker, seq=seq)

        # Initialize per-frame storage
        num_timesteps = 0
        dets_per_t = {}
        ids_per_t = {}

        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) < 10:
                    continue
                frame = int(parts[0])
                obj_id = int(parts[1])
                x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
                l, w, h = float(parts[5]), float(parts[6]), float(parts[7])
                conf = float(parts[8])

                det = [x, y, z, l, w, h]
                if frame not in dets_per_t:
                    dets_per_t[frame] = []
                    ids_per_t[frame] = []
                dets_per_t[frame].append(det)
                ids_per_t[frame].append(obj_id)

                num_timesteps = max(num_timesteps, frame)

        # Pack into lists by timestep
        det_list = []
        id_list = []
        for t in range(1, num_timesteps + 1):
            det_list.append(np.array(dets_per_t.get(t, [])))
            id_list.append(np.array(ids_per_t.get(t, [])))

        output = {
            'num_timesteps': num_timesteps,
            'gt_dets' if is_gt else 'tracker_dets': det_list,
            'gt_ids' if is_gt else 'tracker_ids': id_list,
            'gt_classes' if is_gt else 'tracker_classes': [np.ones_like(ids) for ids in id_list],
            'gt_crowd_ignore_regions' if is_gt else None: [[] for _ in range(num_timesteps)],
        }

        if not is_gt:
            output['tracker_confidences'] = [np.ones(len(d)) for d in det_list]

        return output

    # def _load_raw_file(self, tracker, seq, is_gt):
    #     if is_gt:
    #         file_path = self.config['GT_LOC_FORMAT'].format(
    #             gt_folder=self.config['GT_FOLDER'], seq=seq)
    #     else:
    #         file_path = self.config['TRACKER_LOC_FORMAT'].format(
    #             tracker_folder=self.config['TRACKERS_FOLDER'], tracker=tracker, seq=seq)

    #     raw_data = {}
    #     with open(file_path, 'r') as f:
    #         for line in f:
    #             if not line.strip():
    #                 continue
    #             parts = line.strip().split(',')
    #             if len(parts) < 10:
    #                 continue
    #             frame = int(parts[0])
    #             obj_id = int(parts[1])
    #             x, y, z = map(float, parts[2:5])
    #             l, w, h = map(float, parts[5:8])
    #             conf = float(parts[8])
    #             det = [x, y, z, l, w, h, conf]
    #             raw_data.setdefault(frame, {})[obj_id] = det
    #     return raw_data
    
    def get_similarities(self, gt_dets_t, tracker_dets_t):
        return self._calculate_similarities(gt_dets_t, tracker_dets_t)

    def _calculate_similarities(self, gt_dets_t, tracker_dets_t):
        """Used by HOTA to compute pairwise 3D IoUs"""
        similarity = np.zeros((len(gt_dets_t), len(tracker_dets_t)), dtype=np.float32)
        for i, gt in enumerate(gt_dets_t):
            for j, trk in enumerate(tracker_dets_t):
                similarity[i, j] = self._compute_3d_iou(gt, trk)
        return similarity

    @staticmethod
    def _compute_3d_iou(boxA, boxB):
        x1, y1, z1, l1, w1, h1 = boxA[:6]
        x2, y2, z2, l2, w2, h2 = boxB[:6]

        minA = np.array([x1 - l1/2, y1 - w1/2, z1 - h1/2])
        maxA = np.array([x1 + l1/2, y1 + w1/2, z1 + h1/2])
        minB = np.array([x2 - l2/2, y2 - w2/2, z2 - h2/2])
        maxB = np.array([x2 + l2/2, y2 + w2/2, z2 + h2/2])

        inter_min = np.maximum(minA, minB)
        inter_max = np.minimum(maxA, maxB)
        inter_dims = np.maximum(inter_max - inter_min, 0)
        inter_vol = np.prod(inter_dims)

        volA = np.prod(maxA - minA)
        volB = np.prod(maxB - minB)
        union_vol = volA + volB - inter_vol

        return inter_vol / union_vol if union_vol > 0 else 0.0