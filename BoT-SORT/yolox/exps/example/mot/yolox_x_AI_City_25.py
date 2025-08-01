# encoding: utf-8
import os
import random
import torch
import torch.nn as nn
import torch.distributed as dist

from yolox.exp import Exp as MyExp
from yolox.data import get_yolox_datadir

class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.num_classes = 6
        self.class_names = [
                                "Person",
                                "Forklift",
                                "NovaCarter",
                                "Transporter",
                                "FourierGR1T2",
                                "AgilityDigit"
                            ]
        self.depth = 1.33
        self.width = 1.25
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        self.train_ann = "train_aicity_25.json"
        # self.val_ann = "test.json"    # change to train.json when running on training set
        self.val_ann = "val_aicity_25.json"    # change to train.json when running on training set
        self.input_size = (800, 1440)
        self.test_size = (800, 1440)
        self.random_size = (18, 32)
        self.max_epoch = 15
        self.print_interval = 20
        self.eval_interval = 5
        self.test_conf = 0.001
        self.nmsthre = 0.7
        self.no_aug_epochs = 10
        self.basic_lr_per_img = 0.001 / 64.0
        self.warmup_epochs = 1
        self.MAX_SIZE = 30000

    def get_data_loader(self, batch_size, is_distributed, no_aug=False):
        from yolox.data import (
            MOTDataset,
            TrainTransform,
            YoloBatchSampler,
            DataLoader,
            InfiniteSampler,
            MosaicDetection,
        )

        dataset_mot = MOTDataset(
            # data_dir=os.path.join(get_yolox_datadir(), "mix_det"),
            # data_dir = os.path.join("AIC25_Track1", "Train"),
            data_dir = "AIC25_Track1/Train",
            json_file=self.train_ann,
            name='',
            img_size=self.input_size,
            preproc=TrainTransform(
                rgb_means=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_labels=500,
            ),
            MAX_SIZE= self.MAX_SIZE
        )
        self.dataset_mot = dataset_mot

        dataset = MosaicDetection(
            dataset_mot,
            mosaic=not no_aug,
            img_size=self.input_size,
            preproc=TrainTransform(
                rgb_means=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_labels=1000,
            ),
            degrees=self.degrees,
            translate=self.translate,
            scale=self.scale,
            shear=self.shear,
            perspective=self.perspective,
            enable_mixup=self.enable_mixup,
        )

        self.dataset = dataset

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()

        sampler = InfiniteSampler(
            len(self.dataset), seed=self.seed if self.seed else 0
        )

        batch_sampler = YoloBatchSampler(
            sampler=sampler,
            batch_size=batch_size,
            drop_last=False,
            input_dimension=self.input_size,
            mosaic=not no_aug,
        )

        dataloader_kwargs = {"num_workers": self.data_num_workers, "pin_memory": True}
        dataloader_kwargs["batch_sampler"] = batch_sampler
        train_loader = DataLoader(self.dataset, **dataloader_kwargs)

        return train_loader

    def get_eval_loader(self, batch_size, is_distributed, testdev=False, data_dir=None):
        from yolox.data import MOTDataset, ValTransform

        valdataset = []
        if self.dataset_mot is None:
            valdataset = MOTDataset(
                # data_dir=data_dir if data_dir is not None else os.path.join(get_yolox_datadir(), "MOT17"),
                data_dir = "AIC25_Track1/Train",
                json_file=self.val_ann,
                img_size=self.test_size,
                name='train',   # change to train when running on training set
                preproc=ValTransform(
                    rgb_means=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
                MAX_SIZE=self.MAX_SIZE
            )
        else:
            valdataset = self.dataset_mot

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()
            sampler = torch.utils.data.distributed.DistributedSampler(
                valdataset, shuffle=False
            )
        else:
            sampler = torch.utils.data.SequentialSampler(valdataset)

        dataloader_kwargs = {
            "num_workers": self.data_num_workers,
            "pin_memory": True,
            "sampler": sampler,
        }
        dataloader_kwargs["batch_size"] = batch_size
        val_loader = torch.utils.data.DataLoader(valdataset, **dataloader_kwargs)

        return val_loader

    def get_evaluator(self, batch_size, is_distributed, testdev=False, val_loader=None):
        from yolox.evaluators import COCOEvaluator

        if val_loader is None:
            val_loader = self.get_eval_loader(batch_size, is_distributed, testdev=testdev)
        evaluator = COCOEvaluator(
            dataloader=val_loader,
            img_size=self.test_size,
            confthre=self.test_conf,
            nmsthre=self.nmsthre,
            num_classes=self.num_classes,
            testdev=testdev,
        )
        return evaluator
