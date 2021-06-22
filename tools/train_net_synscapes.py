#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
"""
A main training script.

This scripts reads a given config file and runs the training or evaluation.
It is an entry point that is made to train standard models in detectron2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use detectron2 as an library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.
"""

import logging
import os
from collections import OrderedDict
import torch

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    verify_results,
)
from detectron2.modeling import GeneralizedRCNNWithTTA
from detectron2.data import DatasetCatalog
from collections import namedtuple
from detectron2.data.datasets.synscapes_panoptic import register_synscapes_panoptic_separated
from detectron2.data.datasets.builtin_meta import CITYSCAPES_CATEGORIES


class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains pre-defined default logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can write your
    own training loop. You can use "tools/plain_train_net.py" as an example.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
            evaluator_list.append(
                SemSegEvaluator(
                    dataset_name,
                    distributed=True,
                    output_dir=output_folder,
                )
            )
        if evaluator_type in ["coco", "coco_panoptic_seg"]:
            evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
        if evaluator_type == "coco_panoptic_seg":
            evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
        if evaluator_type == "cityscapes_instance":
            assert (
                torch.cuda.device_count() >= comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesInstanceEvaluator(dataset_name)
        if evaluator_type in ["cityscapes_panoptic_seg", "coco_panoptic_seg"]:
            evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
        if evaluator_type == "cityscapes_panoptic_seg":
            assert (
                torch.cuda.device_count() >= comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            evaluator_list.append(CityscapesSemSegEvaluator(dataset_name))
            evaluator_list.append(CityscapesInstanceEvaluator(dataset_name))
        if evaluator_type == "cityscapes_sem_seg":
            assert (
                torch.cuda.device_count() >= comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesSemSegEvaluator(dataset_name)
        elif evaluator_type == "pascal_voc":
            return PascalVOCDetectionEvaluator(dataset_name)
        elif evaluator_type == "lvis":
            return LVISEvaluator(dataset_name, output_dir=output_folder)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.DATASETS.TRAIN = ("synscapes_train_separated",)
    cfg.DATASETS.TEST = ("synscapes_val_separated",)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):

    # thing classes
    thing_classes = MetadataCatalog.get('cityscapes_fine_panoptic_val').thing_dataset_id_to_contiguous_id.keys()
    # stuff classes
    stuff_classes = MetadataCatalog.get('cityscapes_fine_panoptic_val').stuff_dataset_id_to_contiguous_id.keys()

    meta = {}
    #meta['evaluator_type']="cityscapes_panoptic_seg"
    meta['thing_classes'] = [i['name'] for i in CITYSCAPES_CATEGORIES if i['id'] in thing_classes]
    meta['thing_colors'] = [i['color'] for i in CITYSCAPES_CATEGORIES if i['id'] in thing_classes]
    meta['stuff_classes'] = []
    meta['stuff_classes'].append('things')
    meta['stuff_classes'].extend([i['name'] for i in CITYSCAPES_CATEGORIES if i['id'] in stuff_classes])
    meta['stuff_colors'] = []
    meta['stuff_colors'].append((250, 250, 250))
    meta['stuff_colors'].extend([i['color'] for i in CITYSCAPES_CATEGORIES if i['id'] in stuff_classes])
    thing_id_map = dict(zip(thing_classes, list(range(8))))
    meta['thing_dataset_id_to_contiguous_id'] = thing_id_map
    stuff_plus_one = []
    stuff_plus_one.append(0)
    stuff_plus_one.extend(stuff_classes)
    stuff_id_map = dict(zip(stuff_plus_one, list(range(12))))
    meta['stuff_dataset_id_to_contiguous_id'] = stuff_id_map
    meta['label_divisor'] = 1e5

    register_synscapes_panoptic_separated(name='synscapes_train', metadata=meta, image_root='/srip21-data/models/detectron2/datasets/synscapes/final/train/rgb', panoptic_root='/srip21-data/models/detectron2/datasets/synscapes/final/train/panoptic', panoptic_json='/srip21-data/models/detectron2/datasets/synscapes/final/train/panoptic_train.json',sem_seg_root='/srip21-data/models/detectron2/datasets/synscapes/final/train/sem_seg/', instances_json='/srip21-data/models/detectron2/datasets/synscapes/final/train/train_instance.json')
    register_synscapes_panoptic_separated(name='synscapes_val', metadata=meta, image_root='/srip21-data/models/detectron2/datasets/synscapes/final/val/rgb', panoptic_root='/srip21-data/models/detectron2/datasets/synscapes/final/val/panoptic', panoptic_json='/srip21-data/models/detectron2/datasets/synscapes/final/val/panoptic_val.json',sem_seg_root='/srip21-data/models/detectron2/datasets/synscapes/final/val/sem_seg/', instances_json='/srip21-data/models/detectron2/datasets/synscapes/final/val/val_instance.json') 
   

    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop (see plain_train_net.py) or
    subclassing the trainer.
    """
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    if cfg.TEST.AUG.ENABLED:
        trainer.register_hooks(
            [hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model))]
        )
    return trainer.train()


if __name__ == "__main__":
    
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
