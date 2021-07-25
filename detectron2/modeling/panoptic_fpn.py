import torch
from torch import nn
from typing import Dict, Tuple
from detectron2.structures import ImageList


class PanopticFPN(nn.Module):
    def __init__(self, backbone: Backbone, sem_seg_head: nn.Module, combine_overlap_thresh: float=0.5, combine_stuff_area_thresh: float=4096, combine_instances_score_thresh: float=0.5, **kwargs):
        '''
        @param:
        sem_seg_head: a module for the semantic segmentation head.
        combine_overlap_thresh: combine masks into one instances if
                they have enough overlap
        combine_stuff_area_thresh: ignore stuff areas smaller than this threshold
        combine_instances_score_thresh: ignore instances whose score is
                smaller than this threshold
        '''
        super().__init__()
        self.backbone = backbone
        self.sem_seg_head = sem_seg_head
        self.combine_overlap_thresh = combine_overlap_thresh
        self.combine_stuff_area_thresh = combine_stuff_area_thresh
        self.combine_instances_score_thresh = combine_instances_score_thresh

    def preprocess_image(self, batched_inputs: Tuple[Dict[str, torch.Tensor]]):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images

    def forward(self, batched_source, batched_target):
        if batched_target is None:
            images_source = self.preprocess_image(batched_source)
            features_source = self.backbone(images_source.tensor)

            gt_source_sem_seg = [x["sem_seg"].to(self.device) for x in batched_source]
            gt_source_sem_seg = ImageList.from_tensors(
                gt_source_sem_seg, self.backbone.size_divisibility, self.sem_seg_head.ignore_value
            ).tensor
            sem_seg_results_source, sem_seg
