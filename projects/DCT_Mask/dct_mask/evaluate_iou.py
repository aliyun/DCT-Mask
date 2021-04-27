# coding:utf-8

import os
import argparse
import numpy as np
import json
import cv2
import torch
import copy
from typing import Any, Iterator, List, Union
from detectron2.structures import (
    Boxes,
    PolygonMasks,
    BoxMode,
    polygons_to_bitmask
)
from mask_encoding import DctMaskEncoding


class IOUMetric(object):
    """
    Class to calculate mean-iou using fast_hist method
    """

    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.hist = np.zeros((num_classes, num_classes))

    def _fast_hist(self, label_pred, label_true):
        mask = (label_true >= 0) & (label_true < self.num_classes)
        hist = np.bincount(
            self.num_classes * label_true[mask].astype(int) +
            label_pred[mask], minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)
        return hist

    def add_batch(self, predictions, gts):
        for lp, lt in zip(predictions, gts):
            self.hist += self._fast_hist(lp.flatten(), lt.flatten())

    def evaluate(self):
        acc = np.diag(self.hist).sum() / self.hist.sum()
        acc_cls = np.diag(self.hist) / self.hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(self.hist) / (self.hist.sum(axis=1) + self.hist.sum(axis=0) - np.diag(self.hist))
        mean_iu = np.nanmean(iu)
        freq = self.hist.sum(axis=1) / self.hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        return acc, acc_cls, iu, mean_iu, fwavacc


def rasterize_polygons_within_box_for_arbitrary_shape(
        polygons: List[np.ndarray], box: np.ndarray, mask_size_h: int, mask_size_w: int
) -> torch.Tensor:
    """
    New rasterize. Rasterize polygons within box with specific size.
    Args:
        polygons (list[ndarray[float]]): a list of polygons, which represents an instance.
        box: 4-element numpy array
        mask_size_h (int):
        mask_size_w (int):

    Returns:
        Tensor: BoolTensor of shape (mask_size, mask_size)
    """
    # 1. Shift the polygons w.r.t the boxes
    w, h = box[2] - box[0], box[3] - box[1]

    polygons = copy.deepcopy(polygons)
    for p in polygons:
        p[0::2] = p[0::2] - box[0]
        p[1::2] = p[1::2] - box[1]

    # 2. Rescale the polygons to the new box size
    # max() to avoid division by small number
    ratio_h = mask_size_h / max(h, 0.1)
    ratio_w = mask_size_w / max(w, 0.1)

    if ratio_h == ratio_w:
        for p in polygons:
            p *= ratio_h
    else:
        for p in polygons:
            p[0::2] *= ratio_w
            p[1::2] *= ratio_h

    # 3. Rasterize the polygons with coco api
    mask = polygons_to_bitmask(polygons, mask_size_h, mask_size_w)
    mask = torch.from_numpy(mask)
    return mask


def valid_dct_source(coco, dct_dim, mask_size):
    dct_mask_encoding = DctMaskEncoding(dct_dim, mask_size)
    mIoU = []
    Number = 0
    for ann in coco:
        Number += 1
        bbox = np.array([ann["bbox"]])  # xmin, ymin, w, h
        w, h = bbox[0][2], bbox[0][3]
        w, h = round(w), round(h)
        bbox = BoxMode.convert(bbox, BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)  # x1y1x2y2
        bbox = Boxes(bbox)

        # mask transform.
        mask = PolygonMasks([ann["segmentation"]])
        mask_source = rasterize_polygons_within_box_for_arbitrary_shape(mask.polygons[0], bbox.tensor[0].numpy(), h, w)
        mask_source = mask_source.numpy()  # numpy [h,w] binary
        
        mask_k = mask.crop_and_resize(bbox.tensor, mask_size).float()  # tensor [1,28,28],all 0 or 1
        mask_k = mask_k.view([mask_size, mask_size])
        dct_code = dct_mask_encoding.encode(mask_k)
        mask_re = dct_mask_encoding.decode(dct_code).numpy().squeeze()
        res = cv2.resize(mask_re.astype('float'),
                         dsize=(mask_source.shape[1], mask_source.shape[0]),
                         interpolation=cv2.INTER_LINEAR)
        
        res = np.where(res >= 0.5, 1, 0)
        res = np.reshape(res, [1, -1])
        mask_source = np.reshape(mask_source, [1, -1])
        res = res.astype(int)
        
        IoUevaluate = IOUMetric(2)
        IoUevaluate.add_batch(res, mask_source)
        
        _, _, _, mean_iu, _ = IoUevaluate.evaluate()
        mIoU.append(mean_iu)
        if Number % 1000 == 1:
            print(np.mean(mIoU))
    return np.mean(mIoU)


DATASETS = {
    "coco_2017_train": {
        "img_dir": "coco/train2017",
        "ann_file": "coco/annotations/instances_train2017.json"
    },
    "coco_2017_val": {
        "img_dir": "coco/val2017",
        "ann_file": "coco/annotations/instances_val2017.json"
    }
}


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluation for DCT-Mask Encoding.')
    parser.add_argument('--root', default='datasets', type=str)
    parser.add_argument('--dataset', default='coco_2017_val', type=str)
    # mask encoding params.
    parser.add_argument('--mask_size', default=128, type=int)
    parser.add_argument('--dim', default=300, type=int)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    mask_size = args.mask_size
    dim = args.dim
    
    dataset_root = args.root
    
    data_info = DATASETS[args.dataset]
    img_dir, ann_file = data_info['img_dir'], data_info['ann_file']
    img_dir = os.path.join(dataset_root, img_dir)  # actually we do not use it.
    ann_file = os.path.join(dataset_root, ann_file)

    with open(ann_file, 'r') as f:
        anns = json.load(f)
    anns = anns['annotations']
    coco = list()
    for ann in anns:
        if ann.get('iscrowd', 0) == 0:
            coco.append(ann)
    print("Removed {} images with no usable annotations. {} images left.".format(
        len(anns) - len(coco), len(coco)))

    mean_iou = valid_dct_source(coco, dct_dim=dim, mask_size=mask_size)
    print('mask_size: {}, dim: {}, mIoU: {}'.format(mask_size, dim, mean_iou))
