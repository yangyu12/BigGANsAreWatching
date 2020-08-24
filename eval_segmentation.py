import os
import sys
import argparse
import json
import torch
import numpy as np
from tensorboardX import SummaryWriter

import matplotlib
matplotlib.use("Agg")

from UNet.unet_model import UNet
from gan_mask_gen import MaskGenerator, MaskSynthesizing, it_mask_gen
from data import SegmentationDataset
from metrics import model_metrics, IoU, accuracy, F_max
from BigGAN.gan_load import make_big_gan
from postprocessing import connected_components_filter,\
    SegmentationInference, Threshold

DEFAULT_EVAL_KEY = 'id'
THR_EVAL_KEY = 'thr'
SEGMENTATION_RES = 128
BIGBIGAN_WEIGHTS = 'BigGAN/weights/BigBiGAN_x1.pth'
LATENT_DIRECTION = 'BigGAN/weights/bg_direction.pth'


MASK_SYNTHEZ_DICT = {
    'lighting': MaskSynthesizing.LIGHTING,
    'mean_thr': MaskSynthesizing.MEAN_THR,
}


def main():
    parser = argparse.ArgumentParser(description='GAN-based unsupervised segmentation train')
    parser.add_argument('--unet_weights', type=str, default="")
    parser.add_argument('--seed', type=int, default=2)
    parser.add_argument('--val_images_dirs', nargs='*', type=str, default=[None])
    parser.add_argument('--val_masks_dirs', nargs='*', type=str, default=[None])

    args = parser.parse_args()

    model = UNet().train().cuda()
    model.load_state_dict(torch.load(args.unet_weights))
    evaluate_all_wrappers(model, args.val_images_dirs, args.val_masks_dirs)


@torch.no_grad()
def evaluate(segmentation_model, images_dir, masks_dir, metrics, size=None):
    segmentation_dl = torch.utils.data.DataLoader(
        SegmentationDataset(images_dir, masks_dir, size=size, crop=False), 1, shuffle=False)

    eval_out = model_metrics(segmentation_model, segmentation_dl, stats=metrics)
    print('Segmenation model', eval_out)
    return eval_out


def keys_in_dict_tree(dict_tree, keys):
    for key in keys:
        if key not in dict_tree.keys():
            return False
        dict_tree = dict_tree[key]
    return True


@torch.no_grad()
def evaluate_all_wrappers(model, val_images_dirs, val_masks_dirs):
    model.eval()
    evaluation_dict = {}

    for val_imgs, val_dirs in zip(val_images_dirs, val_masks_dirs):
        ds_name = val_imgs.split('/')[-2]
        print('\nEvaluating {}'.format(ds_name))
        if ds_name not in evaluation_dict.keys():
            evaluation_dict[ds_name] = {}

        if not keys_in_dict_tree(evaluation_dict, [ds_name, DEFAULT_EVAL_KEY]):
            evaluation_dict[ds_name][DEFAULT_EVAL_KEY] = evaluate(
                SegmentationInference(model, resize_to=SEGMENTATION_RES),
                val_imgs, val_dirs, (F_max,)
            )

        if not keys_in_dict_tree(evaluation_dict, [ds_name, THR_EVAL_KEY]):
            evaluation_dict[ds_name][THR_EVAL_KEY] = evaluate(
                Threshold(model, resize_to=SEGMENTATION_RES), val_imgs, val_dirs, (IoU, accuracy)
            )


if __name__ == '__main__':
    main()
