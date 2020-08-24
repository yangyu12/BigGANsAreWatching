import os
import sys
import argparse
import json
import torch
import numpy as np
from tensorboardX import SummaryWriter

import matplotlib
matplotlib.use("Agg")

from gan_mask_gen import MaskGenerator, MaskSynthesizing, it_mask_gen
from data import EvalPLDataset
from metrics import model_metrics, IoU, accuracy
from BigGAN.gan_load import make_big_gan
from postprocessing import PseudoLabelGenerator

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
    parser.add_argument('--gan_weights', type=str, default=BIGBIGAN_WEIGHTS)
    parser.add_argument('--bg_direction', type=str, required=True)

    parser.add_argument('--z', type=str, default=None)
    parser.add_argument('--seed', type=int, default=2)
    parser.add_argument('--val_images_dirs', nargs='*', type=str, default=[None])
    parser.add_argument('--val_masks_dirs', nargs='*', type=str, default=[None])

    args = parser.parse_args()

    # load G
    G = make_big_gan(args.gan_weights).eval().cuda()
    bg_direction = torch.load(args.bg_direction)
    evaluate_pseudo_labels(G, bg_direction, args.z, args.val_images_dirs, args.val_masks_dirs)


@torch.no_grad()
def evaluate(segmentation_model, images_dir, masks_dir, metrics):
    segmentation_dl = torch.utils.data.DataLoader(
        EvalPLDataset(images_dir, masks_dir), 1, shuffle=False
    )

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
def evaluate_pseudo_labels(G, bg_direction, zs, val_images_dirs, val_masks_dirs):
    # load embeddings
    embedding_name = zs.split("/")[-1][:-4]
    zs = torch.from_numpy(np.load(zs))

    evaluation_dict = {}

    for val_imgs, val_dirs in zip(val_images_dirs, val_masks_dirs):
        ds_name = val_imgs.split('/')[-2]
        print('\nEvaluating {}'.format(ds_name))
        if ds_name not in evaluation_dict.keys():
            evaluation_dict[ds_name] = {}

        if not keys_in_dict_tree(evaluation_dict, [ds_name, THR_EVAL_KEY]):
            evaluation_dict[ds_name][THR_EVAL_KEY] = evaluate(
                PseudoLabelGenerator(G, bg_direction, zs, save_dir=f"results/{embedding_name}_sythetic_data"),
                val_imgs, val_dirs, (IoU, accuracy)
            )


if __name__ == '__main__':
    main()
