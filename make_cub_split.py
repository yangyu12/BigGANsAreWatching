import numpy as np
import os
from PIL import Image
from tqdm import tqdm


def main(data="images", set="train", crop=True):
    DATA_PATH = "./data/CUB"
    SAVE_DIR = os.path.join(DATA_PATH, f"{set}_crop_{data}" if crop else f"{set}_{data}")
    os.makedirs(SAVE_DIR, exist_ok=True)

    splits = np.loadtxt(os.path.join(DATA_PATH, "train_val_test_split.txt"), int)
    files = np.loadtxt(os.path.join(DATA_PATH, "images.txt"), str)[:, 1]
    bbox_x_y_w_h = np.loadtxt(os.path.join(DATA_PATH, "bounding_boxes.txt"), float)
    bbox_x_y_w_h = bbox_x_y_w_h.astype(np.int)

    idx = {"train": 0, "val": 1, "test": 2}[set]
    files = files[splits[:, 1] == idx]
    bbox_x_y_w_h = bbox_x_y_w_h[splits[:, 1] == idx, 1:]

    for filename, (x1, y1, w, h) in tqdm(zip(files, bbox_x_y_w_h)):
        if data == "images":
            img = Image.open(os.path.join(DATA_PATH, data, filename)).convert("RGB")
        else:
            img = Image.open(os.path.join(DATA_PATH, data, filename[:-3] + "png"))
        if crop:
            img = img.crop((x1, y1, x1+w-1, y1+h-1))
        if data == "images":
            save_file = os.path.join(SAVE_DIR, filename)
        else:
            save_file = os.path.join(SAVE_DIR, filename[:-3] + "png")
        os.makedirs(os.path.dirname(save_file), exist_ok=True)
        img.save(save_file)


# def copy_all():
#     DATA_PATH = "./data/CUB"
#     SAVE_DIR = os.path.join(DATA_PATH, "all_images")
#     os.makedirs(SAVE_DIR, exist_ok=True)
#
#     files = np.loadtxt(os.path.join(DATA_PATH, "images.txt"), str)[:, 1]
#     for filename in tqdm(files):
#         img = Image.open(os.path.join(DATA_PATH, "images", filename)).convert("RGB")
#         img.save(os.path.join(SAVE_DIR, filename.split("/")[-1]))


if __name__ == '__main__':
    main(data="images", set="test", crop=False)
    main(data="segmentations", set="test", crop=False)
