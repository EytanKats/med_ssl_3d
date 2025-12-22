import sys
sys.path.append(".")

import os
import json
import glob
import torch
import random
import numpy as np
import nibabel as nib
from joblib import Parallel, delayed

from utils.img_ops import extract_foreground_mask_mri, extract_foreground_mask_ct, random_foreground_crop


RAW_DATA_PATTERN_MRI = '/home/eytan/storage/staff/eytankats/data/nako_10k/images_mri_stitched//**/wat.nii.gz'
RAW_DATA_PATTERN_CT = '/home/eytan/storage/datasets/TotalSegmentatorV2//**/ct.nii.gz'
OUTPUT_DIR = '/home/eytan/storage/staff/eytankats/projects/medssl3d/data/patches_unpaired_nako_tsct'
DATASET_FILE_NAME = 'dataset.json'
MAX_PATCH_NUM = 10000
PATCH_SIZE = (128, 128, 128)
MIN_FG_RATIO = 0.7

os.makedirs(OUTPUT_DIR, exist_ok=True)

img_paths_mri = glob.glob(RAW_DATA_PATTERN_MRI)
img_paths_ct = glob.glob(RAW_DATA_PATTERN_CT)


def crop_and_save(cnt):
    patch_found = False
    while not patch_found:

        if cnt % 2 == 0:
            img_path = random.choice(img_paths_mri)
        else:
            img_path = random.choice(img_paths_ct)

        # load image
        img_nib = nib.load(img_path)
        img_np = img_nib.get_fdata(dtype=np.float32)

        # flip to standartized orientation image
        img_np = np.flip(img_np, axis=1)
        img_np = np.flip(img_np, axis=0)
        img_np = np.ascontiguousarray(img_np)  # avoid negative strides

        # resample to isotropic resolution
        img_t = torch.from_numpy(img_np[None, None])  # [N, C, D, H, W]

        resolution = torch.rand(1)[0] + 1  # get resolution between 1 and 2

        zooms = img_nib.header.get_zooms()[:3]
        orig_size = torch.tensor(img_t.shape[-3:])
        target_size = (orig_size * torch.tensor(zooms) / resolution).round().int().tolist()

        img_t_resampled = torch.nn.functional.interpolate(img_t, size=target_size, mode="trilinear", align_corners=False)
        img_np_resampled = img_t_resampled.squeeze().numpy()

        # get foreground mask
        if cnt % 2 == 0:
            non_zero_mask = extract_foreground_mask_mri(img_t_resampled)
        else:
            non_zero_mask = extract_foreground_mask_ct(img_t_resampled)

        # crop patch
        patch, start_d, start_h, start_w = random_foreground_crop(img_np_resampled, non_zero_mask, patch_size=PATCH_SIZE, min_fg_ratio=MIN_FG_RATIO)
        if patch is not None:
            patch_found = True

    # import matplotlib.pyplot as plt
    # plt.imshow(patch[:, :, 64], cmap='gray')
    # plt.show()

    # save patch
    patch_path = os.path.join(OUTPUT_DIR, f'{os.path.basename(os.path.dirname(img_path))}_{start_d}_{start_h}_{start_w}.npy')
    np.save(patch_path, patch)

results = Parallel(n_jobs=10)(
    delayed(crop_and_save)(patch_idx) for patch_idx in range(MAX_PATCH_NUM)
)

dataset = {"training": []}
patches = glob.glob(os.path.join(OUTPUT_DIR, '*.npy'))
for patch in patches:

    # add patch to dataset
    entry = {"image": os.path.basename(patch)}
    dataset["training"].append(entry)

# save dataset file
with open(os.path.join(OUTPUT_DIR, DATASET_FILE_NAME), "w") as f:
    json.dump(dataset, f, indent=4)




