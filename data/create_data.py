import os
import json
import glob
import torch
import random
import numpy as np
import nibabel as nib
from scipy.ndimage import binary_opening


RAW_DATA_PATTERN = '/home/eytan/storage/staff/eytankats/data/nako_10k/images_mri_stitched//**/wat.nii.gz'
OUTPUT_DIR = '/home/eytan/storage/staff/eytankats/projects/medssl3d/data/npy128_fg80'
DATASET_FILE_NAME = 'dataset.json'
MAX_PATCH_NUM = 1000
PATCH_SIZE = (128, 128, 128)
MIN_FG_RATIO = 0.7

def random_foreground_crop(image, mask, patch_size, min_fg_ratio=0.8, max_tries=100):

    """
    Randomly crop a 3D patch from an image and mask such that
    at least `min_fg_ratio` of the patch is foreground in the mask.

    Args:
        image (ndarray): 3D array of shape (D, H, W).
        mask (ndarray): 3D binary mask (same shape as image).
        patch_size (tuple): (d, h, w) size of the patch to crop.
        min_fg_ratio (float): Minimum ratio of foreground voxels required.
        max_tries (int): Max attempts before giving up.

    Returns:
        cropped_img (ndarray)
    """

    assert image.shape == mask.shape, "Image and mask must have same shape."

    D, H, W = image.shape
    d, h, w = patch_size
    assert d <= D and h <= H and w <= W, "Patch size must fit inside image."

    best_fg_ratio = 0
    for _ in range(max_tries):

        # Sample random start indices
        start_d = np.random.randint(0, D - d + 1)
        start_h = np.random.randint(0, H - h + 1)
        start_w = np.random.randint(0, W - w + 1)

        # Crop patch
        patch_img = image[start_d:start_d + d, start_h:start_h + h, start_w:start_w + w]
        patch_mask = mask[start_d:start_d + d, start_h:start_h + h, start_w:start_w + w]

        # Check foreground ratio
        fg_ratio = patch_mask.sum() / patch_mask.size
        if fg_ratio >= min_fg_ratio:
            return patch_img, start_d, start_h, start_w

        if fg_ratio > best_fg_ratio:
            best_fg_ratio = fg_ratio
            best_patch_img = patch_img
            best_start_d = start_d
            best_start_h = start_h
            best_start_w = start_w

    # If no patch found, return the last one anyway
    print(f'Warning: Could not find patch with required foreground ratio, saving best attempt with ratio {best_fg_ratio}.' )

    if best_fg_ratio == 0:
        return None, None, None, None
    else:
        return best_patch_img, best_start_d, best_start_h, best_start_w

os.makedirs(OUTPUT_DIR, exist_ok=True)
dataset = {"training": []}
img_paths = glob.glob(RAW_DATA_PATTERN)
for _ in range(MAX_PATCH_NUM):

    img_path = random.choice(img_paths)

    # load image
    img_nib = nib.load(img_path)
    img_np = img_nib.get_fdata(dtype=np.float32)  # copy() is a workaround to prevent negative stride that prevents initializing torch image

    # flip to standartized orientation image
    img_np = np.flip(img_np, axis=1)
    img_np = np.flip(img_np, axis=0)
    img_np = np.ascontiguousarray(img_np)  # avoid negative strides

    # resample to isotropic resolution 1.5x1.5x1.5
    img_t = torch.from_numpy(img_np[None, None])  # [N, C, D, H, W]

    zooms = img_nib.header.get_zooms()[:3]
    orig_size = torch.tensor(img_t.shape[-3:])
    target_size = (orig_size * torch.tensor(zooms) / 1.5).round().int().tolist()

    img_t_resampled = torch.nn.functional.interpolate(img_t, size=target_size, mode="trilinear", align_corners=False)
    img_np_resampled = img_t_resampled.squeeze().numpy()

    # get foreground mask
    img_np_normalized = img_np_resampled / np.max(img_np_resampled)
    non_zero_mask = img_np_normalized > 0.02
    non_zero_mask = binary_opening(non_zero_mask, iterations=5) # clean nonzero mask by morphological opening

    # crop patch
    patch, start_d, start_h, start_w = random_foreground_crop(img_np_resampled, non_zero_mask, patch_size=PATCH_SIZE, min_fg_ratio=MIN_FG_RATIO)
    if patch is None:
        continue

    # save patch
    patch_path = os.path.join(OUTPUT_DIR, f'{os.path.basename(os.path.dirname(img_path))}_{start_d}_{start_h}_{start_w}.npy')
    np.save(patch_path, patch)

    # add patch to dataset
    entry = {"image": os.path.basename(patch_path)}
    dataset["training"].append(entry)

# save dataset file
with open(os.path.join(OUTPUT_DIR, DATASET_FILE_NAME), "w") as f:
    json.dump(dataset, f, indent=4)




