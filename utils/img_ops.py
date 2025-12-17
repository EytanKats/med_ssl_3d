import numpy as np

from monai.transforms import GaussianSmooth

from scipy.ndimage import binary_fill_holes
from scipy.ndimage import label as scipy_label
from skimage.morphology import binary_closing, ball, binary_opening
from skimage.measure import regionprops


def extract_foreground_mask_mri(volume, smooth_sigma=1.0):

    # Smooth to reduce noise
    smooth = GaussianSmooth(sigma=smooth_sigma)(volume[0])[0].numpy()

    # Adaptive threshold (Otsu-like heuristic)
    threshold = np.percentile(smooth, 40)
    mask = smooth > threshold

    # Morphological cleanup
    mask = binary_closing(mask, ball(7))
    mask = binary_fill_holes(mask)

    # Keep only largest connected component
    labeled, num = scipy_label(mask)
    if num > 0:
        counts = np.bincount(labeled.ravel())
        counts[0] = 0  # ignore background
        mask = labeled == np.argmax(counts)

    return mask.astype(np.uint8)


def extract_foreground_mask_ct(ct_volume):

    # 1. Threshold: everything above ~-400 HU is likely body/tissue
    ct_volume = ct_volume.squeeze().numpy()
    mask = ct_volume > 0.2

    # 2. Morphological cleanup (close small holes, remove isolated noise)
    mask = binary_closing(mask, footprint=np.ones((5, 5, 5)))
    mask = binary_opening(mask, footprint=np.ones((3, 3, 3)))
    mask = binary_fill_holes(mask)

    # 3. Keep only the largest connected component (the patient body)
    labeled_mask, num_features = scipy_label(mask)
    if num_features > 1:
        regions = regionprops(labeled_mask)
        largest_region = max(regions, key=lambda r: r.area)
        mask = labeled_mask == largest_region.label

    return mask.astype(np.uint8)


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
    if d >= D or h >= H or w >= W:
        print(f'Warning: Patch size must fit inside image. Patch size: {d}, {h}, {w}. Image size: {D}, {H}, {W}')
        return None, None, None, None

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

    if best_fg_ratio == 0: # If there is no foreground found return Nones
        return None, None, None, None
    else: # If no patch found, return the last one anyway
        print(f'Warning: Could not find patch with required foreground ratio, saving best attempt with ratio {best_fg_ratio}.')
        return best_patch_img, best_start_d, best_start_h, best_start_w
