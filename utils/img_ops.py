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