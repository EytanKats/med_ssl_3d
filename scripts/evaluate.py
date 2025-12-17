import sys
sys.path.append(".")

import os
import ants
import json
import torch
import numpy as np
import pandas as pd

from monai.bundle import ConfigParser

import matplotlib.pyplot as plt

from utils.img_ops import extract_foreground_mask_mri, extract_foreground_mask_ct
from utils.metrics import dice_coeff
from utils.misc import jacobian_metrics
from utils.pca import pca_lowrank_transform
from utils.convex_adam_utils import MINDSSC
from utils.imports import import_module_from_path
from utils.convexAdam_3D import  convex_adam_3d_param


def plot_registration_summary(
    fixed_img, moving_img, warped_img,
    fixed_lbl, moving_lbl, warped_lbl,
    disp_field,  # (3, H, W, D)
    init_dice, final_dice, sdlogj, fold_ratio,
    lbl_num,
    slice_idx=None,
    save_plot=False,
    output_path=""
):
    """
    Visualize 3D registration result and metrics.
    Args:
        fixed_img, moving_img, warped_img: 3D numpy arrays (H, W, D)
        fixed_lbl, moving_lbl, warped_lbl: 3D integer label maps (H, W, D)
        disp_field: 4D numpy array (3, H, W, D)
        init_dice, final_dice, sdlogj, fold_ratio: floats
        lbl_num: int, number of labels
        slice_idx: optional index along last dimension (D); defaults to center
        save_plot: bool, whether to save plot
        output_path: str, path to save plot
    """
    # Choose central slice if not given
    if slice_idx is None:
        slice_idx = fixed_img.shape[-1] // 2

    # Extract 2D slices
    fx = fixed_img[:, :, slice_idx]
    mv = moving_img[:, :, slice_idx]
    wp = warped_img[:, :, slice_idx]

    fxl = fixed_lbl[:, :, slice_idx]
    mvl = moving_lbl[:, :, slice_idx]
    wpl = warped_lbl[:, :, slice_idx]

    # Create figure
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    cmap_img = "gray"

    # --- Row 1: images + field ---
    def plot_deformed_grid(ax, disp_slice, step=8, color='green', lw=0.8, title="Deformed Grid"):
        """Visualize a 2D displacement field as a deformed grid."""
        H, W, _ = disp_slice.shape
        grid_y, grid_x = np.mgrid[0:H:step, 0:W:step]
        disp_y = disp_slice[::step, ::step, 0]
        disp_x = disp_slice[::step, ::step, 1]
        ax.plot(grid_x + disp_x, grid_y + disp_y, color=color, lw=lw)
        ax.plot((grid_x + disp_x).T, (grid_y + disp_y).T, color=color, lw=lw)
        ax.set_title(title)
        ax.axis("off")

    axes[0, 0].imshow(fx, cmap=cmap_img)
    axes[0, 0].set_title("Fixed image")
    axes[0, 1].imshow(mv, cmap=cmap_img)
    axes[0, 1].set_title("Moving image")
    axes[0, 2].imshow(wp, cmap=cmap_img)
    axes[0, 2].set_title("Warped image")
    plot_deformed_grid(axes[0, 3], disp_slice = disp_field[..., slice_idx, :2], title="Deformation Field")
    axes[0, 3].set_title("Displacement grid")

    # --- Row 2: overlays ---
    def overlay(ax, img, lbl, lbl_num, title="", cmap_img="gray", alpha=0.8):
        """
        Overlay multi-label segmentation mask on top of a grayscale image.
        Each label is visualized with a distinct color, and unlabeled regions remain transparent.
        """
        # Show the base grayscale image
        ax.imshow(img, cmap=cmap_img)

        # Prepare color map (up to 20 distinct colors, extend if needed)
        cmap_lbl = plt.cm.get_cmap("gist_rainbow", lbl_num)

        # Convert labels to RGBA (each integer gets distinct color)
        lbl_rgba = cmap_lbl(lbl.astype(int))

        # Apply transparency: only labeled pixels are visible
        mask = lbl > 0
        lbl_rgba[..., -1] = mask.astype(float) * alpha  # alpha controls overlay intensity

        # Overlay the colored labels
        ax.imshow(lbl_rgba, interpolation="none")
        ax.set_title(title)
        ax.axis("off")

    overlay(axes[1, 0], fx, fxl, lbl_num, "Fixed + fixed labels")
    overlay(axes[1, 1], fx, mvl, lbl_num, "Fixed + moving labels")
    overlay(axes[1, 2], fx, wpl, lbl_num, "Fixed + warped labels")

    # --- Row 2 last cell: metrics ---
    axes[1, 3].axis("off")
    text = (
        f"Initial Dice: {init_dice:.3f}\n"
        f"Final Dice: {final_dice:.3f}\n"
        f"SDlogJ: {sdlogj:.3f}\n"
        f"Foldings: {fold_ratio:.2f}%"
    )
    axes[1, 3].text(
        0.5, 0.5, text,
        ha="center", va="center",
        fontsize=12, family="monospace",
        bbox=dict(facecolor="lightgray", edgecolor="black", boxstyle="round,pad=0.5")
    )

    for ax in axes.flat:
        ax.axis("off")

    plt.tight_layout()

    if save_plot:
        plt.savefig(output_path)
    else:
        plt.show()


def register_with_ants(fix_img_path, mov_img_path, mov_lbl_path, type_of_transform='Affine'):

    # Load images and labels
    fix_img_ants = ants.image_read(fix_img_path)
    mov_img_ants = ants.image_read(mov_img_path)
    mov_lbl_ants = ants.image_read(mov_lbl_path)

    # Run registration
    reg = ants.registration(fixed=fix_img_ants, moving=mov_img_ants, type_of_transform=type_of_transform)

    # Get warped image and labels
    warped_img_affine = reg['warpedmovout'].numpy()
    warped_lbl_affine = ants.apply_transforms(
        fixed=fix_img_ants,
        moving=mov_lbl_ants,
        transformlist=reg["fwdtransforms"],
        interpolator="nearestNeighbor"
    ).numpy()

    return warped_img_affine, warped_lbl_affine


def run_pca(features_1, features_2, img_shape, features_shape):

    D, H, W = img_shape[0], img_shape[1], img_shape[2]
    FD, FH, FW = features_shape[0], features_shape[1], features_shape[2]

    all_features = np.concatenate((features_1, features_2), axis=0)
    reduced_patches, eigenvalues = pca_lowrank_transform(all_features, 12)

    # Rehape features to feature map
    features_1_rescaled = reduced_patches[:FD * FH * FW, :]
    features_1_rescaled = features_1_rescaled.reshape([FD, FH, FW, -1])

    features_2_rescaled = reduced_patches[FD * FH * FW:, :]
    features_2_rescaled = features_2_rescaled.reshape([FD, FH, FW, -1])

    # Interpolate features to image dimension
    features_1_rescaled = features_1_rescaled.unsqueeze(0).permute(0, 4, 1, 2, 3).cuda()
    features_1_rescaled = torch.nn.functional.interpolate(features_1_rescaled, size=(D, H, W), mode="trilinear", align_corners=False)

    features_2_rescaled = features_2_rescaled.unsqueeze(0).permute(0, 4, 1, 2, 3).cuda()
    features_2_rescaled = torch.nn.functional.interpolate(features_2_rescaled, size=(D, H, W), mode="trilinear", align_corners=False)

    return features_1_rescaled, features_2_rescaled

# Paths to configuration files
CONFIGURATION = [
    '/home/eytan/projects/medical_ssl_3d/configs/evaluate_medssl.yaml',
    '/home/eytan/projects/medical_ssl_3d/configs/datasets/nako_evaluation_data.yaml'
    # '/home/eytan/projects/medical_ssl_3d/configs/datasets/abdomen_ctct.yaml'
]

# Parse configuration files and import project as a module
parser = ConfigParser()
parser.read_config(CONFIGURATION)
parser.parse()

project_path = parser.get("project")
import_module_from_path("project", project_path)

# Create output folder
save_artifacts = parser.get("save_artifacts")
output_dir = parser.get("output_dir")
if output_dir:
    os.makedirs(output_dir, exist_ok=True)

# Get method
method = parser.get("method")
if method == "medssl":
    features_dir = parser.get("features_dir")
    patch_size = parser.get("patch_size")
elif method == "dinov2":
    features_dir = parser.get("features_dir")
    feature_shape = parser.get("feature_shape")

# Get registration parameters
use_preregistration = parser.get("use_preregistration")

num_iter = parser.get("num_iter")
lr = parser.get("lr")
smooth_weight = parser.get("smooth_weight")
disp_hw = parser.get("disp_hw")
grid_sp = parser.get("grid_sp")
grid_sp_adam = parser.get("grid_sp_adam")
iter_smooth_kernel = parser.get("iter_smooth_kernel")
iter_smooth_num = parser.get("iter_smooth_num")
final_upsample = parser.get("final_upsample")
loss_func = parser.get("loss_func")
magnitute_scale = parser.get("magnitute_scale")

# Create placeholders for evaluation results
df_results = pd.DataFrame()
fixed_name_list = []
moving_name_list = []
sdjlog_list = []
fold_ratio_list = []
mean_initial_dice_list = []
mean_final_dice_list = []

organs_initial_dice_dict = {}
organs_final_dice_dict = {}
labels_file_path = parser.get("labels_file")
with open(labels_file_path, 'r') as f:
    data = json.load(f)
    labels = data['labels'][0].keys()

for label in labels:
    organs_initial_dice_dict[label + "_initial_dice"] = []
    organs_final_dice_dict[label + "_final_dice"] = []

# Get data loader and iterate over dataset
data_module = parser.get_parsed_content("data_module")
data_loader =data_module.val_dataloader()
for data_idx, data in enumerate(data_loader):

    # Save names of fixed and moving images
    fixed_path = data_module.val_dataset.data[data_idx]["fixed"]
    moving_path = data_module.val_dataset.data[data_idx]["moving"]
    moving_lbl_path = data_module.val_dataset.data[data_idx]["label_moving"]

    fixed_name_list.append(os.path.basename(fixed_path)[:-7])
    moving_name_list.append(os.path.basename(moving_path)[:-7])

    # Extract data
    fix_img = data['fixed']
    fix_lbl = data['label_fixed']
    fix_mask = extract_foreground_mask_mri(fix_img)
    # fix_mask = extract_foreground_mask_ct(fix_img)

    # Do ants "pre-registration" and substitute moving image and label with warped ones
    if use_preregistration:
        warped_img_ants, warped_lbl_ants = register_with_ants(fixed_path, moving_path, moving_lbl_path, type_of_transform='Affine')
        mov_img = torch.from_numpy(warped_img_ants).unsqueeze(0).unsqueeze(0)
        mov_lbl = torch.from_numpy(warped_lbl_ants).unsqueeze(0).unsqueeze(0)
    else:
        mov_img = data['moving']
        mov_lbl = data['label_moving']
    mov_mask = extract_foreground_mask_mri(mov_img)
    # mov_mask = extract_foreground_mask_ct(mov_img)

    # Derive number of labels
    lbl_num = int((mov_lbl.max() + 1).item())

    if method == "mind":  # Get MIND features

        mov_feature =  MINDSSC(mov_img.cuda(),3,2).detach().cpu().numpy()
        fix_feature = MINDSSC(fix_img.cuda(),3,2).detach().cpu().numpy()

        mov_feature = np.transpose(mov_feature.squeeze(0), (1, 2, 3, 0))
        fix_feature = np.transpose(fix_feature.squeeze(0), (1, 2, 3, 0))
    elif method == "medssl" or method == "dinov2":  # Get MEDSSL features

        if method == "medssl":
            mov_feature = np.load(os.path.join(features_dir, os.path.basename(moving_path)[:-7] + '_moving_features.npy'))[1:, :]
            fix_feature = np.load(os.path.join(features_dir, os.path.basename(fixed_path)[:-7] + '_fixed_features.npy'))[1:, :]
            features_shape = fix_img.shape[2] // patch_size[0], fix_img.shape[3] // patch_size[1], fix_img.shape[4] // patch_size[2]
        elif method == "dinov2":
            mov_feature = np.load(os.path.join(features_dir, os.path.basename(moving_path)[:-7] + '_moving_features.npy'))
            fix_feature = np.load(os.path.join(features_dir, os.path.basename(fixed_path)[:-7] + '_fixed_features.npy'))
            features_shape = feature_shape[0], feature_shape[1], fix_img.shape[4]

        fix_feature, mov_feature = run_pca(fix_feature, mov_feature, fix_img.shape[2:], features_shape)

        mov_feature = mov_feature.permute(0, 2, 3, 4, 1).squeeze().cpu().numpy()
        fix_feature = fix_feature.permute(0, 2, 3, 4, 1).squeeze().cpu().numpy()

    # Run ConvexAdam
    dice = dice_coeff(torch.tensor(mov_lbl).contiguous(), torch.tensor(fix_lbl).contiguous(), lbl_num)
    print(f"\nInitial dice: {dice}, {dice.mean().item():.4f}")

    # Save initial dice scores
    mean_initial_dice_list.append(dice.mean().item())
    for label_idx, label in enumerate(labels):
        organs_initial_dice_dict[label + "_initial_dice"].append(dice[label_idx])

    disp = convex_adam_3d_param(
        fix_feature * magnitute_scale,
        mov_feature * magnitute_scale,
        fix_lbl.cuda(),
        mov_lbl.cuda(),
        loss_func=loss_func,
        grid_sp=grid_sp,
        disp_hw=disp_hw,
        grid_sp_adam=grid_sp_adam,
        lambda_weight=smooth_weight,
        selected_niter=num_iter,
        lr=lr,
        disp_init=None,
        iter_smooth_kernel=iter_smooth_kernel,
        iter_smooth_num=iter_smooth_num,
        end_smooth_kernel=1,
        final_upsample=final_upsample
    )

    # Warp moving image, moving label and moving mask
    # Build grid
    D, H, W = mov_img.shape[2], mov_img.shape[3], mov_img.shape[4]

    zs = torch.linspace(0, D - 1, D)
    ys = torch.linspace(0, H - 1, H)
    xs = torch.linspace(0, W - 1, W)
    Z, Y, X = torch.meshgrid(zs, ys, xs, indexing='ij')

    displaced_x = X + disp[..., 2]
    displaced_y = Y + disp[..., 1]
    displaced_z = Z + disp[..., 0]

    # Normalize to [-1,1] along each axis
    nx = 2.0 * displaced_x / (W - 1) - 1.0
    ny = 2.0 * displaced_y / (H - 1) - 1.0
    nz = 2.0 * displaced_z / (D - 1) - 1.0

    grid = torch.stack((nx, ny, nz), dim=-1)  # (D,H,W,3)
    grid = grid.unsqueeze(0).float()  # (1,D,H,W,3)

    # Warp
    warped_img = torch.nn.functional.grid_sample(mov_img, grid, mode='bilinear')
    warped_lbl = torch.nn.functional.grid_sample(mov_lbl, grid, mode='nearest')
    warped_mask = torch.nn.functional.grid_sample(torch.from_numpy(mov_mask).unsqueeze(0).unsqueeze(0).float(), grid, mode='nearest')

    # Calculate smoothness metrics
    sdjlog, fold_ratio = jacobian_metrics(torch.tensor(disp).permute(3, 0, 1, 2).unsqueeze(0), fix_mask, warped_mask.byte().squeeze().numpy())
    print(f'SDJLog: {sdjlog}, FoldRatio: {fold_ratio}')

    # Save smoothness metrics
    sdjlog_list.append(sdjlog)
    fold_ratio_list.append(fold_ratio)

    # Save final dice scores
    dice = dice_coeff(torch.tensor(warped_lbl).contiguous(), torch.tensor(fix_lbl).contiguous(), lbl_num)
    print(f"Final dice: {dice}, {dice.mean().item():.4f}")

    mean_final_dice_list.append(dice.mean().item())
    for label_idx, label in enumerate(labels):
        organs_final_dice_dict[label + "_final_dice"].append(dice[label_idx])


    # Save or plot registration summary
    name = "fix" + os.path.basename(fixed_path)[:-7] + "_" + "mov" + os.path.basename(moving_path)[:-7]
    plot_registration_summary(
        fixed_img=fix_img.squeeze().cpu().numpy(),
        moving_img=mov_img.squeeze().cpu().numpy(),
        warped_img=warped_img.squeeze().cpu().numpy(),
        fixed_lbl=fix_lbl.squeeze().cpu().numpy(),
        moving_lbl=mov_lbl.squeeze().cpu().numpy(),
        warped_lbl=warped_lbl.squeeze().cpu().numpy(),
        disp_field=disp,
        init_dice=mean_initial_dice_list[data_idx],
        final_dice=mean_final_dice_list[data_idx],
        sdlogj=sdjlog_list[data_idx],
        fold_ratio=fold_ratio_list[data_idx],
        lbl_num=lbl_num,
        save_plot=save_artifacts,
        output_path=os.path.join(output_dir, name + "_summary.png")
    )

    # Save warped image and label, and displacement field
    if save_artifacts:
        np.save(os.path.join(output_dir, name + "_warped_img.npy"), warped_img.squeeze().cpu().numpy())
        np.save(os.path.join(output_dir, name + "_warped_lbl.npy"), warped_lbl.squeeze().cpu().numpy())
        np.save(os.path.join(output_dir, name + "_disp.npy"), disp)

# Calculate mean and std for metrics and save CSV file with results
fixed_name_list.append("mean")
fixed_name_list.append("std")

moving_name_list.append("")
moving_name_list.append("")

mean_initial_dice_list.append(np.mean(mean_initial_dice_list))
mean_initial_dice_list.append(np.std(mean_initial_dice_list[:-1]))

mean_final_dice_list.append(np.mean(mean_final_dice_list))
mean_final_dice_list.append(np.std(mean_final_dice_list[:-1]))

sdjlog_list.append(np.mean(sdjlog_list))
sdjlog_list.append(np.std(sdjlog_list[:-1]))

fold_ratio_list.append(np.mean(fold_ratio_list))
fold_ratio_list.append(np.std(fold_ratio_list[:-1]))

for label in labels:
    organs_initial_dice_dict[label + "_initial_dice"].append(np.mean(organs_initial_dice_dict[label + "_initial_dice"]))
    organs_initial_dice_dict[label + "_initial_dice"].append(np.std(organs_initial_dice_dict[label + "_initial_dice"][:-1]))

    organs_final_dice_dict[label + "_final_dice"].append(np.mean(organs_final_dice_dict[label + "_final_dice"]))
    organs_final_dice_dict[label + "_final_dice"].append(np.std(organs_final_dice_dict[label + "_final_dice"][:-1]))

if save_artifacts:
    metrics_dict = {
        "fixed": fixed_name_list,
        "moving": moving_name_list,
        "mean_initial_dice": mean_initial_dice_list,
        "mean_final_dice": mean_final_dice_list,
        "sdjlog": sdjlog_list,
        "fold_ratio": fold_ratio_list,
    }
    metrics_dict.update(organs_initial_dice_dict)
    metrics_dict.update(organs_final_dice_dict)
    df = pd.DataFrame(metrics_dict)
    df.to_csv(os.path.join(output_dir, "metrics.csv"))

print(f'\nRegistration results with {method} features:')
print(f'- initial dice: {mean_initial_dice_list[-2] * 100:.2f}({mean_initial_dice_list[-1] * 100:.2f})')
print(f'- final dice: {mean_final_dice_list[-2] * 100:.2f}({mean_final_dice_list[-1] * 100:.2f})')
print(f'- sdlogj: {sdjlog_list[-2]:.3f}({sdjlog_list[-1]:.3f})')
print(f'- fold_ratio: {fold_ratio_list[-2]:.2f}({fold_ratio_list[-1]:.2f})')