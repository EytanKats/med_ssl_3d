import os
import sys
import torch
import einops
import pathlib
import numpy as np

from omegaconf import OmegaConf
from skimage.transform import resize
from monai.bundle import ConfigParser

import dinov2.utils.utils as dinov2_utils
from dinov2.models import build_model_from_cfg
from utils.imports import import_module_from_path


def load_config(config_name: str):
    config_filename = config_name + ".yaml"
    return OmegaConf.load(pathlib.Path(__file__).parent.resolve() / config_filename)


def load_and_merge_config(config_name: str):
    dinov2_default_config = load_config('{0:s}/configs/models/ssl_default_config'.format(os.path.dirname(sys.path[0])))
    default_config = OmegaConf.create(dinov2_default_config)
    loaded_config = load_config('{0:s}/configs/models/vitl14_reg4_pretrain'.format(os.path.dirname(sys.path[0])))
    return OmegaConf.merge(default_config, loaded_config)


def build_model_for_eval(config, pretrained_weights):
    model, _ = build_model_from_cfg(config, only_teacher=True)
    dinov2_utils.load_pretrained_weights(model, pretrained_weights, "teacher")
    model.eval()
    model.cuda()
    return model


def load_model():

    conf_fn = '{0:s}/configs/models/vitl14_reg4_pretrain'.format(os.path.dirname(sys.path[0]))
    model_fn = 'dinockpt/dinov2/dinov2_vitl14_reg4_pretrain.pth'
    model_url = 'https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_reg4_pretrain.pth'

    # Check if ckpt exists, download if not
    if not os.path.exists(model_fn):
        import urllib.request
        os.makedirs(os.path.dirname(model_fn), exist_ok=True)
        print(f"Downloading model from {model_url} to {model_fn}...")
        urllib.request.urlretrieve(model_url, model_fn)
        print("Download complete.")
    else:
        print("DINOv2 model found.")

    conf = load_and_merge_config(conf_fn)
    model = build_model_for_eval(conf, model_fn)
    model.cuda().eval()
    return model


def extract_encoder_feature(input_array, model):
    assert len(input_array.shape) == 3
    input_rgb_array = input_array[np.newaxis, :, :, :]
    input_tensor = torch.Tensor(np.transpose(input_rgb_array, [0, 3, 1, 2]))
    feature_array = model.forward_features(input_tensor.cuda())['x_norm_patchtokens'].detach().cpu().numpy()
    del input_tensor

    return feature_array


def remove_uniform_intensity_slices(image_data):
    slices_to_keep_indices = [i for i in range(image_data.shape[2])
                              if not np.max(image_data[:,:,i]) == np.min(image_data[:,:,i])]

    # Extract slices to keep
    filtered_image_data = image_data[:,:,slices_to_keep_indices]

    return filtered_image_data, slices_to_keep_indices


def case_preprocess(mov_arr, fix_arr):

    pad_indices = []
    filtered_image_data, slices_to_keep_indices = remove_uniform_intensity_slices(fix_arr)
    pad_indices.append(slices_to_keep_indices)
    fix_arr = filtered_image_data
    mov_arr = mov_arr[:, :, slices_to_keep_indices]
    orig_chunked_shape = fix_arr.shape

    return mov_arr, fix_arr, slices_to_keep_indices, orig_chunked_shape


def encode_3D_gap(
        input_arr,
        model,
        gap=6,
        feature_height=64,
        feature_width=64,
        patch_size=14,
        embed_dim=1024,
):
    imageH, imageW, slice_num = input_arr.shape

    resized_arr = resize(input_arr, (feature_height * patch_size, feature_width * patch_size, slice_num), anti_aliasing=True)

    print(f"patch size: {patch_size}")
    print(f"feature height: {feature_height}, feature width: {feature_width}, slice num: {slice_num}")
    print(f"original shape: {input_arr.shape}")
    print(f"resized shape: {resized_arr.shape}")

    # 3D image into 2D model, stack each slices feature
    img_feature = np.zeros([feature_height * feature_width, slice_num, embed_dim])
    encoding_slice_idx = np.arange(0, slice_num - 1, gap).tolist()
    encoding_slice_idx.append(slice_num - 1)

    prev_slice = 0
    for slice_id in encoding_slice_idx:
        input_slice = resized_arr[:, :, slice_id, np.newaxis]
        input_slice = np.repeat(input_slice, 3, axis=2)
        featrure = extract_encoder_feature(input_slice, model)
        featrure = einops.rearrange(featrure, '1 n c -> n c')
        print("\rslice id:{} feature shape:{} ".format(slice_id, featrure.shape), end="")
        img_feature[:, slice_id, :] = featrure

        # interpolating the feature of the skipped slices
        if slice_id > 0 and slice_id < slice_num - 1:
            for i in range(1, gap):
                slice_id_gap = slice_id - i
                if slice_id_gap >= 0:
                    featrure_gap = (featrure * (gap - i) + img_feature[:, prev_slice, :] * i) / gap
                    img_feature[:, slice_id_gap, :] = featrure_gap
        elif slice_id == slice_num - 1:
            last_gap = slice_num - encoding_slice_idx[-2]
            for i in range(1, last_gap):
                slice_id_gap = slice_num - i
                featrure_gap = (featrure * (last_gap - i) + img_feature[:, prev_slice, :] * i) / last_gap
                img_feature[:, slice_id_gap, :] = featrure_gap
        prev_slice = slice_id

    img_feature = img_feature.reshape([feature_height * feature_width * slice_num, embed_dim])

    return img_feature


# Paths to configuration files
CONFIGURATION = [
    '/home/eytan/projects/medical_ssl_3d/configs/evaluate_dinov2.yaml',
    '/home/eytan/projects/medical_ssl_3d/configs/datasets/nako_evaluation_data.yaml'
]

# Parse configuration files and import project as a module
parser = ConfigParser()
parser.read_config(CONFIGURATION)
parser.parse()

project_path = parser.get("project")
import_module_from_path("project", project_path)

# Create output folder
output_dir = parser.get("features_dir")
os.makedirs(output_dir, exist_ok=True)

# Get model
model = load_model()

# Get data loader and iterate over dataset
data_module = parser.get_parsed_content("data_module")
data_loader =data_module.val_dataloader()

for data_idx, data in enumerate(data_loader):

    # Get paths of fixed and moving images
    fixed_path = data_module.val_dataset.data[data_idx]["fixed"]
    moving_path = data_module.val_dataset.data[data_idx]["moving"]

    # Preprocess data
    fix_arr = data['fixed'].squeeze().numpy()
    mov_arr = data['moving'].squeeze().numpy()
    mov_arr, fix_arr, slices_to_keep_indices, orig_chunked_shape = case_preprocess(mov_arr, fix_arr)

    # Extract and save features
    mov_feature = encode_3D_gap(mov_arr, model)
    fix_feature = encode_3D_gap(fix_arr, model)

    np.save(os.path.join(output_dir, os.path.basename(fixed_path)[:-7] + "_" + "fixed_features.npy"), fix_feature)
    np.save(os.path.join(output_dir, os.path.basename(moving_path)[:-7] + "_" + "moving_features.npy"), mov_feature)
