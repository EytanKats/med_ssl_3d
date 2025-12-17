"""
Script to run 3D DINOv2 training or prediction using a YAML config.
- Uses PyTorch Lightning, MONAI, and custom project modules.
- Accepts config file and overrides via CLI (with fire).
"""

import sys
sys.path.append(".")

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

from torch.utils.data import DataLoader
from monai.bundle import ConfigParser

import fire
from typing import Union, List
from utils.imports import import_module_from_path
import torch
import monai

torch.serialization.safe_globals([monai.data.meta_tensor.MetaTensor])


def run(mode, config_file: Union[str, List[str]], **config_overrides):
    """
    Run training or prediction based on the mode parameter.

    Args:
        config_file (str): Path to the configuration file (YAML)
        mode (str): Either "train" or "predict"
        **config_overrides: Additional configuration overrides (key=value)
    """

    assert mode in ["fit", "predict"], "Unsupported mode"

    parser = ConfigParser()
    parser.read_config(config_file)
    parser.parse()
    parser.update(config_overrides)

    project_path = parser.get("project")
    import_module_from_path("project", project_path)

    trainer = parser.get_parsed_content("trainer")
    lightning_module = parser.get_parsed_content("lightning_module")
    data_module = parser.get_parsed_content("data_module")

    getattr(trainer, mode)(lightning_module, data_module)

if __name__ == "__main__":
    # Set environment variables for better performance
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Run pretraining
    run(
        'fit',
        config_file=[
        '/home/eytan/projects/medical_ssl_3d/configs/train.yaml',
        '/home/eytan/projects/medical_ssl_3d/configs/models/primus.yaml',
        # '/home/eytan/projects/medical_ssl_3d/configs/datasets/nako_128_fg70.yaml',
        '/home/eytan/projects/medical_ssl_3d/configs/datasets/nako_ts_128.yaml',
        '/home/eytan/projects/medical_ssl_3d/configs/datasets/nako30_pairs.yaml',
        # '/home/eytan/projects/medical_ssl_3d/configs/datasets/abdomen_ctct.yaml',
        # '/home/eytan/projects/medical_ssl_3d/configs/datasets/popi.yaml',
        ]
    )

    # Run segmentation
    # run(
    #     'fit',
    #     config_file=[
    #         '/home/eytan/projects/medical_ssl_3d/configs/linear_evaluation.yaml',
    #         '/home/eytan/projects/medical_ssl_3d/configs/datasets/nako30_seg.yaml',
    #         '/home/eytan/projects/medical_ssl_3d/configs/datasets/nako30_seg_val.yaml',
    #     ]
    # )

    # Run training or prediction
    # fire.Fire(run)
