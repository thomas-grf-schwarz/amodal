import torch
import pickle
from data import *


def get_data_emg(
    train_tensor_dataset_path,
    val_tensor_dataset_path,
    mean,
    var,
    clip_at
    ):

    mean = torch.tensor(mean)
    var = torch.tensor(var)
    transform_steps = [
        lambda x: torch.from_numpy(x), 
        lambda x: x[None, ...], 
        lambda x: (x - torch.tensor(mean)) / torch.sqrt(var + 1e-6),
        lambda x: torch.clip(x, -clip_at, clip_at),
    ]
    with open(train_tensor_dataset_path, "rb") as f:
        train_dataset = RAMDataset(
            TransformedDataset(
                dataset=pickle.load(f), 
                transform=partial(transforms, steps=transform_steps)
                )
            )
    with open(val_tensor_dataset_path, "rb") as f:
        val_dataset = RAMDataset(
            TransformedDataset(
                dataset=pickle.load(f), 
                transform=partial(transforms, steps=transform_steps)
                )
            )
    return train_dataset, val_dataset


def change_input_config_emg(config):
    config.image_size = PatchSize(cfg.image_size)
    config.patch_size = PatchSize(cfg.patch_size)
    return config