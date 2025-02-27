import torch
from torch import nn
from transformers import (
    ViTMAEForPreTraining,
    ViTMAEConfig,
    ViTMAEModel,
    get_scheduler
    )
import einops
import collections
import pickle
from functools import partial
from data import (
    RAMDataset, 
    TransformedDataset, 
    transforms
    )


class PatchSize(collections.abc.Iterable):

    def __init__(self, sizes):
        self.sizes = sizes

    def __getitem__(self, idx):
        return self.sizes[idx]

    def __pow__(self, power):
        """
        handle non-square images
        """
        product = 1
        for size in self.sizes:
            product *= size
        return product

    def __iter__(self):
        return iter(self.sizes)


class AnyMAEPatchifier(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

    def forward(self, input_values):
        p1, p2 = self.config.patch_size
        c = self.config.num_channels
        assert (input_values.shape[2] % p1 == 0 and input_values.shape[3] % p2 == 0)

        h = input_values.shape[2] // p1  # number of patches vertically
        w = input_values.shape[3] // p2  # number of patches horizontally

        return einops.rearrange(
            input_values, 
            'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', 
            p1=p1, p2=p2, c=c
            )


class AnyMAEUnpatchifier(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

    def forward(self, patches):
        b, l, p1xp2xc = patches.shape
        p1, p2 = self.config.patch_size
        c = self.config.num_channels

        h = int(self.config.image_size[0] / p1)
        w = int(self.config.image_size[1] / p2)

        assert patches.shape[2] == p1 * p2
        assert patches.shape[1] == h * w

        return einops.rearrange(
            patches, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', 
            h=h, w=w, p1=p1, p2=p2, c=c
            )


class AnyMAE(ViTMAEForPreTraining):

    def __init__(self, vit, patchifier, unpatchifier, config):
        super().__init__(config)
        self.vit = vit
        self.patchifier = patchifier
        self.unpatchifier = unpatchifier
        torch.nn.init.normal_(
            self.vit.embeddings.position_embeddings.data, 
            std=self.vit.embeddings.config.initializer_range
        )
        self.vit.embeddings.position_embeddings.requires_grad = True

    def patchify(self, pixel_values, **kwargs):
        return self.patchifier(pixel_values)

    def unpatchify(self, pixel_values, **kwargs):
        return self.unpatchifier(pixel_values)


class AnyMAEForDownstream(nn.Module):

    def __init__(self, any_vit, head):
        super().__init__()
        self.any_vit = any_vit
        self.head = head
        self.any_vit.embeddings.config.mask_ratio = 0.0

    def forward(self, input_values):
        out = self.any_vit(input_values)
        return self.head(out.last_hidden_state)


def make_config(image_size: PatchSize, patch_size: PatchSize) -> ViTMAEConfig:
    config = ViTMAEConfig()
    config.image_size = image_size
    config.patch_size = patch_size
    return config


def get_anymae(config: ViTMAEConfig) -> AnyMAE:
    vit = ViTMAEModel(config)
    patchifier = AnyMAEPatchifier(config)
    unpatchifier = AnyMAEUnpatchifier(config)
    return AnyMAE(vit, patchifier, unpatchifier, config)


def make_config_emg() -> ViTMAEConfig:
    config = make_config(image_size=PatchSize([8, 256]), patch_size=PatchSize([1, 32]))
    config.num_channels = 1
    config.num_hidden_layers = 5
    config.num_attention_heads = 32
    config.intermediate_size = 1024
    config.hidden_size = 800
    config.mask_ratio = 0.75
    config.decoder_num_hidden_layers = 3
    config.decoder_hidden_size = 512
    config.decoder_intermediate_size = 1024

    config.mean = -0.0042
    config.std = 0.0137
    config.clip_at = 10

    config.lr_multiplier = 2

    return config

def get_emg_data():
    config = make_config_emg()
    transform_steps = [
        lambda x: torch.from_numpy(x), 
        lambda x: x[None, ...], 
        lambda x: (x - mean) / torch.sqrt(var + 1e-6),
        lambda x: torch.clip(x, -config.clip_at, config.clip_at),
    ]
    mean = torch.tensor(config.mean)
    var = torch.tensor(config.std)
    with open(config.train_tensor_dataset_path, "rb") as f:
        train_dataset = RAMDataset(
            TransformedDataset(
                dataset=pickle.load(f), 
                transform=partial(transforms, steps=transform_steps)
                )
            )
    with open(config.train_tensor_dataset_path, "rb") as f:
        val_dataset = RAMDataset(
            TransformedDataset(
                dataset=pickle.load(f), 
                transform=partial(transforms, steps=transform_steps)
                )
            )
    return train_dataset, val_dataset

from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
from tqdm import tqdm

def train_anymae(train_dataset, val_dataset):

    config = make_config_emg()
    train_dataset, val_dataset = get_emg_data(config)
    anymae = get_anymae(config)

    # Configuration
    batch_size = 32
    base_learning_rate = config.lr_multiplier * 4 * 1.5e-4 * (batch_size / 256)
    weight_decay = 0.0 # 0.005
    num_train_epochs = 800
    warmup_ratio = 0.05
    logging_steps = 10
    save_strategy = "epoch"
    eval_strategy = "epoch"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, num_workers=2, shuffle=True)
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, num_workers=2, shuffle=False)

    num_training_steps = num_train_epochs
    num_warmup_steps = warmup_ratio * num_training_steps

    inputs_overfit = None
    # inputs_overfit, _ = next(iter(train_dataloader))
    # inputs_overfit = inputs_overfit.to(device)
    train_losses = []
    val_losses = []
    learning_rates = []

    # Optimizer
    optimizer = AdamW(anymae.parameters(), lr=base_learning_rate, weight_decay=weight_decay)

    # Learning rate scheduler
    scheduler = get_scheduler("cosine", optimizer, num_warmup_steps, num_training_steps)

    # GradScaler for mixed precision training
    scaler = GradScaler()

    # Training loop
    anymae = anymae.to(device)
    best_loss = float("inf")

    for epoch in range(num_train_epochs):
        anymae.train()
        running_loss = 0.0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_train_epochs}")

        for step, (inputs, labels) in enumerate(progress_bar):
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass under autocast for mixed precision
            optimizer.zero_grad()
            with autocast():
                if inputs_overfit is not None:
                    outputs = anymae(inputs_overfit)
                else:
                    outputs = anymae(inputs)

                loss = outputs.loss       

            # Backward pass with GradScaler
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

            # Log every `logging_steps`
            if step % logging_steps == 0 and step != 0:
                print(f"Step {step}, Loss: {loss.item()}")
                train_losses.append(loss.item())
            learning_rates.append(scheduler.get_last_lr())

        # Scheduler step (epoch-based cosine annealing)
        scheduler.step()

        # Evaluation strategy (epoch-based)
        if eval_strategy == "epoch":
            anymae.eval()
            with torch.no_grad():
                val_loss = 0.0
                for inputs, labels in val_dataloader:
                    inputs, labels = inputs.to(device), labels.to(device)

                    # Validation forward pass under autocast
                    with autocast():
                        outputs = anymae(inputs)
                        val_loss += outputs.loss.item()

            avg_val_loss = val_loss / len(val_dataloader)
            print(f"Epoch {epoch+1}, Validation Loss: {avg_val_loss}")
            val_losses.append(avg_val_loss)

            # Save best model
            if save_strategy == "epoch" and avg_val_loss < best_loss:
                best_loss = avg_val_loss
                torch.save(anymae.state_dict(), "best_model.pth")


def convert_model_to_int8_mixed_precision_trainable(model):

    from torchao.prototype.quantized_training import (
        int8_mixed_precision_training, 
        Int8MixedPrecisionTrainingConfig
        )
    from torchao import quantize_

    # customize which matmul is left in original precision.
    config = Int8MixedPrecisionTrainingConfig(
        output=True,
        grad_input=True,
        grad_weight=False,
    )
    quantize_(model, int8_mixed_precision_training(config))

    return model



