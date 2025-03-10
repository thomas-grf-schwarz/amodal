import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from transformers import (
    ViTMAEForPreTraining,
    ViTMAEConfig,
    ViTMAEModel,
    get_scheduler
    )
import einops
from functools import partial
from omegaconf import DictConfig
import hydra
from tqdm import tqdm


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


def get_anymae(config: ViTMAEConfig) -> AnyMAE:
    vit = ViTMAEModel(config)
    patchifier = AnyMAEPatchifier(config)
    unpatchifier = AnyMAEUnpatchifier(config)
    return AnyMAE(vit, patchifier, unpatchifier, config)


def make_vitmae_config(cfg: OmegaConf) -> ViTMAEConfig:
    
    config = ViTMAEConfig()

    config.num_channels = cfg.num_channels  # e.g., 1
    config.num_hidden_layers = cfg.num_hidden_layers  # e.g., 5
    config.num_attention_heads = cfg.num_attention_heads  # e.g., 32
    config.intermediate_size = cfg.intermediate_size  # e.g., 1024
    config.hidden_size = cfg.hidden_size  # e.g., 800
    config.mask_ratio = cfg.mask_ratio  # e.g., 0.75
    config.decoder_num_hidden_layers = cfg.decoder_num_hidden_layers  # e.g., 3
    config.decoder_hidden_size = cfg.decoder_hidden_size  # e.g., 512
    config.decoder_intermediate_size = cfg.decoder_intermediate_size  # e.g., 1024

    config.mean = cfg.mean  # e.g., -0.0042
    config.std = cfg.std  # e.g., 0.0137
    config.clip_at = cfg.clip_at  # e.g., 10

    config.lr_multiplier = cfg.lr_multiplier  # e.g., 2

    return config


import mlflow
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from transformers import get_scheduler
from tqdm import tqdm

def train_anymae(config, train_dataset, val_dataset):
    config = make_config_emg()
    train_dataset, val_dataset = get_emg_data(config)
    anymae = get_anymae(config)

    # Configuration
    batch_size = config.batch_size
    base_learning_rate = config.lr_multiplier * 4 * 1.5e-4 * (batch_size / 256)
    weight_decay = 0.0
    num_train_epochs = config.epochs
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

    # Optimizer
    optimizer = AdamW(anymae.parameters(), lr=base_learning_rate, weight_decay=weight_decay)

    # Learning rate scheduler
    scheduler = get_scheduler("cosine", optimizer, num_warmup_steps, num_training_steps)

    # GradScaler for mixed precision training
    scaler = GradScaler()

    # Training loop
    anymae = anymae.to(device)
    best_loss = float("inf")

    # Start MLflow experiment
    mlflow.start_run()
    mlflow.log_params({
        "batch_size": batch_size,
        "learning_rate": base_learning_rate,
        "weight_decay": weight_decay,
        "num_epochs": num_train_epochs,
        "warmup_ratio": warmup_ratio
    })

    for epoch in range(num_train_epochs):
        anymae.train()
        running_loss = 0.0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_train_epochs}")

        for step, (inputs, labels) in enumerate(progress_bar):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            with autocast():
                outputs = anymae(inputs)
                loss = outputs.loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

            # Log to MLflow
            mlflow.log_metric("train_loss", loss.item(), step=epoch * len(train_dataloader) + step)
            mlflow.log_metric("learning_rate", scheduler.get_last_lr()[0], step=epoch * len(train_dataloader) + step)

        # Scheduler step
        scheduler.step()

        # Validation
        if eval_strategy == "epoch":
            anymae.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, labels in val_dataloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    with autocast():
                        outputs = anymae(inputs)
                        val_loss += outputs.loss.item()

            avg_val_loss = val_loss / len(val_dataloader)
            print(f"Epoch {epoch+1}, Validation Loss: {avg_val_loss}")

            # Log validation loss to MLflow
            mlflow.log_metric("val_loss", avg_val_loss, step=epoch)

            # Save best model
            if save_strategy == "epoch" and avg_val_loss < best_loss:
                best_loss = avg_val_loss
                torch.save(anymae.state_dict(), "best_model.pth")
                mlflow.log_artifact("models/best_model.pth")

    mlflow.end_run()



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


@hydra.main(version_base=None, config_path="configs", config_name="mae")
def main_dist(cfg: DictConfig):

    get_data = hydra.utils.instantiate(cfg.get_data)
    change_input_config = hydra.utils.instantiate(cfg.change_input_config)
    
    cfg = make_vitmae_config(cfg)
    cfg = change_input_config(cfg)
    train_dataset, val_dataset = get_data(cfg)
    anymae = get_anymae(cfg)

    if cfg.convert_model_to_int8_mixed_precision_trainable:
        anymae = convert_model_to_int8_mixed_precision_trainable(anymae)

    anymae = train_anymae(anymae, train_dataset, val_dataset)
    import pdb; pdb.set_trace()


if __name__ == "__main__":
    main()


