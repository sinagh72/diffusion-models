import lightning.pytorch as pl
import numpy as np
import torch.nn.functional as F
from generative.inferers import DiffusionInferer
import torch
from generative.networks.nets.diffusion_model_unet import DiffusionModelEncoder
from generative.networks.schedulers import DDPMScheduler, DDIMScheduler
from torchvision.transforms.v2 import ToPILImage, Resize, InterpolationMode
from PIL import Image


class DiffusionEncoder(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.encoder = DiffusionModelEncoder(
            spatial_dims=kwargs["spatial_dims"],
            in_channels=kwargs["in_channels"],
            out_channels=kwargs["out_channels"],
            num_res_blocks=kwargs["num_res_blocks"],
            num_channels=kwargs["num_channels"],
            attention_levels=kwargs["attention_levels"],
            num_head_channels=kwargs["num_head_channels"],

        )

        self.scheduler = DDIMScheduler(num_train_timesteps=kwargs["num_train_timesteps"], schedule="cosine")
        self.lr = kwargs["lr"]
        self.metrics = {"train": {"loss": []}, "val": {"loss": []}}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.encoder.parameters(), lr=self.lr)
        return optimizer

    def forward(self, x):
        noise = torch.randn_like(x).to(self.device)
        timesteps = torch.randint(0, self.scheduler.num_train_timesteps, (noise.shape[0],),
                                  device=noise.device).long()
        noisy_x = self.scheduler.add_noise(x, noise, timesteps)  # add t steps of noise to the input image
        pred = self.encoder(noisy_x, timesteps)
        return pred

    def training_step(self, batch, batch_idx):
        images, labels = batch
        pred = self(images)  # Forward pass
        """"""
        loss = F.cross_entropy(pred.float(), labels)
        self.metrics["train"]["loss"].append(loss.item())
        return loss

    def on_train_epoch_end(self):
        self.stack_update(session="train")

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        pred = self(images)  # Forward pass
        """"""
        loss = F.cross_entropy(pred.float(), labels)
        self.metrics["val"]["loss"].append(loss.item())
        return loss

    def on_validation_epoch_end(self):
        self.stack_update(session="val")
    def stack_update(self, session):
        log = {}
        for key in self.metrics[session]:
            log[session + "_" + key] = np.stack(self.metrics[session][key]).mean()
            self.metrics[session][key].clear()
        self.log_dict(log, sync_dist=True, on_epoch=True, prog_bar=True, logger=True)

