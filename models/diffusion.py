import lightning.pytorch as pl
import numpy as np
import torch.nn.functional as F
import torchvision
from generative.inferers import DiffusionInferer
from generative.networks.nets import DiffusionModelUNet
import torch
from generative.networks.schedulers import DDPMScheduler, DDIMScheduler
from matplotlib import pyplot as plt
from torchvision.transforms.v2 import ToPILImage, Resize, InterpolationMode
from PIL import Image


class Diffusion(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.unet = DiffusionModelUNet(
            spatial_dims=kwargs["spatial_dims"],
            in_channels=kwargs["in_channels"],
            out_channels=kwargs["out_channels"],
            num_res_blocks=kwargs["num_res_blocks"],
            num_channels=kwargs["num_channels"],
            attention_levels=kwargs["attention_levels"],
            num_head_channels=kwargs["num_head_channels"],
            cross_attention_dim=kwargs["cross_attention_dim"],
            with_conditioning=kwargs["with_conditioning"]
        )

        self.scheduler = DDIMScheduler(num_train_timesteps=kwargs["num_train_timesteps"], schedule="cosine")
        self.lr = kwargs["lr"]
        self.inferer = DiffusionInferer(self.scheduler)
        self.metrics = {"train": {"loss": []}, "val": {"loss": []}}
        self.save_fig_path = kwargs["save_fig_path"]
        self.to_pil = ToPILImage('L')
        self.img_size = kwargs["img_size"]
        self.generate = kwargs["generate"]

        if "semantic_encoder" in kwargs:
            self.semantic_encoder = kwargs["semantic_encoder"]
        else:
            self.semantic_encoder = torchvision.models.resnet18()
            self.semantic_encoder.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.semantic_encoder.fc = torch.nn.Linear(512, kwargs["embedding_dim"])

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer

    def forward(self, x):
        noise = torch.randn_like(x).to(self.device)
        timesteps = torch.randint(0, self.inferer.scheduler.num_train_timesteps, (noise.shape[0],),
                                  device=noise.device).long()
        latent = self.semantic_encoder(x)
        noise_pred = self.inferer(inputs=x, diffusion_model=self.unet, noise=noise, timesteps=timesteps,
                                  condition=latent.unsqueeze(2))
        return noise, noise_pred

    def training_step(self, batch, batch_idx):
        images, _ = batch
        noise, noise_pred = self.forward(images)  # Forward pass
        """"""
        loss = F.mse_loss(noise_pred.float(), noise.float())
        self.metrics["train"]["loss"].append(loss.item())
        return loss

    def on_train_epoch_end(self):
        self.stack_update(session="train")

    def validation_step(self, batch, batch_idx):
        images, _ = batch
        noise, noise_pred = self.forward(images)  # Forward pass
        """"""
        loss = F.mse_loss(noise_pred.float(), noise.float())
        self.metrics["val"]["loss"].append(loss.item())
        return loss

    def on_validation_epoch_end(self):
        self.stack_update(session="val")
        if self.generate and self.current_epoch > 0:
            self.generate_sample(epoch=self.current_epoch, stack_imgs=10)

    def stack_update(self, session):
        log = {}
        for key in self.metrics[session]:
            log[session + "_" + key] = np.stack(self.metrics[session][key]).mean()
            self.metrics[session][key].clear()
        self.log_dict(log, sync_dist=True, on_epoch=True, prog_bar=True, logger=True)

    def generate_sample(self, epoch, stack_imgs, resize=None, channles=1):

        img_size = self.img_size // 4
        self.semantic_encoder.eval()
        with torch.no_grad():
            noise = torch.randn((1, channles, img_size, img_size)).to(self.device)
            self.scheduler.set_timesteps(num_inference_steps=1000)
            latent = self.semantic_encoder(noise)
            decoded = self.inferer.sample(
                input_noise=noise, diffusion_model=self.unet, scheduler=self.scheduler,
                save_intermediates=False, conditioning=latent.unsqueeze(2)
            )
            grid = torchvision.utils.make_grid(torch.cat([noise[:4], noise[:4], decoded[:4]]), nrow=4,
                                               padding=2, normalize=True, scale_each=False, pad_value=0)
            plt.figure(figsize=(15, 5))
            plt.imshow(grid.detach().cpu().numpy()[0], cmap='gray')
            plt.axis('off')

        self.semantic_encoder.train()
