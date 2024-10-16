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
from lightning.pytorch.utilities import rank_zero_only
from generative.losses import PerceptualLoss
from numpy import random
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
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
        
        # self.scheduler_ddim = DDIMScheduler(
        #     num_train_timesteps=kwargs["num_train_timesteps"]//4, schedule="scaled_linear_beta", beta_start=0.0005, beta_end=0.0195, clip_sample=False
        # )
        # self.scheduler_ddpm =  DDPMScheduler(num_train_timesteps=kwargs["num_train_timesteps"], schedule="scaled_linear_beta", beta_start=0.0005, beta_end=0.0195)

        self.scheduler_ddpm =  DDPMScheduler(num_train_timesteps=kwargs["num_train_timesteps"])
        
        self.lr = kwargs["lr"]
        self.inferer = DiffusionInferer(self.scheduler_ddpm)
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
        
        self.sample_imgs = []
        self.perceptual_loss = PerceptualLoss(spatial_dims=kwargs["spatial_dims"], network_type="resnet50")
        self.perceptual_weight = 0.001

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def noise_timestep(self, x):
        # noise = torch.clamp(torch.randn_like(x).to(self.device),  min=-1, max=1)
        noise = torch.randn_like(x).to(self.device)
        timesteps = torch.randint(0, self.inferer.scheduler.num_train_timesteps, (noise.shape[0],),
                                  device=noise.device).long()
        return noise, timesteps

    def forward(self, x):
        # latent = self.semantic_encoder(x.repeat(1, 3, 1, 1))
        latent = self.semantic_encoder(x)
        noise, timesteps = self.noise_timestep(x)
        noise_pred = self.inferer(inputs=x, diffusion_model=self.unet, noise=noise, timesteps=timesteps,
                                  condition=latent.unsqueeze(2))
        pred_list = []
        for i in range(x.shape[0]):
            x_pred, _ = self.scheduler_ddpm.step(noise_pred[i], int(timesteps[i]), x[i])
            pred_list.append(x_pred)
        pred_list = torch.stack(pred_list, dim=0)
        p_loss = self.perceptual_loss(pred_list.float().clone(), x.float().clone())
        # print(p_loss)
        # print(p_loss)
        # x_single = ((x[0][0] + 0.5) *0.5)
        # x_np = x_single.detach().cpu().numpy()
        # print(x_np.shape)
        # plt.imshow(x_np)
        # plt.axis('off')
        # plt.savefig("original_"+str(self.current_epoch) + "_"+str(self.global_step))


        # x_single = ((x_pred[0] +0.5) *0.5)
        # x_np = x_single.detach().cpu().numpy()
        # plt.imshow(x_np)
        # plt.axis('off')
        # plt.savefig("pred_"+str(self.current_epoch) + "_"+str(self.global_step))


        
        # p_loss = self.perceptual_loss(latent.float(), x.float())
        return noise, noise_pred, p_loss

    def training_step(self, batch, batch_idx):
        images, _ = batch
        noise, noise_pred, p_loss = self.forward(images)  # Forward pass
        """"""
        loss = F.mse_loss(noise_pred.float(), noise.float()) + self.perceptual_weight * p_loss
        self.metrics["train"]["loss"].append(loss.item())
        return loss

    def on_train_epoch_end(self):
        self.stack_update(session="train")

    def validation_step(self, batch, batch_idx):
        images, _ = batch
        noise, noise_pred, p_loss = self.forward(images)  # Forward pass
        """"""
        loss = F.mse_loss(noise_pred.float(), noise.float()) + self.perceptual_weight * p_loss
        self.metrics["val"]["loss"].append(loss.item())
        """"""
        if len(self.sample_imgs) < 10:
            for img in images:
                self.sample_imgs.append(img)
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

    @rank_zero_only
    def generate_sample(self, epoch, stack_imgs, resize=None):
        if resize is not None:
            resize = Resize((resize, resize), InterpolationMode.BICUBIC)
            img_size = resize
        else:
            resize = Resize((self.img_size, self.img_size), InterpolationMode.BICUBIC)
            img_size = self.img_size

        pil_images = []
        for i in range(stack_imgs):
            self.scheduler_ddpm.set_timesteps(num_inference_steps=1000)
            noise = torch.randn((1, 1, img_size, img_size)).to(self.device)
            # noise = torch.clamp(torch.randn((1, 3, img_size, img_size)).to(self.device),  min=-1, max=1)
            # noise = torch.randn((1, 3, img_size, img_size)).to(self.device)
            # latent = self.semantic_encoder(noise.repeat(1, 3, 1, 1))
            random_idx = random.randint(0, len(self.sample_imgs))
            latent = self.semantic_encoder(self.sample_imgs[random_idx].unsqueeze(0)).unsqueeze(2)
            decoded = self.inferer.sample(
                input_noise=noise, diffusion_model=self.unet, scheduler=self.scheduler_ddpm,
                save_intermediates=False, conditioning=latent
            )
            img = ((decoded * 0.5) + 0.5) * 255
            pil_images.append(img.squeeze(0)[0].to(torch.uint8))
        # Convert each tensor image slice to a PIL image and collect them in a list
        pil_images = [resize(self.to_pil(j)) for j in pil_images]  # squeeze to remove channel dimension

        # Calculate total width and maximum height for the concatenated image
        total_width = sum(image.size[0] for image in pil_images)
        max_height = max(image.size[1] for image in pil_images)
        # Create a new image with the calculated dimensions
        concatenated_image = Image.new('L', (total_width, max_height))
        # Concatenate images horizontally
        x_offset = 0
        for image in pil_images:
            concatenated_image.paste(image, (x_offset, 0))
            x_offset += image.size[0]
        # Save the concatenated image
        concatenated_image.save(self.save_fig_path + f"/sample_{epoch}.png")
        self.sample_imgs.clear()