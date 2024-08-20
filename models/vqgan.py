import random

import lightning.pytorch as pl
import numpy as np
import torch.nn.functional as F
from generative.losses import PerceptualLoss, PatchAdversarialLoss
from generative.networks.nets import PatchDiscriminator, VQVAE
import torch
from PIL import Image
from torchvision.transforms import ToPILImage, v2
from models.vqa import VQA


class VQGAN(VQA):
    def __init__(self, **kwargs):
        kwargs["lr"] = 0
        super().__init__(**kwargs)
        self.perceptual_loss = PerceptualLoss(spatial_dims=kwargs["spatial_dims"], network_type="alex")
        self.perceptual_weight = kwargs["perceptual_weight"]

        self.discriminator = PatchDiscriminator(spatial_dims=kwargs["spatial_dims"],
                                                num_layers_d=kwargs["latent_channels"],
                                                num_channels=kwargs["path_discriminator_num_channels"],
                                                in_channels=kwargs["in_channels"],
                                                )
        self.adv_loss = PatchAdversarialLoss(criterion="least_squares")
        self.adv_weight = kwargs["adv_weight"]
        self.lr_g = kwargs["lr_g"]
        self.lr_d = kwargs["lr_d"]

        self.automatic_optimization = False
        self.generator_warmup = kwargs["generator_warmup"]

        self.metrics["train"] = {"loss_g": [], "loss_d": [], "recons_loss": []}

    def configure_optimizers(self):
        optimizer_g = torch.optim.AdamW(self.net.parameters(), lr=self.lr_g)
        optimizer_d = torch.optim.AdamW(self.discriminator.parameters(), lr=self.lr_d)
        return [optimizer_g, optimizer_d], []

    def training_step(self, batch, batch_idx):
        images, labels = batch
        optimizer_g, optimizer_d = self.optimizers()

        self.toggle_optimizer(optimizer_g)
        optimizer_g.zero_grad(set_to_none=True)
        reconstruction, quantization_loss = self(images)  # Forward pass
        logits_fake = self.discriminator(reconstruction.contiguous().float())[-1]
        recons_loss = F.l1_loss(reconstruction.float(), images.float())
        self.metrics["train"]["recons_loss"].append(recons_loss.item())
        generator_loss = self.adv_loss(logits_fake, target_is_real=True, for_discriminator=False)
        p_loss = self.perceptual_loss(reconstruction.float(), images.float())
        loss_g = recons_loss + quantization_loss + (self.perceptual_weight * p_loss) + self.adv_weight * generator_loss

        self.manual_backward(loss_g)
        optimizer_g.step()
        self.metrics["train"]["loss_g"].append(loss_g.item())
        self.untoggle_optimizer(optimizer_g)

        if self.current_epoch > self.generator_warmup:
            self.toggle_optimizer(optimizer_d)
            optimizer_d.zero_grad(set_to_none=True)
            logits_fake = self.discriminator(reconstruction.contiguous().detach())[-1]
            loss_d_fake = self.adv_loss(logits_fake, target_is_real=False, for_discriminator=True)
            logits_real = self.discriminator(images.contiguous().detach())[-1]
            loss_d_real = self.adv_loss(logits_real, target_is_real=True, for_discriminator=True)
            discriminator_loss = (loss_d_fake + loss_d_real) * 0.5
            loss_d = self.adv_weight * discriminator_loss
            self.manual_backward(loss_d)
            optimizer_d.step()
            self.metrics["train"]["loss_d"].append(loss_d.item())
            self.untoggle_optimizer(optimizer_d)

    def validation_step(self, batch, batch_idx):
        images, _ = batch
        reconstruction, quantization_loss = self(images)  # Forward pass
        # Get the first reconstruction from the first validation batch for visualisation purposes
        self.intermediary_images.append(reconstruction[:10, 0])
        reconstruction, quantization_loss = self(images)  # Forward pass
        val_loss = F.l1_loss(reconstruction.float(), images.float())
        self.metrics["val"]["val_loss"].append(val_loss.item())
        return val_loss
