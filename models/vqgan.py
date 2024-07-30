import random

import lightning.pytorch as pl
import numpy as np
import torch.nn.functional as F
from generative.losses import PerceptualLoss, PatchAdversarialLoss
from generative.networks.nets import PatchDiscriminator, VQVAE
import torch
from PIL import Image
from torchvision.transforms import ToPILImage, v2


class VQGAN(pl.LightningModule):
    def __init__(self, **kwargs):

        super().__init__()
        self.vqvae = VQVAE(
            spatial_dims=kwargs["spatial_dims"],
            in_channels=kwargs["in_channels"],
            out_channels=kwargs["out_channels"],
            num_res_layers=kwargs["num_res_layers"],
            downsample_parameters=kwargs["downsample_parameters"],
            upsample_parameters=kwargs["upsample_parameters"],
            num_channels=kwargs["num_channels"],
            num_res_channels=kwargs["num_res_channels"],
            num_embeddings=kwargs["num_embeddings"],
            embedding_dim=kwargs["embedding_dim"],
        )
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

        self.metrics = {"train": {"loss_g": [], "loss_d": [], "recons_loss": []}, "val": {"val_loss": []}}
        self.automatic_optimization = False
        self.generator_warmup = kwargs["generator_warmup"]

        self.img_size = kwargs["img_size"]
        self.save_fig_path = kwargs["save_fig_path"]
        self.intermediary_images = []

    def configure_optimizers(self):
        optimizer_g = torch.optim.AdamW(self.vqvae.parameters(), lr=self.lr_g)
        optimizer_d = torch.optim.AdamW(self.discriminator.parameters(), lr=self.lr_d)
        return [optimizer_g, optimizer_d], []

    def forward(self, x):
        reconstruction, quantization_loss = self.vqvae(images=x)
        return reconstruction, quantization_loss

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

    def on_train_epoch_end(self):
        self.stack_update(session="train")

    def validation_step(self, batch, batch_idx):
        images, _ = batch
        reconstruction, quantization_loss = self(images)  # Forward pass
        # Get the first reconstruction from the first validation batch for visualisation purposes
        self.intermediary_images.append(reconstruction[:10, 0])
        reconstruction, quantization_loss = self(images)  # Forward pass
        val_loss = F.l1_loss(reconstruction.float(), images.float())
        self.metrics["val"]["val_loss"].append(val_loss.item())
        return val_loss

    def on_validation_epoch_end(self):
        self.stack_update(session="val")
        self.log_img()

    def stack_update(self, session):
        log = {}
        for key in self.metrics[session]:
            if len(self.metrics[session][key]) > 1:
                log[key] = np.stack(self.metrics[session][key]).mean()
                self.metrics[session][key].clear()
        self.log_dict(log, sync_dist=True, on_epoch=True, prog_bar=True, logger=True)

    def log_img(self):
        to_pil = ToPILImage('L')

        for i, img in enumerate(self.intermediary_images):
            img = (img + 1) / 2 * 255
            img = img.to(torch.uint8)

            # Convert each tensor image slice to a PIL image and collect them in a list
            pil_images = [to_pil(img[j].unsqueeze(0)) for j in
                          range(img.shape[0])]  # Unsqueeze to add channel dimension

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
            concatenated_image.save(f"{self.save_fig_path}/sample{i}_e{self.current_epoch}.png")
        self.intermediary_images.clear()
