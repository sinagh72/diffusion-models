import random

import lightning.pytorch as pl
import numpy as np
import torch.nn.functional as F
from generative.losses import PerceptualLoss, PatchAdversarialLoss
from generative.networks.nets import PatchDiscriminator, VQVAE
import torch
from PIL import Image
from torchvision.transforms import ToPILImage, v2


class VQA(pl.LightningModule):
    def __init__(self, **kwargs):

        super().__init__()
        self.net = VQVAE(
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

        self.lr = kwargs["lr"]
        self.metrics = {"train": {"loss": [], "recons_loss": []}, "val": {"val_loss": []}}

        self.img_size = kwargs["img_size"]
        self.save_fig_path = kwargs["save_fig_path"]
        self.intermediary_images = []

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.lr)
        return optimizer

    def forward(self, x):
        reconstruction, quantization_loss = self.net(images=x)
        return reconstruction, quantization_loss

    def training_step(self, batch, batch_idx):
        images, labels = batch
        reconstruction, quantization_loss = self(images)  # Forward pass
        recons_loss = F.l1_loss(reconstruction.float(), images.float())
        self.metrics["train"]["recons_loss"].append(recons_loss.item())
        loss = recons_loss + quantization_loss
        return loss

    def on_train_epoch_end(self):
        self.stack_update(session="train")

    def validation_step(self, batch, batch_idx):
        images, _ = batch
        reconstruction, quantization_loss = self(images)  # Forward pass
        chosen = np.random.randint(len(reconstruction[0]))
        # Get the first reconstruction from the first validation batch for visualisation purposes
        self.intermediary_images.append(reconstruction[chosen, 0])
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

    def log_img(self, num=10):
        to_pil = ToPILImage('L')
        num = len(self.intermediary_images) if num > len(self.intermediary_images) else num

        for i, img in enumerate(self.intermediary_images[:num]):
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
