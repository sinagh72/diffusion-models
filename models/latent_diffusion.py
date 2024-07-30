import lightning.pytorch as pl
import numpy as np
import torch.nn.functional as F
from generative.inferers import LatentDiffusionInferer
from generative.losses import PerceptualLoss, PatchAdversarialLoss
from generative.networks.nets import AutoencoderKL, PatchDiscriminator, DiffusionModelUNet, VQVAE
import torch
from generative.networks.nets.diffusion_model_unet import get_timestep_embedding
from generative.networks.schedulers import DDPMScheduler
from torchvision.transforms.v2 import ToPILImage, Resize, InterpolationMode
from PIL import Image


class LatentDiffusion(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.in_channels = kwargs["in_channels"]

        self.unet = DiffusionModelUNet(
            spatial_dims=kwargs["spatial_dims"],
            in_channels=self.in_channels,
            out_channels=kwargs["out_channels"],
            num_res_blocks=kwargs["num_res_blocks"],
            num_channels=kwargs["num_channels"],
            attention_levels=kwargs["attention_levels"],
            num_head_channels=kwargs["num_head_channels"],
            # with_conditioning=kwargs["with_conditioning"],
            # cross_attention_dim=kwargs["cross_attention_dim"]
        )

        scheduler = DDPMScheduler(num_train_timesteps=kwargs["num_train_timesteps"],
                                  schedule="linear_beta",
                                  beta_start=kwargs["beta_start"],
                                  beta_end=kwargs["beta_end"])

        self.lr = kwargs["lr"]
        self.autoencoder = kwargs["autoencoder"].to(self.device)
        self.autoencoder.eval()

        # self.embed = torch.nn.Embedding(num_embeddings=3, embedding_dim=kwargs["embedding_dimension"], padding_idx=0)
        # self.condition_dropout = kwargs["condition_dropout"]

        with torch.no_grad():
            z = self.autoencoder.encode_stage_2_inputs(kwargs["sample"].to(self.device))
        scale_factor = 1 / torch.std(z)
        self.inferer = LatentDiffusionInferer(scheduler, scale_factor=scale_factor)

        self.metrics = {"train": {"loss": []}, "val": {"loss": []}}
        self.save_fig_path = kwargs["save_fig_path"]
        self.to_pil = ToPILImage('L')
        self.img_size = kwargs["img_size"]
        self.generate = kwargs["generate"]
        self.stack_imgs = kwargs["stack_imgs"]
        self.ae_type = "vae"

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer

    def forward(self, x, classes):
        if isinstance(self.autoencoder, VQVAE):
            z = self.autoencoder.encode(x)
        elif isinstance(self.autoencoder, AutoencoderKL):
            z_mu, z_sigma = self.autoencoder.encode(x)
            z = self.autoencoder.sampling(z_mu, z_sigma)

        noise = torch.randn_like(z).to(self.device)
        timesteps = torch.randint(0, self.inferer.scheduler.num_train_timesteps, (z.shape[0],), device=z.device).long()
        noise_pred = self.inferer(
            inputs=x,
            diffusion_model=self.unet,
            noise=noise,
            timesteps=timesteps,
            autoencoder_model=self.autoencoder,
        )
        return noise, noise_pred

    def training_step(self, batch, batch_idx):
        images, classes = batch
        x, x_pred = self(images, classes)  # Forward pass

        """"""
        loss = F.mse_loss(x_pred.float(), x.float())
        self.metrics["train"]["loss"].append(loss.item())
        return loss

    def on_train_epoch_end(self):
        self.stack_update(session="train")

    def validation_step(self, batch, batch_idx):
        images, classes = batch
        x, x_pred = self(images, classes)  # Forward pass
        loss = F.mse_loss(x_pred.float(), x.float())
        self.metrics["val"]["loss"].append(loss.item())
        return loss

    def on_validation_epoch_end(self):
        self.stack_update(session="val")
        if self.generate:
            self.generate_sample(epoch=self.current_epoch, stack_imgs=self.stack_imgs)

    def stack_update(self, session):
        log = {}
        for key in self.metrics[session]:
            log[session + "_" + key] = np.stack(self.metrics[session][key]).mean()
            self.metrics[session][key].clear()
        self.log_dict(log, sync_dist=True, on_epoch=True, prog_bar=True, logger=True)

    def generate_sample(self, epoch, stack_imgs, resize=None):
        if resize is not None:
            resize = Resize((resize, resize), InterpolationMode.BICUBIC)
        else:
            resize = Resize((self.img_size, self.img_size), InterpolationMode.BICUBIC)

        img_size = self.img_size // 4
        pil_images = []
        for i in range(stack_imgs):
            noise = torch.randn((1, self.in_channels, img_size, img_size)).to(self.device)
            self.inferer.scheduler.set_timesteps(num_inference_steps=1000)
            decoded = self.inferer.sample(
                input_noise=noise, diffusion_model=self.unet, scheduler=self.inferer.scheduler,
                autoencoder_model=self.autoencoder
            )
            img = (decoded + 1) / 2 * 255
            pil_images.append(img.to(torch.uint8))
        # Convert each tensor image slice to a PIL image and collect them in a list
        pil_images = [resize(self.to_pil(j.squeeze(0))) for j in pil_images]  # squeeze to remove channel dimension

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

    def extract_features(self, x, timesteps, context=None, class_labels=None):
        t_emb = get_timestep_embedding(timesteps, self.unet.block_out_channels[0])
        t_emb = t_emb.to(dtype=x.dtype)
        emb = self.unet.time_embed(t_emb)

        if self.unet.num_class_embeds is not None:
            if class_labels is None:
                raise ValueError("class_labels should be provided when num_class_embeds > 0")
            class_emb = self.unet.class_embedding(class_labels)
            class_emb = class_emb.to(dtype=x.dtype)
            emb = emb + class_emb

        h = self.unet.conv_in(x)

        down_block_res_samples = [h]
        for downsample_block in self.unet.down_blocks:
            h, res_samples = downsample_block(hidden_states=h, temb=emb, context=context)
            for residual in res_samples:
                down_block_res_samples.append(residual)

        h = self.unet.middle_block(hidden_states=h, temb=emb, context=context)

        return h  # Extracted features from the middle block
