import glob
import os
from monai.utils import set_determinism
from PIL import Image

from dataset.datamodule_handler import get_data_modules
from models.latent_diffusion import Diffusion
from models.vae import VAE
from transforms.apply_transforms import get_train_transformation, get_test_transformation
from utils.labels import get_full_classes
from utils.utils import set_seed
import torch
from torchvision.transforms import Resize, InterpolationMode

if __name__ == "__main__":
    batch_size = 128
    diffusion_model = "diffusion_4"
    vae_model = "vae_2"
    device = 0
    img_size = 128
    gen_img_number = 10
    vae_path = glob.glob(os.path.join(f"./checkpoints/{vae_model}/", f"vae*.ckpt"))
    diffusion_path = glob.glob(os.path.join(f"./checkpoints/{diffusion_model}/", f"diffusion*.ckpt"))
    save_fig_path = "./samples"
    if not os.path.exists(save_fig_path):
        os.makedirs(save_fig_path)
    # dataset
    """loading vae"""
    if len(vae_path) > 0:
        vae = VAE.load_from_checkpoint(vae_path[0],
                                       strict=True,
                                       spatial_dims=2,
                                       in_channels=1,
                                       out_channels=1,
                                       vae_num_channels=(128, 128, 256),
                                       latent_channels=3,
                                       num_res_blocks=2,
                                       attention_levels=(False, False, False),
                                       with_encoder_nonlocal_attn=False,
                                       with_decoder_nonlocal_attn=False,
                                       perceptual_weight=0.001,
                                       adv_weight=0.01,
                                       path_discriminator_num_channels=64,
                                       lr_g=1e-4,
                                       lr_d=5e-4,
                                       kl_weight=1e-6,
                                       autoencoder_warm_up_n_epochs=10,
                                       img_size=img_size,
                                       save_fig_path="none"
                                       ).to(f"cuda:{device}")
    else:
        print("No such VAE exists!")
        exit(-1)
    if len(diffusion_path) > 0:
        data_modules = get_data_modules(batch_size=batch_size,
                                        classes=get_full_classes(),
                                        filter_img=False,
                                        train_transform=get_test_transformation(img_size, channels=1, unsharp=True),
                                        threemm=True)

        dataiter = iter(data_modules[0][0].train_dataloader())
        images, _ = next(dataiter)
        diffusion = Diffusion.load_from_checkpoint(
            diffusion_path[0],
            spatial_dims=2,
            in_channels=3,
            out_channels=3,
            num_res_blocks=2,
            num_channels=(128, 256, 512),
            attention_levels=(False, True, True),
            num_head_channels=(0, 256, 512),
            num_train_timesteps=1000,
            beta_start=0.0015,
            beta_end=0.0195,
            lr=1e-4,
            vae=vae.autoencoderkl,
            sample=images,
            save_fig_path=save_fig_path,
            img_size=img_size,
            resize=img_size*2
        ).to(f"cuda:{device}")
        diffusion.eval()
        diffusion.img_size = img_size
        diffusion.resize = Resize(img_size*2, InterpolationMode.BICUBIC)
        with torch.no_grad():
            for i in range(gen_img_number):
               diffusion.generate_sample(epoch=i, stack_imgs=1)

    else:
        print("No such Diffusion exists!")
        exit(-1)
