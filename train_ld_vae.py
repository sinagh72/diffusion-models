import glob
import os
import torch.utils.data as data
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from monai.utils import set_determinism
import lightning.pytorch as pl
from dataset.OCT_dataset import OCTDataset
from dataset.datamodule_handler import get_data_modules
from models.latent_diffusion import LatentDiffusion
from models.vae import VAE
from transforms.apply_transforms import get_train_transformation, get_test_transformation, get_train_transformation2
from utils.labels import get_full_classes
from utils.utils import set_seed
import torch

if __name__ == "__main__":
    # print_config()
    set_determinism(42)
    set_seed(42)
    batch_size = 128
    num_workers = torch.cuda.device_count() * 2
    epochs = 100
    comments = "ld_vae_2"
    ae_model = "vae_2"
    devices = [0]
    img_size = 128
    vae_path = glob.glob(os.path.join(f"./checkpoints/{ae_model}/", f"*.ckpt"))
    # dataset
    print('loading dataset..')

    data_modules = get_data_modules(batch_size=batch_size,
                                    classes=get_full_classes(),
                                    filter_img=False,
                                    threemm=True
                                    )
    combined_train_list = (
            data_modules[0].data_train.img_paths +
            data_modules[1].data_train.img_paths +
            data_modules[2].data_train.img_paths +
            data_modules[3].data_train.img_paths +
            data_modules[4].data_train.img_paths +
            data_modules[5].data_train.img_paths
    )
    train_dataset = OCTDataset(transform=get_train_transformation2(img_size, channels=1),
                               data_dir="",
                               img_paths=combined_train_list)

    train_loader = data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=True
    )
    combined_val_list = (
            data_modules[0].data_val.img_paths
            +
            data_modules[1].data_val.img_paths
            +
            data_modules[2].data_val.img_paths +
            data_modules[3].data_val.img_paths +
            data_modules[4].data_val.img_paths +
            data_modules[5].data_val.img_paths
    )
    val_dataset = OCTDataset(transform=get_test_transformation(img_size, channels=1, unsharp=False),
                             data_dir="",
                             img_paths=combined_val_list)
    val_loader = data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=True
    )
    """loading vae"""
    if len(vae_path) > 0:
        autoencoder = VAE.load_from_checkpoint(vae_path[0],
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
                                               lr_g=2e-4,
                                               lr_d=2e-6,
                                               kl_weight=1e-6,
                                               autoencoder_warm_up_n_epochs=10,
                                               img_size=img_size,
                                               save_fig_path=""
                                               )
    else:
        print("No VAE exists!")
        exit(-1)
    save_path = os.path.join(f"checkpoints", comments)
    save_fig_path = os.path.join(save_path, "latent_diffusion_samples")
    if not os.path.exists(save_fig_path):
        os.makedirs(save_fig_path)

    dataiter = iter(train_loader)
    images, _ = next(dataiter)
    diffusion = LatentDiffusion(
        spatial_dims=2,  # spatial dimensions (2D or 3D, e.g., height and width for images).
        in_channels=3,  # The number of input channels in the input tensor
        out_channels=3,  #The number of output channels in the final output tensor
        num_res_blocks=2,  #The number of residual blocks to use at each level of the network.
        num_channels=(128, 256, 512),
        attention_levels=(False, True, True),
        num_head_channels=(0, 256, 512),
        num_train_timesteps=1000,
        beta_start=0.0015,
        beta_end=0.0195,
        # embedding_dimension=64,
        # with_conditioning=False,
        # condition_dropout=0.15,
        # cross_attention_dim=32,
        lr=1e-4,
        autoencoder=autoencoder.autoencoder,
        sample=images,
        save_fig_path=save_fig_path,
        img_size=img_size,
        generate=True,
        stack_imgs=5
    )
    early_stopping = EarlyStopping(monitor="val_loss", patience=10, verbose=False, mode="min")
    tb_logger = TensorBoardLogger(save_dir=os.path.join("checkpoints", comments), name="latent_diffusion")
    checkpoint = ModelCheckpoint(dirpath=save_path,
                                 filename="latent_diffusion-{val_loss:.5f}", save_weights_only=False,
                                 mode="min", monitor="val_loss", save_top_k=1)

    trainer = pl.Trainer(accelerator='gpu', devices=devices, max_epochs=epochs,
                         callbacks=[
                             early_stopping,
                             checkpoint],
                         logger=[tb_logger],
                         log_every_n_steps=1,
                         check_val_every_n_epoch=1
                         # strategy=DDPStrategy(find_unused_parameters=True),
                         # sync_batchnorm=True,
                         )
    model_path = glob.glob(os.path.join(save_path, "latent_diffusion-*.ckpt"))
    if len(model_path) > 0:
        trainer.fit(model=diffusion,
                    train_dataloaders=train_loader,
                    val_dataloaders=val_loader,
                    ckpt_path=model_path[0])
    else:
        trainer.fit(model=diffusion,
                    train_dataloaders=train_loader,
                    val_dataloaders=val_loader)
