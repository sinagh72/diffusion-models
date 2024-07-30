import glob
import os
import torch.utils.data as data
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from monai.utils import set_determinism
import lightning.pytorch as pl

from dataset.OCT_dataset import OCTDataset
from dataset.datamodule_handler import get_data_modules
from models.cls import CLS
from models.latent_diffusion import LatentDiffusion
from models.vae import VAE
from transforms.apply_transforms import get_test_transformation, get_train_transformation2
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
    comments = "cls_1"
    ae_model = "vae_2"
    diffusion_model = "diffusion_3"
    devices = [1]
    img_size = 128
    vae_path = glob.glob(os.path.join(f"./checkpoints/{ae_model}/", f"*.ckpt"))
    diffusion_path = glob.glob(os.path.join(f"./checkpoints/{diffusion_model}/", f"diffusion*.ckpt"))
    # dataset
    print('loading dataset..')

    data_modules = get_data_modules(batch_size=batch_size,
                                    classes=get_full_classes(),
                                    filter_img=False,
                                    threemm=True,
                                    train_transform=get_train_transformation2(img_size, channels=1),
                                    test_transform=get_test_transformation(img_size, channels=1)
                                    )
    combined_train_list = (data_modules[0][0].data_train.img_paths +
                           data_modules[1][0].data_train.img_paths +
                           data_modules[2][0].data_train.img_paths +
                           data_modules[3][0].data_train.img_paths +
                           data_modules[4][0].data_train.img_paths +
                           data_modules[5][0].data_train.img_paths
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
    combined_val_list = (data_modules[0][0].data_val.img_paths +
                         data_modules[1][0].data_val.img_paths +
                         data_modules[2][0].data_val.img_paths +
                         data_modules[3][0].data_val.img_paths +
                         data_modules[4][0].data_val.img_paths +
                         data_modules[5][0].data_val.img_paths)
    val_dataset = OCTDataset(transform=get_test_transformation(img_size, channels=1),
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
    """loading ae"""
    if len(vae_path) > 0:
        autoencoder = VQGAN.load_from_checkpoint(vae_path[0],
                                                 strict=True,
                                                 spatial_dims=2,
                                                 in_channels=1,
                                                 out_channels=1,
                                                 num_channels=(256, 512),
                                                 num_res_channels=512,
                                                 num_res_layers=2,
                                                 downsample_parameters=((2, 4, 1, 1), (2, 4, 1, 1)),
                                                 upsample_parameters=((2, 4, 1, 1, 0), (2, 4, 1, 1, 0)),
                                                 num_embeddings=256,
                                                 embedding_dim=32,
                                                 latent_channels=3,
                                                 path_discriminator_num_channels=64,
                                                 perceptual_weight=0.001,
                                                 adv_weight=0.01,
                                                 lr_g=2e-4,
                                                 lr_d=2e-5,
                                                 generator_warmup=10,
                                                 img_size=img_size,
                                                 save_fig_path="none"
                                                 )
    else:
        print("No AE exists!")
        exit(-1)

    if len(diffusion_path) > 0:
        dataiter = iter(train_loader)
        images, _ = next(dataiter)
        diffusion = LatentDiffusion.load_from_checkpoint(
            diffusion_path[0],
            spatial_dims=2,  # spatial dimensions (2D or 3D, e.g., height and width for images).
            in_channels=32,  # The number of input channels in the input tensor
            out_channels=32,  # The number of output channels in the final output tensor
            num_res_blocks=2,  # The number of residual blocks to use at each level of the network.
            num_channels=(128, 256, 512),
            attention_levels=(False, True, True),
            num_head_channels=(0, 256, 512),
            num_train_timesteps=1000,
            beta_start=0.0015,
            beta_end=0.0195,
            lr=1e-4,
            autoencoder=autoencoder.vqvae,
            sample=images,
            save_fig_path="save_fig_path",
            img_size=img_size,
            generate=False,
        )

    save_path = os.path.join(f"checkpoints", comments)
    cls_model = CLS(encoder=diffusion, feature_dim=32*32*3, classes=data_modules[1][2], lr=1e-5)
    early_stopping = EarlyStopping(monitor="val_loss", patience=100, verbose=False, mode="min")
    tb_logger = TensorBoardLogger(save_dir=os.path.join("checkpoints", comments), name="centralized_cls")
    checkpoint = ModelCheckpoint(dirpath=save_path,
                                 filename="centralized_cls-{val_loss:.5f}", save_weights_only=False,
                                 mode="min", monitor="val_loss", save_top_k=1)
    #
    trainer = pl.Trainer(accelerator='gpu',
                         devices=devices,
                         max_epochs=epochs,
                         callbacks=[
                             early_stopping,
                             checkpoint],
                         logger=[tb_logger],
                         log_every_n_steps=1,
                         # strategy=DDPStrategy(find_unused_parameters=True),
                         deterministic=True,
                         )
    trainer.fit(model=cls_model,
                datamodule=data_modules[1][0])
