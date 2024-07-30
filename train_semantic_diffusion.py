import glob
import os
import torch.utils.data as data
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.strategies import DDPStrategy
from monai.utils import set_determinism
import lightning.pytorch as pl

from dataset.OCT_dataset import OCTDataset
from dataset.datamodule_handler import get_data_modules
from models.diffusion import Diffusion
from models.vae import VAE
from transforms.apply_transforms import get_train_transformation, get_test_transformation, get_train_transformation2
from utils.labels import get_full_classes
from utils.utils import set_seed
import torch

if __name__ == "__main__":
    # print_config()
    set_determinism(42)
    set_seed(42)
    batch_size = 30
    num_workers = torch.cuda.device_count() * 4
    epochs = 100
    comments = "semantic_diffusion_1"
    devices = [0,1]
    img_size = 128
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
            data_modules[4].data_train.img_paths
            +
            data_modules[5].data_train.img_paths
    )
    train_dataset = OCTDataset(transform=get_train_transformation(img_size, channels=1),
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
            data_modules[0].data_val.img_paths +
            data_modules[1].data_val.img_paths +
            data_modules[2].data_val.img_paths +
            data_modules[3].data_val.img_paths +
            data_modules[4].data_val.img_paths
            +
            data_modules[5].data_val.img_paths
    )
    val_dataset = OCTDataset(transform=get_test_transformation(img_size, channels=1, unsharp=True),
                             data_dir="",
                             img_paths=combined_val_list)
    val_loader = data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=True
    )
    save_path = os.path.join(f"checkpoints", comments)
    save_fig_path = os.path.join(save_path, "diffusion_samples")
    if not os.path.exists(save_fig_path):
        os.makedirs(save_fig_path)

    diffusion = Diffusion(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        num_channels=(128, 128, 256),
        attention_levels=(False, True, True),
        num_res_blocks=1,
        with_conditioning=True,
        num_head_channels=64,
        embedding_dim=256,
        cross_attention_dim=1,
        num_train_timesteps=1000,
        lr=1e-5,
        save_fig_path=save_fig_path,
        img_size=img_size,
        generate=False
    )
    early_stopping = EarlyStopping(monitor="val_loss", patience=10, verbose=False, mode="min")
    tb_logger = TensorBoardLogger(save_dir=os.path.join("checkpoints", comments), name="diffusion")
    checkpoint = ModelCheckpoint(dirpath=save_path,
                                 filename="diffusion-{val_loss:.5f}", save_weights_only=False,
                                 mode="min", monitor="val_loss", save_top_k=1)

    trainer = pl.Trainer(accelerator='gpu', devices=devices, max_epochs=epochs,
                         callbacks=[
                             early_stopping,
                             checkpoint],
                         logger=[tb_logger],
                         check_val_every_n_epoch=1,

                         )
    model_path = glob.glob(os.path.join(save_path, "diffusion-*.ckpt"))
    if len(model_path) > 0:
        trainer.fit(model=diffusion,
                    train_dataloaders=train_loader,
                    val_dataloaders=val_loader,
                    ckpt_path=model_path[0])
    else:
        trainer.fit(model=diffusion,
                    train_dataloaders=train_loader,
                    val_dataloaders=val_loader)
