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
from models.vqgan import VQGAN
from transforms.apply_transforms import get_train_transformation, get_test_transformation
from utils.labels import get_full_classes
from utils.utils import set_seed
import torch

if __name__ == "__main__":
    # print_config()
    set_determinism(42)
    set_seed(42)
    batch_size = 24
    num_workers = torch.cuda.device_count() * 2
    epochs = 100
    comments = "vqgan_1"
    devices = [0, 1, 2, 3]
    img_size = 224
    generator_warmup = 10
    # dataset
    print('loading dataset..')

    data_modules = get_data_modules(batch_size=batch_size,
                                    classes=get_full_classes(),
                                    filter_img=False,
                                    threemm=True
                                    )
    combined_train_list = (data_modules[0].data_train.img_paths +
                           data_modules[1].data_train.img_paths +
                           data_modules[2].data_train.img_paths +
                           data_modules[3].data_train.img_paths +
                           data_modules[4].data_train.img_paths +
                           data_modules[5].data_train.img_paths +
                           data_modules[6].data_train.img_paths +
                           data_modules[7].data_train.img_paths +
                           data_modules[7].data_unlabeled.img_paths +
                           data_modules[8].data_unlabeled.img_paths
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
    combined_val_list = (data_modules[0].data_val.img_paths +
                         data_modules[1].data_val.img_paths +
                         data_modules[2].data_val.img_paths +
                         data_modules[3].data_val.img_paths +
                         data_modules[4].data_val.img_paths +
                         data_modules[5].data_val.img_paths +
                         data_modules[6].data_val.img_paths +
                         data_modules[7].data_val.img_paths
                         )
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
    save_path = os.path.join(f"checkpoints", comments)
    save_fig_path = os.path.join(save_path, "vqa_samples")
    if not os.path.exists(save_fig_path):
        os.makedirs(save_fig_path)
    vqgan = VQGAN(
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
        generator_warmup=generator_warmup,
        img_size=img_size,
        save_fig_path=save_fig_path
    )
    tb_logger = TensorBoardLogger(save_dir=os.path.join("checkpoints", comments), name="centralized_vqgan")
    checkpoint = ModelCheckpoint(dirpath=save_path,
                                 filename="vqgan-{val_loss:.5f}",
                                 mode="min", monitor="val_loss", save_top_k=1)
    early_stopping = EarlyStopping(monitor="val_loss", patience=10, verbose=False, mode="min")
    trainer = pl.Trainer(accelerator='gpu',
                         devices=devices,
                         max_epochs=epochs,
                         callbacks=[
                             early_stopping,
                             checkpoint],
                         logger=[tb_logger],
                         strategy=DDPStrategy(find_unused_parameters=True),
                         sync_batchnorm=True,
                         )
    model_path = glob.glob(os.path.join(save_path, "vqgan-*.ckpt"))
    if len(model_path) > 0:
        trainer.fit(model=vqgan,
                    train_dataloaders=train_loader,
                    val_dataloaders=val_loader,
                    ckpt_path=model_path[0])
    else:
        trainer.fit(model=vqgan,
                    train_dataloaders=train_loader,
                    val_dataloaders=val_loader
                    )
