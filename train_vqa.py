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
from models.vqa import VQA
from transforms.apply_transforms import get_train_transformation, get_test_transformation
from utils.labels import get_full_classes
from utils.utils import set_seed
import torch

if __name__ == "__main__":
    # print_config()
    set_determinism(42)
    set_seed(42)
    batch_size = 32
    num_workers = torch.cuda.device_count() * 2
    epochs = 100
    comments = "vqa_1"
    devices = [0, 1]
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
                           data_modules[7].data_val.img_paths
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
                         data_modules[6].data_val.img_paths)
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
    vqa = VQA(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        num_res_layers=2,
        downsample_parameters=((2, 4, 1, 1), (2, 4, 1, 1)),
        upsample_parameters=((2, 4, 1, 1, 0), (2, 4, 1, 1, 0)),
        num_channels=(256, 256),
        num_res_channels=(256, 256),
        num_embeddings=256,
        embedding_dim=32,
        img_size=img_size,
        save_fig_path=save_fig_path,
        lr=1e-4,
    )
    tb_logger = TensorBoardLogger(save_dir=os.path.join("checkpoints", comments), name="centralized_vqa")
    checkpoint = ModelCheckpoint(dirpath=save_path,
                                 filename="vqa-{val_loss:.5f}",
                                 mode="min", monitor="val_loss", save_top_k=1)
    early_stopping = EarlyStopping(monitor="val_loss", patience=10, verbose=False, mode="min")
    trainer = pl.Trainer(accelerator='gpu',
                         devices=devices,
                         max_epochs=epochs,
                         callbacks=[
                             early_stopping,
                             checkpoint],
                         logger=[tb_logger],
                         strategy=DDPStrategy(),
                         sync_batchnorm=True,
                         )
    model_path = glob.glob(os.path.join(save_path, "vqa-*.ckpt"))
    if len(model_path) > 0:
        trainer.fit(model=vqa,
                    train_dataloaders=train_loader,
                    val_dataloaders=val_loader,
                    ckpt_path=model_path[0])
    else:
        trainer.fit(model=vqa,
                    train_dataloaders=train_loader,
                    val_dataloaders=val_loader
                    )
