import copy
import glob
import os
import torch.utils.data as data
import torchvision
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.strategies import DDPStrategy
from monai.utils import set_determinism
import lightning.pytorch as pl
from torchvision.models import Swin_V2_T_Weights, ResNet18_Weights, ResNet50_Weights, ConvNeXt_Base_Weights, \
    ConvNeXt_Small_Weights, ConvNeXt_Tiny_Weights

from dataset.OCT_dataset import OCTDataset
from dataset.datamodule_handler import get_data_modules
from models.cls import Classifier
from models.diffusion import Diffusion
from models.vae import VAE
from transforms.apply_transforms import get_train_transformation, get_test_transformation, get_train_transformation2
from utils.evaluate import evaluate_model
from utils.labels import get_full_classes, get_merged_classes
from utils.utils import set_seed
import torch

if __name__ == "__main__":
    # print_config()
    set_determinism(42)
    set_seed(42)
    batch_size = 24
    num_workers = torch.cuda.device_count() * 4
    epochs = 100
    pt_comments = "semantic_diffusion_4"
    ft_comments = "ft_semantic_diffusion_4"
    devices = [1]
    img_size = 128
    embedding_dim = 256
    resume_pt = True
    resume_ft = False
    lr_ft = 1e-4
    # semantic encoder
    # semantic_encoder = torchvision.models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    # semantic_encoder.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

    semantic_encoder = torchvision.models.swin_v2_t(weights=Swin_V2_T_Weights.IMAGENET1K_V1)
    semantic_encoder.features[0][0] = torch.nn.Conv2d(1, 96, kernel_size=4, stride=4)

    # semantic_encoder = torchvision.models.convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
    # semantic_encoder.features[0][0] = torch.nn.Conv2d(1, 96, kernel_size=4, stride=4)
    # dataset
    print('loading dataset..')

    data_modules = get_data_modules(batch_size=batch_size,
                                    classes=get_full_classes(),
                                    filter_img=False,
                                    threemm=True,
                                    )
    combined_train_list = (
            data_modules[0].data_train.img_paths +
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
    combined_val_list = (
            data_modules[0].data_val.img_paths +
            data_modules[1].data_val.img_paths +
            data_modules[2].data_val.img_paths +
            data_modules[3].data_val.img_paths +
            data_modules[4].data_val.img_paths +
            data_modules[5].data_val.img_paths +
            data_modules[6].data_val.img_paths +
            data_modules[7].data_val.img_paths
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
    save_path = os.path.join(f"checkpoints", pt_comments)
    save_fig_path = os.path.join(save_path, "diffusion_samples")
    if not os.path.exists(save_fig_path):
        os.makedirs(save_fig_path)
    keywords = {
        "spatial_dims": 2,
        "in_channels": 1,
        "out_channels": 1,
        "num_channels": (128, 128, 256),
        "attention_levels": (False, True, True),
        "num_res_blocks": 1,
        "with_conditioning": True,
        "num_head_channels": 64,
        "embedding_dim": embedding_dim,
        "cross_attention_dim": 1,
        "num_train_timesteps": 1000,
        "lr": 1e-5,
        "save_fig_path": save_fig_path,
        "img_size": img_size,
        "generate": False,
        "semantic_encoder": semantic_encoder}

    diffusion = Diffusion(**keywords)
    early_stopping = EarlyStopping(monitor="val_loss", patience=10, verbose=False, mode="min")
    tb_logger = TensorBoardLogger(save_dir=os.path.join("checkpoints", pt_comments), name="diffusion")
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
        if resume_pt:
            trainer.fit(model=diffusion,
                        train_dataloaders=train_loader,
                        val_dataloaders=val_loader,
                        ckpt_path=model_path[0])
        else:
            diffusion = Diffusion.load_from_checkpoint(model_path[0],
                                                       **keywords
                                                       )
    else:
        trainer.fit(model=diffusion,
                    train_dataloaders=train_loader,
                    val_dataloaders=val_loader)
    model_path = glob.glob(os.path.join(save_path, "diffusion-*.ckpt"))
    diffusion = Diffusion.load_from_checkpoint(model_path[0], **keywords)
    # for classification
    data_modules = get_data_modules(batch_size=batch_size,
                                    classes=get_merged_classes(),
                                    filter_img=False,
                                    threemm=True,
                                    train_transform=get_train_transformation(img_size, channels=1),
                                    test_transform=get_test_transformation(img_size, channels=1)
                                    )[:-2]
    for data_module in data_modules:
        save_path = os.path.join(f"checkpoints", ft_comments, data_module.dataset_name)
        cls_model = Classifier(encoder=copy.copy(diffusion.semantic_encoder),
                               feature_dim=1000,
                               classes=data_module.classes,
                               lr=lr_ft)
        early_stopping = EarlyStopping(monitor="val_loss", patience=10, verbose=False, mode="min")
        tb_logger = TensorBoardLogger(save_dir=save_path, name="cls")
        checkpoint = ModelCheckpoint(dirpath=save_path,
                                     filename="cls-{val_loss:.5f}", save_weights_only=False,
                                     mode="min", monitor="val_loss", save_top_k=1)

        trainer = pl.Trainer(accelerator='gpu', devices=devices, max_epochs=epochs,
                             callbacks=[
                                 early_stopping,
                                 checkpoint],
                             logger=[tb_logger],
                             check_val_every_n_epoch=1,

                             )
        model_path = glob.glob(os.path.join(save_path, "cls-*.ckpt"), )
        if len(model_path) > 0:
            if resume_ft:
                trainer.fit(model=cls_model,
                            datamodule=data_module,
                            ckpt_path=model_path[0]
                            )
            else:
                cls_model = Classifier.load_from_checkpoint(model_path[0],
                                                            encoder=copy.copy(diffusion.semantic_encoder),
                                                            feature_dim=1000,
                                                            classes=data_module.classes,
                                                            lr=lr_ft)
        else:
            trainer.fit(model=cls_model,
                        datamodule=data_module,
                        )

        evaluate_model(model=cls_model,
                       classes=data_module.classes,
                       dataset_name=data_module.dataset_name,
                       data_modules=data_modules[:-1],
                       devices=devices,
                       comments=ft_comments,
                       epochs=100
                       )
