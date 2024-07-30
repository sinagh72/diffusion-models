import copy
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
from models.cls import Classifier
from models.diffusion import Diffusion
from models.vae import VAE
from transforms.apply_transforms import get_train_transformation, get_test_transformation, get_train_transformation2
from utils.evaluate import evaluate_model
from utils.labels import get_full_classes, get_merged_classes
from utils.utils import set_seed
import torch

if __name__ == "__main__":
    set_determinism(42)
    set_seed(42)
    batch_size = 32
    num_workers = torch.cuda.device_count() * 4
    epochs = 100
    resume = True
    diffusion_model = "semantic_diffusion_1"
    comments = "finetune_semantic_diffusion_1"
    devices = [1]
    img_size = 128
    # dataset
    print('loading dataset..')

    data_modules = get_data_modules(batch_size=batch_size,
                                    classes=get_merged_classes(),
                                    train_transform=get_train_transformation(img_size, 1),
                                    test_transform=get_test_transformation(img_size, 1),
                                    filter_img=True,
                                    threemm=True
                                    )

    diffusion_path = glob.glob(os.path.join(f"checkpoints", diffusion_model, "*.ckpt"))
    diffusion = Diffusion.load_from_checkpoint(diffusion_path[0],
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
                                               save_fig_path=None,
                                               img_size=img_size,
                                               generate=False
                                               )

    for data_module in data_modules:
        save_path = os.path.join(f"checkpoints", comments, data_module.dataset_name)
        cls_model = Classifier(encoder=copy.copy(diffusion.semantic_encoder),
                               feature_dim=256,
                               classes=data_module.classes,
                               lr=1e-4)
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
            if resume:
                trainer.fit(model=cls_model,
                            datamodule=data_module,
                            ckpt_path=model_path[0]
                            )
            else:
                cls_model = Classifier.load_from_checkpoint(model_path[0],
                                                            encoder=copy.copy(diffusion.semantic_encoder),
                                                            feature_dim=256,
                                                            classes=data_module.classes,
                                                            lr=1e-4)
        else:
            trainer.fit(model=cls_model,
                        datamodule=data_module,
                        )

        evaluate_model(model=cls_model,
                       classes=data_module.classes,
                       dataset_name=data_module.dataset_name,
                       data_modules=data_modules,
                       devices=devices,
                       comments=comments,
                       epochs=100
                       )
