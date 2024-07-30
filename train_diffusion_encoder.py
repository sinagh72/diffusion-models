import glob
import os
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from monai.utils import set_determinism
import lightning.pytorch as pl
from dataset.datamodule_handler import get_data_modules
from models.diffusion_encoder import DiffusionEncoder
from transforms.apply_transforms import get_train_transformation, get_test_transformation, get_train_transformation2
from utils.labels import get_merged_classes
from utils.utils import set_seed
import torch

if __name__ == "__main__":
    # print_config()
    set_determinism(42)
    set_seed(42)
    batch_size = 32
    num_workers = torch.cuda.device_count() * 2
    epochs = 100
    comments = "diffusion_encoder_1"
    devices = [1]
    img_size = 64
    # dataset
    print('loading dataset..')

    data_modules = get_data_modules(batch_size=batch_size,
                                    classes=get_merged_classes(),
                                    train_transform=get_train_transformation(img_size=img_size, channels=1),
                                    test_transform=get_test_transformation(img_size=img_size, channels=1),
                                    filter_img=True,
                                    threemm=True
                                    )

    for datamodule in data_modules:
        diffusion_encoder = DiffusionEncoder(
            spatial_dims=2,
            in_channels=1,
            out_channels=len(datamodule.classes),
            num_channels=(32, 64, 64),
            attention_levels=(False, True, True),
            num_res_blocks=(1, 1, 1),
            num_head_channels=64,
            num_train_timesteps=1000,
            lr=2.5e-5,
        )
        save_path = os.path.join(f"checkpoints", comments, datamodule.dataset_name)

        # early_stopping = EarlyStopping(monitor="val_loss", patience=10, verbose=False, mode="min")
        tb_logger = TensorBoardLogger(save_dir=save_path,
                                      name="diffusion")
        checkpoint = ModelCheckpoint(dirpath=save_path,
                                     filename="diffusion-{val_loss:.5f}", save_weights_only=False,
                                     mode="min", monitor="val_loss", save_top_k=1)

        trainer = pl.Trainer(accelerator='gpu', devices=devices, max_epochs=epochs,
                             callbacks=[
                                 # early_stopping,
                                 checkpoint],
                             logger=[tb_logger],
                             check_val_every_n_epoch=10
                             )
        model_path = glob.glob(os.path.join(save_path, "diffusion-*.ckpt"))
        if len(model_path) > 0:
            trainer.fit(model=diffusion_encoder,
                        datamodule=datamodule,
                        ckpt_path=model_path[0])
        else:
            trainer.fit(model=diffusion_encoder,
                        datamodule=datamodule
                        )
