import copy
import glob
import os
import torch.utils.data as data
import torchvision
from dotenv import load_dotenv
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.strategies import DDPStrategy
from monai.utils import set_determinism
import lightning.pytorch as pl
from torchvision.models import Swin_V2_T_Weights, ResNet18_Weights, ResNet50_Weights, ConvNeXt_Base_Weights, \
    ConvNeXt_Small_Weights, ConvNeXt_Tiny_Weights

from dataset.OCT_dataset import OCTDataset
from dataset.datamodule_handler import get_data_modules, get_datamodule
from models.cls import Classifier, LSTMClassifier
from models.diffusion import Diffusion
from models.vae import VAE
from transforms.apply_transforms import get_train_transformation, get_test_transformation, get_train_transformation2
from utils.evaluate import evaluate_model
from utils.labels import get_full_classes, get_merged_classes
from utils.log_results import log_results
from utils.utils import set_seed
import torch




if __name__ == "__main__":
    # print_config()
    set_determinism(42)
    set_seed(42)
    batch_size = 128
    num_workers = torch.cuda.device_count() * 4
    epochs = 100
    comments = "task1_baseline_8"
    devices = [1]
    img_size = 128
    embedding_dim = 256
    resume = False
    lr = 1e-4

    mario_classes = {"Reduced": 0,
                     "Stable": 1,
                     "Increased": 2,
                     "Uninterpretable": 3,
                     }
    load_dotenv(dotenv_path='./data/.env')
    dataset_path = os.getenv("DS8" + "_PATH")

    mario_datamodule = get_datamodule(dataset_name="DS8",
                                      dataset_path=dataset_path,
                                      batch_size=batch_size,
                                      mario_classes=mario_classes,
                                      train_transform=get_train_transformation(img_size, channels=3),
                                      test_transform=get_test_transformation(img_size, channels=3),
                                      )
    mario_datamodule.setup("task1_train")
    mario_datamodule.setup("task1_val")
    mario_datamodule.setup("task1_test")
    mario_datamodule.setup("task2_train")
    mario_datamodule.setup("task2_val")
    mario_datamodule.setup("task2_test")
    """task 1"""
    save_path = os.path.join(f"checkpoints", comments, mario_datamodule.dataset_name)
    cls_model = LSTMClassifier(
        # encoder=torchvision.models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1),
                                encoder=torchvision.models.swin_v2_t(weights=Swin_V2_T_Weights.IMAGENET1K_V1),
                           feature_dim=1000,
                           classes=mario_datamodule.classes,
                           lr=lr)
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
                        train_dataloaders=mario_datamodule.task1_train_dataloader(),
                        val_dataloaders=mario_datamodule.task1_val_dataloader(),
                        ckpt_path=model_path[0]
                        )
        else:
            cls_model = LSTMClassifier.load_from_checkpoint(model_path[0],
                                                        encoder=torchvision.models.swin_v2_t(weights=Swin_V2_T_Weights.IMAGENET1K_V1),
                                                        feature_dim=1000,
                                                        classes=mario_datamodule.classes,
                                                        lr=lr)
    else:
        trainer.fit(model=cls_model,
                    train_dataloaders=mario_datamodule.task1_train_dataloader(),
                    val_dataloaders=mario_datamodule.task1_val_dataloader(),
                    )

    test_results = trainer.test(cls_model, mario_datamodule.task1_test_dataloader())
    log_results(classes=mario_datamodule.classes,
                results=test_results,
                comments=mario_datamodule.dataset_name + "_" + comments,
                test_name=mario_datamodule.dataset_name,
                approach="centralized",
                epochs=epochs)


