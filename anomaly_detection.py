import glob
import os
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from matplotlib import pyplot as plt
from monai.utils import set_determinism
import lightning.pytorch as pl
from torch import autocast
from tqdm import tqdm
import torch.nn.functional as F
from dataset.datamodule_handler import get_data_modules
from models.diffusion import Diffusion
from models.diffusion_encoder import DiffusionEncoder
from transforms.apply_transforms import get_train_transformation, get_test_transformation
from utils.labels import get_full_classes, get_merged_classes
from utils.utils import set_seed
import torch

if __name__ == "__main__":
    set_determinism(42)
    set_seed(42)
    batch_size = 32
    num_workers = torch.cuda.device_count() * 4
    epochs = 100
    comments = "anomaly_detection_1"
    diffusion = "diffusion_1"
    diffusion_encoder = "diffusion_encoder_1"
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
        de_save_path = os.path.join(f"checkpoints", diffusion_encoder, datamodule.dataset_name)
        de_model_path = glob.glob(os.path.join(de_save_path, "diffusion-*.ckpt"))
        if len(de_model_path) > 0:
            de = DiffusionEncoder.load_from_checkpoint(
                de_model_path[0],
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
            de.eval()
        else:
            print("No such diffusion encoder for ", datamodule.dataset_name)
            continue

        diffusion_save_path = os.path.join(f"checkpoints", diffusion)
        d_model_path = glob.glob(os.path.join(diffusion_save_path, "diffusion-*.ckpt"))
        if len(d_model_path) > 0:
            diffusion = Diffusion.load_from_checkpoint(
                d_model_path[0],
                spatial_dims=2,
                in_channels=1,
                out_channels=1,
                num_channels=(64, 64, 64),
                attention_levels=(False, False, True),
                num_res_blocks=1,
                num_head_channels=64,
                num_train_timesteps=1000,
                lr=2.5e-5,
                save_fig_path="none",
                img_size=img_size,
                generate=False
            )
            diffusion.eval()
        else:
            print("No Such diffusion")
            exit(-1)

        L = 200
        y = torch.tensor(0)  # define the desired class label
        scale = 6  # define the desired gradient scale s

        for data in datamodule.train_dataloader():
            img, label = data[0]
            current_img = torch.t_copy(img)
            progress_bar = tqdm(range(L))  # go back and forth L timesteps

            for i in progress_bar:  # go through the denoising process
                t = L - i
                with autocast(enabled=True):
                    with torch.no_grad():
                        model_output = diffusion(
                            current_img, timesteps=torch.Tensor((t,)).to(current_img.device)
                        ).detach()  # this is supposed to be epsilon

                    with torch.enable_grad():
                        x_in = current_img.detach().requires_grad_(True)
                        logits = de(x_in, timesteps=torch.Tensor((t,)).to(current_img.device))
                        log_probs = F.log_softmax(logits, dim=-1)
                        selected = log_probs[range(len(logits)), y.view(-1)]
                        a = torch.autograd.grad(selected.sum(), x_in)[0]
                        alpha_prod_t = diffusion.scheduler.alphas_cumprod[t]
                        updated_noise = (
                                model_output - (1 - alpha_prod_t).sqrt() * scale * a
                        )  # update the predicted noise epsilon with the gradient of the classifier

                current_img, _ = diffusion.scheduler.step(updated_noise, t, current_img)
                torch.cuda.empty_cache()

            plt.style.use("default")
            plt.imshow(current_img[0, 0].cpu().detach().numpy(), vmin=0, vmax=1, cmap="gray")
            plt.tight_layout()
            plt.axis("off")
            plt.show()

            diff = abs(img.cpu() - current_img[0, 0].cpu()).detach().numpy()
            plt.style.use("default")
            plt.imshow(diff[0, ...], cmap="jet")
            plt.tight_layout()
            plt.axis("off")
            plt.show()

        # save_path = os.path.join("checkpoints", comments, datamodule.dataset_name)
        # # early_stopping = EarlyStopping(monitor="val_loss", patience=10, verbose=False, mode="min")
        # tb_logger = TensorBoardLogger(save_dir=save_path,
        #                               name="diffusion")
        # checkpoint = ModelCheckpoint(dirpath=save_path,
        #                              filename="diffusion-{val_loss:.5f}", save_weights_only=False,
        #                              mode="min", monitor="val_loss", save_top_k=1)
        #
        # trainer = pl.Trainer(accelerator='gpu', devices=devices, max_epochs=epochs,
        #                      callbacks=[
        #                          # early_stopping,
        #                          checkpoint],
        #                      logger=[tb_logger],
        #                      check_val_every_n_epoch=1
        #                      )
        # model_path = glob.glob(os.path.join(save_path, "*.ckpt"))
        # if len(model_path) > 0:
        #     trainer.fit(model=diffusion_encoder,
        #                 datamodule=datamodule,
        #                 ckpt_path=model_path[0])
        # else:
        #     trainer.fit(model=diffusion_encoder,
        #                 datamodule=datamodule
        #                 )
