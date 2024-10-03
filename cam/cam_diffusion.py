import copy
import glob
import os.path

import torchvision
import torchvision.transforms.v2 as T
from monai.utils import set_determinism
from torchvision.models import ResNet50_Weights
from torchvision.transforms import InterpolationMode
from cam.functions import apply_cam
from cam.utils import get_args
from pytorch_grad_cam import GradCAM, \
    ScoreCAM, \
    GradCAMPlusPlus, \
    AblationCAM, \
    XGradCAM, \
    EigenCAM, \
    EigenGradCAM, \
    LayerCAM, \
    FullGrad

from dataset.datamodule_handler import get_data_modules
from models.cls import Classifier
from models.diffusion import Diffusion
import torch

from transforms.apply_transforms import get_train_transformation, get_test_transformation
from utils.labels import get_merged_classes
from utils.utils import set_seed

if __name__ == '__main__':
    """ python swinT_example.py -image-path <path_to_image>
    Example usage of using cam-methods on a SwinTransformers network.

    """
    method = "eigencam"
    args = get_args(method)
    methods = \
        {
            "gradcam": GradCAM,
            "scorecam": ScoreCAM,
            "gradcam++": GradCAMPlusPlus,
            "ablationcam": AblationCAM,
            "xgradcam": XGradCAM,
            "eigencam": EigenCAM,
            "eigengradcam": EigenGradCAM,
            "layercam": LayerCAM,
            "fullgrad": FullGrad
        }

    if args.method not in list(methods.keys()):
        raise Exception(f"method should be one of {list(methods.keys())}")
    set_determinism(42)
    set_seed(42)
    diffusion_model = "semantic_diffusion_3"
    diffusion_classification_model = "ft_semantic_diffusion_3"
    img_size = 128
    batch_size = 1
    lr_ft = 1e-4

    data_modules = get_data_modules(batch_size=batch_size,
                                    classes=get_merged_classes(),
                                    filter_img=False,
                                    threemm=True,
                                    train_transform=get_train_transformation(img_size, channels=1),
                                    test_transform=get_test_transformation(img_size, channels=1),
                                    env_path="../data/.env"
                                    )[:-2]

    resize = T.Resize(size=(224, 224), interpolation=InterpolationMode.BICUBIC)

    to_pil = T.Compose([T.Grayscale(1),
                        T.ToImage()])

    # semantic encoder
    semantic_encoder = torchvision.models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    semantic_encoder.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

    # semantic_encoder = torchvision.models.swin_v2_t(weights=Swin_V2_T_Weights.IMAGENET1K_V1)
    # semantic_encoder.features[0][0] = torch.nn.Conv2d(1, 96, kernel_size=4, stride=4)

    # semantic_encoder = torchvision.models.convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
    # semantic_encoder.features[0][0] = torch.nn.Conv2d(1, 96, kernel_size=4, stride=4)


    keywords = {
        "spatial_dims": 2,
        "in_channels": 1,
        "out_channels": 1,
        "num_channels": (128, 128, 256),
        "attention_levels": (False, True, True),
        "num_res_blocks": 1,
        "with_conditioning": True,
        "num_head_channels": 64,
        "embedding_dim": 256,
        "cross_attention_dim": 1,
        "num_train_timesteps": 1000,
        "lr": 1e-5,
        "save_fig_path": "",
        "img_size": img_size,
        "generate": False,
        "semantic_encoder": semantic_encoder}

    model_path = glob.glob(os.path.join(os.path.join(f"checkpoints", diffusion_model), "diffusion-*.ckpt"))
    diffusion = Diffusion.load_from_checkpoint(model_path[0], **keywords)
    diffusion.eval()

    for data_module, client_name, classes in data_modules:
        if not os.path.exists(f"./{method}_res/{diffusion_model}/" + client_name):
            os.makedirs(f"./{method}_res/{diffusion_model}/" + client_name)

        model = Classifier.load_from_checkpoint(model_path[0],
                                                    encoder=copy.copy(diffusion.semantic_encoder),
                                                    feature_dim=1000,
                                                    classes=data_module.classes,
                                                    lr=lr_ft)

        # print(model.encoder[0].features[-1][-1])
        model.eval()
        model.to("cuda:1")

        # target_layers = [model.layers[-1].blocks[-1].norm1]
        target_layers = [model.encoder[0].features[-1][-1].norm2]
        apply_cam(args, methods, model, target_layers, data_modules, client_name, phase, batch_size, resize, classes,
                  method)
