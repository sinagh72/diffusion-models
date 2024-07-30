import copy
import glob
import os.path
import numpy as np
import torch
import torchvision.models
import torchvision.transforms.v2 as T
import matplotlib.pyplot as plt
from torchvision.transforms import InterpolationMode
from torch import nn
from dataset.data_module_handler import get_data_modules
from models.base import BaseNet
from models.cls import MLP, ClsMIM
from models.simmim import SimMimWrapper
from pytorch_grad_cam import GradCAM, \
    ScoreCAM, \
    GradCAMPlusPlus, \
    AblationCAM, \
    XGradCAM, \
    EigenCAM, \
    EigenGradCAM, \
    LayerCAM, \
    FullGrad

from pytorch_grad_cam.utils.image import show_cam_on_image, \
    preprocess_image
from pytorch_grad_cam.ablation_layer import AblationLayerVit
from transforms.apply_transforms import get_finetune_transformation, get_test_transformation
from usage import get_args, reshape_transform, find_key_by_value
from plots.auc_plots import get_simmim_model, get_cls_model
from util.data_labels import get_merged_classes
from util.utils import set_seed

if __name__ == '__main__':
    """ python swinT_example.py -image-path <path_to_image>
    Example usage of using cam-methods on a SwinTransformers network.

    """
    args = get_args("eigengradcam")
    # args = get_args("scorecam")
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
    set_seed(42)
    img_size = 128
    batch_size = 1
    param = {
        "wd": 1e-6,
        "lr": 3e-5,
        "beta1": 0.9,
        "beta2": 0.999,
    }
    data_modules = get_data_modules(batch_size=batch_size,
                                    classes=get_merged_classes(),
                                    train_transform=get_finetune_transformation(img_size),
                                    test_transform=get_test_transformation(img_size,
                                                                           apply_adaptation=True),
                                    threemm=True)
    resize = T.Resize(size=(224, 224), interpolation=InterpolationMode.BICUBIC)

    to_pil = T.Compose([T.Grayscale(1),
                        T.ToImage()])
    architecture = "fl_ex7_2_cls_100_0.5"
    # architecture = "ex12_l_swinv2_t_100"

    phase = "fl"
    for data_module, client_name, classes in data_modules:
        if client_name != "DS2":
            continue
        if not os.path.exists(f"./grad_cam_res/{phase}/" + client_name):
            os.makedirs(f"./grad_cam_res/{phase}/" + client_name)
        param["step_size"] = len(data_module.train_dataloader())
        model_path = glob.glob(os.path.join("./centralized", "checkpoints", architecture, client_name, "*.ckpt"))

        simim = SimMimWrapper(
                               lr=1.5e-3,
                               wd=0.05,
                               min_lr=1e-5,
                               patience=10,
                               epochs=100,
                               warmup_lr=1e-5,
                               warmup_epochs=10,
                               gamma=0.1,
                               device="cuda:1",
                               weights=False
                                                   )

        state_dict = torch.load('fl/models/model_round_4.pth')
        # Adjust the keys by removing the 'model.' prefix
        adjusted_state_dict = {'model.' + key: value for key, value in state_dict.items()}
        simim.load_state_dict(adjusted_state_dict)

        model = ClsMIM.load_from_checkpoint(model_path[0],
                                            encoder=copy.deepcopy(simim.model.encoder),
                                            wd=param["wd"],
                                            lr=param["lr"],
                                            beta1=param["beta1"],
                                            beta2=param["beta2"],
                                            step_size=param["step_size"],
                                            gamma=0.5,
                                            classes=classes,
                                            feature_dim=1000
                                            )

        model.eval()
        model.to("cuda:1")

        # model = timm.create_model('swin_base_patch4_window7_224', pretrained=True)
        # model.eval()
        # print(model)

        # target_layers = [model.layers[-1].blocks[-1].norm1]
        target_layers = [model.encoder[0].features[-1][-1].norm2]
        if args.method not in methods:
            raise Exception(f"Method {args.method} not implemented")

        # Initialize Grad-CAM
        # grad_cam = GradCAMM(model, target_layers)
        for module, c_name, cc in data_modules:
            if c_name != client_name:
                continue

            for cat in cc:
                if not os.path.exists(f"./grad_cam_res/{phase}/" + client_name + "/" + cat):
                    os.makedirs(f"./grad_cam_res/{phase}/" + client_name + "/" + cat)

            test_loader = module.filtered_test_dataloader(classes)

            j = 1
            counter = {}
            for data in test_loader:
                if args.method == "ablationcam":
                    cam = methods[args.method](model=model,
                                               target_layers=target_layers,
                                               reshape_transform=reshape_transform,
                                               ablation_layer=AblationLayerVit())
                else:
                    cam = methods[args.method](model=model,
                                               target_layers=target_layers,
                                               reshape_transform=reshape_transform)
                cam.batch_size = batch_size

                images = data[0]
                label = data[1]
                if label in counter:
                    if counter[label] > 200:
                        continue
                    counter[label] += 1
                else:
                    counter[label] = 1
                with torch.no_grad():
                    preds = model(images.to("cuda:1")).argmax(dim=-1)
                    prediction = preds.cpu().item() == label
                    # print(preds, label, prediction)

                grayscale_cam = cam(input_tensor=images,
                                    targets=None,
                                    eigen_smooth=args.eigen_smooth,
                                    aug_smooth=args.aug_smooth)

                # Here grayscale_cam has only one image in the batch
                grayscale_cam = grayscale_cam[0, :]
                # img = (images + 1) / 2 * 255
                # img.to(torch.uint8)
                # # Convert each tensor image slice to a PIL image and collect them in a list
                # rgb_img = to_pil(img.squeeze(0))
                # print(grayscale_cam.shape)
                # print(np.float32(images.squeeze(0).permute(1, 2, 0).numpy()).shape)
                cam_image = show_cam_on_image(np.float32((images.squeeze(0).permute(1, 2, 0).numpy() + 1) / 2),
                                              grayscale_cam,
                                              image_weight=0.7)
                plt.imshow(resize(cam_image), "gray")  # Make sure the image is in RGB format if it's a color image
                plt.axis('off')  # Hide the axis
                # plt.show()
                actual_category = find_key_by_value(cc, label.item())
                predicted = find_key_by_value(classes, preds.item()) + "_"

                plt.savefig(os.path.join("./grad_cam_res", phase,
                                         client_name, actual_category, str(prediction.item())
                                         + "_" + predicted + str(j) + ".png"),
                            bbox_inches='tight', pad_inches=0)
                # cam_image = to_pil(cam_image)
                # cam_image.save(f"/haha_cam_{j}.jpg")
                plt.close()
                # cam_image = to_pil(cam_image)
                # cam_image.save(f"/haha_cam_{j}.jpg")
                print(f"image {j} has been proceed!")
                j += 1
                # hsv_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)
                # cv2.imwrite(f'./haha_cam_{j}.jpg', hsv_image)
