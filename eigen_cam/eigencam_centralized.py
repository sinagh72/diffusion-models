import argparse
import copy
import os.path

import cv2
import torch
import torchvision.models
import torchvision.transforms.v2 as T
import matplotlib.pyplot as plt
from torchvision.transforms import InterpolationMode

from dataset.data_module_handler import get_data_modules
from models.gradcam import GradCAMM
from pytorch_grad_cam import GradCAM, \
    ScoreCAM, \
    GradCAMPlusPlus, \
    AblationCAM, \
    XGradCAM, \
    EigenCAM, \
    EigenGradCAM, \
    LayerCAM, \
    FullGrad

from pytorch_grad_cam.ablation_layer import AblationLayerVit
from transforms.apply_transforms import get_finetune_transformation, get_test_transformation
from plots.auc_plots import get_simmim_model, get_cls_model
from util.data_labels import get_merged_classes
from util.utils import set_seed


def find_key_by_value(my_dict, target_value):
    """
    Return the first key from the dictionary that has the specified value.

    Parameters:
    - my_dict (dict): The dictionary to search.
    - target_value: The value for which the key needs to be found.

    Returns:
    - key: The first key that matches the given value.
    - None: If no key with that value exists.
    """
    # Iterate over each key-value pair in the dictionary
    for key, value in my_dict.items():
        if value == target_value:
            return key  # Return the first key that matches the value

    return None  # Return None if no key matches


# Example usage


def get_args(method="gradcam++"):
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_false', default=True,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument(
        '--image-path',
        type=str,
        default='./examples/both.png',
        help='Input image path')
    parser.add_argument('--aug_smooth', action='store_true',
                        help='Apply test time augmentation to smooth the CAM')
    parser.add_argument(
        '--eigen_smooth',
        action='store_true',
        help='Reduce noise by taking the first principle componenet'
             'of cam_weights*activations')

    parser.add_argument(
        '--method',
        type=str,
        default=method,
        help='Can be gradcam/gradcam++/scorecam/xgradcam/ablationcam')

    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print('Using GPU for acceleration')
    else:
        print('Using CPU for computation')

    return args


def reshape_transform(tensor, height=4, width=4):
    # print(tensor.shape)
    result = tensor.reshape(tensor.size(0),
                            height, width, tensor.size(-1))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result


if __name__ == '__main__':
    """ python swinT_example.py -image-path <path_to_image>
    Example usage of using cam-methods on a SwinTransformers network.

    """
    args = get_args("eigengradcam")
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
    architecture = "simmim_ex40"
    img_size = 128
    batch_size = 1
    simmim = get_simmim_model(architecture, root="./centralized")
    simmim.eval()
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
    for data_module, client_name, classes in data_modules:
        if client_name != "DS5":
            continue
        if not os.path.exists("./grad_cam_res/cl/" + client_name):
            os.makedirs("./grad_cam_res/cl/" + client_name)
        param["step_size"] = len(data_module.train_dataloader())
        model = get_cls_model(architecture, copy.deepcopy(simmim.model.encoder), client_name=client_name,
                              classes=classes, param=param, root="./centralized").cuda()
        # print(model.encoder[0].features[-1][-1])
        model.eval()
        model.to("cuda:1")

        # model = timm.create_model('swin_base_patch4_window7_224', pretrained=True)
        # model.eval()
        # print(model)

        # target_layers = [model.layers[-1].blocks[-1].norm1]
        target_layers = [model.encoder[0].features[-1][-1].norm2]

        if args.method not in methods:
            raise Exception(f"Method {args.method} not implemented")

        if args.method == "ablationcam":
            cam = methods[args.method](model=model,
                                       target_layers=target_layers,
                                       reshape_transform=reshape_transform,
                                       ablation_layer=AblationLayerVit())
        else:
            cam = methods[args.method](model=model,
                                       target_layers=target_layers,
                                       reshape_transform=reshape_transform)

        # Initialize Grad-CAM
        # grad_cam = GradCAMM(model, target_layers)
        for module, c_name, cc in data_modules:
            if c_name != client_name:
                continue
            for cat in cc:
                if not os.path.exists("./grad_cam_res/cl/" + client_name + "/" + cat):
                    os.makedirs("./grad_cam_res/cl/" + client_name + "/" + cat)

            test_loader = module.filtered_test_dataloader(classes)

            # AblationCAM and ScoreCAM have batched implementations.
            # You can override the internal batch size for faster computation.
            cam.batch_size = batch_size

            # heatmap = grad_cam.generate_heatmap(images.cuda())
            # to_pil(images.squeeze(0))
            # Display
            # img = (images + 1) / 2 * 255
            # img = img.to(torch.uint8)
            # img = to_pil(img.squeeze(0))
            # plt.imshow(img.squeeze(0))
            # print(heatmap.shape)
            # plt.imshow(heatmap.cpu().numpy(), alpha=0.5, cmap='jet')
            # plt.show()
            j = 1
            counter = {}
            for data in test_loader:
                images = data[0]
                label = data[1]
                # with torch.no_grad():
                #     preds = model(images.to("cuda:1")).argmax(dim=-1)
                #     prediction = preds.cpu().item() == label
                # print(preds, label, prediction)

                # grayscale_cam = cam(input_tensor=images,
                #                     targets=None,
                #                     eigen_smooth=args.eigen_smooth,
                #                     aug_smooth=args.aug_smooth)

                # Here grayscale_cam has only one image in the batch
                # grayscale_cam = grayscale_cam[0, :]

                # img = (images + 1) / 2 * 255
                # img.to(torch.uint8)
                # # Convert each tensor image slice to a PIL image and collect them in a list
                # rgb_img = to_pil(img.squeeze(0))
                # print(grayscale_cam.shape)
                # print(np.float32(images.squeeze(0).permute(1, 2, 0).numpy()).shape)
                # cam_image = show_cam_on_image(np.float32((images.squeeze(0).permute(1, 2, 0).numpy() + 1) / 2),
                #                               grayscale_cam,
                #                               image_weight=0.7)

                # plt.show()
                actual_category = find_key_by_value(cc, label.item())
                if actual_category != "CSR":
                    continue
                # predicted = find_key_by_value(classes, preds.item()) + "_"
                # if not os.path.exists(os.path.join("./grad_cam_res", "gradcam", client_name + "-" + actual_category)):
                #     continue
                # imgs = os.listdir(os.path.join("./grad_cam_res", "gradcam", client_name + "-" + actual_category))
                # for img in imgs:
                #     if img.endswith(f"_{j}.png"):
                out = to_pil(resize(images.squeeze(0))).squeeze(0)
                plt.imshow(out,
                           "gray")  # Make sure the image is in RGB format if it's a color image
                plt.axis('off')  # Hide the axis
                plt.savefig(
                    os.path.join("./grad_cam_res", "gradcam", client_name,
                                 f"original_img_{j}.png"), bbox_inches='tight', pad_inches=0)
                plt.close()
                print("saved!")
                        # break

                # plt.savefig(os.path.join("./grad_cam_res","cl", client_name, actual_category, str(prediction.item())
                #                          + "_" + predicted + str(j) + ".png"),
                #             bbox_inches='tight', pad_inches=0)
                # plt.close()
                # cam_image = to_pil(cam_image)
                # cam_image.save(f"/haha_cam_{j}.jpg")
                if j % 1000 == 0:
                    print(f"image {j} has been proceed!")
                j += 1

                # hsv_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)
                # cv2.imwrite(f'./haha_cam_{j}.jpg', hsv_image)
