import argparse
import copy
import glob
import os.path
import numpy as np
import torch
import torchvision
import torchvision.transforms.v2 as T
import matplotlib.pyplot as plt
from torchvision.transforms import InterpolationMode
from dataset.datamodule_handler import get_data_modules
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

from models.cls import Classifier
from models.diffusion import Diffusion
from transforms.apply_transforms import get_test_transformation, get_train_transformation
from utils.labels import get_merged_classes
from utils.utils import set_seed, find_key_by_value


def get_classifier(classifier_model, dataset_name, encoder):
    classifier_path = glob.glob(os.path.join(f"../checkpoints", classifier_model, dataset_name, "cls-*.ckpt"), )
    cls_model = Classifier.load_from_checkpoint(classifier_path[0],
                                                encoder=encoder,
                                                feature_dim=256,
                                                classes=data_module.classes,
                                                lr=1e-4)
    return cls_model
def get_diffusion_classifier(diffusion_model, classifier_model, dataset_name):
    diffusion_path = glob.glob(os.path.join(f"../checkpoints", diffusion_model, "*.ckpt"))
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

    cls_model = get_classifier(classifier_model, dataset_name, encoder=copy.copy(diffusion.semantic_encoder))
    return cls_model


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


if __name__ == '__main__':

    """ python swinT_example.py -image-path <path_to_image>
    Example usage of using cam-methods on a SwinTransformers network.

    """
    args = get_args("gradcam++")
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
    data_modules = get_data_modules(batch_size=batch_size,
                                    classes=get_merged_classes(),
                                    train_transform=get_train_transformation(img_size, 3),
                                    test_transform=get_test_transformation(img_size, 3),
                                    threemm=True,
                                    env_path="../data/.env")
    resize = T.Resize(size=(224, 224), interpolation=InterpolationMode.BICUBIC)

    to_pil = T.Compose([T.Grayscale(1),
                        T.ToImage()])
    # architecture = "fl_ex7_2_cls_100_0.5"
    # classifier_model = "finetune_semantic_diffusion_1"
    classifier_model = "resnet18_1"
    diffusion_model = "semantic_diffusion_1"

    file_name = "local"
    for data_module in data_modules:
        if data_module.dataset_name != "DS1":
            continue
        save_eigencam_dir = os.path.join(f"./{file_name}", data_module.dataset_name)
        for category in data_module.classes:
            if not os.path.exists(os.path.join(save_eigencam_dir, category)):
                os.makedirs(os.path.join(save_eigencam_dir, category))

        # model = get_diffusion_classifier(diffusion_model=diffusion_model,
        #                                  classifier_model=classifier_model,
        #                                  dataset_name=data_module.dataset_name)
        model = get_classifier(classifier_model=classifier_model,
                               dataset_name=data_module.dataset_name,
                               encoder=torchvision.models.resnet18(num_classes=256))
        # print(model.encoder[0].features[-1][-1])
        model.eval()
        model.to("cuda:1")

        # model = timm.create_model('swin_base_patch4_window7_224', pretrained=True)
        # model.eval()
        # print(model)

        # target_layers = [model.layers[-1].blocks[-1].norm1]
        # target_layers = [model.encoder[0].features[-1][-1].norm2]
        print(model.encoder[0].layer4[-1])
        target_layers = [model.encoder[0].layer4[-1]]
        # print(model.encoder[0].features[-1][-1].norm1)
        if args.method not in methods:
            raise Exception(f"Method {args.method} not implemented")

        if args.method == "ablationcam":
            cam = methods[args.method](model=model,
                                       target_layers=target_layers,
                                       )
        else:
            cam = methods[args.method](model=model,
                                       target_layers=target_layers,
                                       )

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
        for data in data_module.test_dataloader():
            images = data[0]
            label = data[1]
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
            actual_category = find_key_by_value(data_module.classes, label.item())
            predicted = find_key_by_value(data_module.classes, preds.item()) + "_"

            plt.savefig(os.path.join(save_eigencam_dir, actual_category, str(prediction.item())
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
