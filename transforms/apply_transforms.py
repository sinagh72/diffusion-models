import torch

from transforms.transformations import ZScoreNormalization, UnsharpMaskTransform, FastSVDNA, SobelFilter, CustomRotation
import torchvision.transforms.v2 as T


def get_train_transformation(img_size, channels=3, mean=0.5, std=0.5):
    return T.Compose([
        T.Resize((img_size, img_size), T.InterpolationMode.BICUBIC),
        CustomRotation(angles=[0, 90, 180, 270]),
        # T.ToTensor(),
        # FastSVDNA(target_path="./transforms/NORMAL-36734-8.jpeg", img_size=img_size),
        T.RandomApply([UnsharpMaskTransform(radius=2, percent=150, threshold=3)], p=0.2),
        T.RandomHorizontalFlip(p=0.25),
        T.RandomVerticalFlip(p=0.25),
        T.RandomApply([T.ColorJitter(0.5, 0.5)], p=0.2),
        T.RandomApply([T.GaussianBlur(kernel_size=int(5), sigma=(0.75, 1.5))], p=0.2),
        T.RandomApply([T.ElasticTransform(alpha=(50.0, 200.0), sigma=(5.0, 10.0))], p=0.2),
        T.Grayscale(channels),
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize((mean,), (std,))
    ])


def get_train_transformation2(img_size, channels=3, mean=0.5, std=0.5):
    return T.Compose([
        T.Resize((img_size, img_size), T.InterpolationMode.BICUBIC),
        # CustomRotation(angles=[0, 90, 180, 270]),
        # T.ToTensor(),
        # FastSVDNA(target_path="./transforms/NORMAL-36734-8.jpeg", img_size=img_size),
        UnsharpMaskTransform(radius=2, percent=150, threshold=3),
        # T.RandomHorizontalFlip(p=0.5),
        # T.RandomApply([T.ColorJitter(0.3, 0.3)], p=0.2),
        T.Grayscale(channels),
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize([mean], [std])
    ])


def get_test_transformation(img_size, channels=3, mean=0.5, std=0.5, unsharp=False):
    if unsharp:
        return T.Compose([
            T.Resize((img_size, img_size), T.InterpolationMode.BICUBIC),
            # T.ToTensor(),
            # FastSVDNA(target_path="./transforms/NORMAL-36734-8.jpeg", img_size=img_size),
            UnsharpMaskTransform(radius=2, percent=150, threshold=3),
            T.Grayscale(channels),
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize([mean], [std])
        ])
    else:
        return T.Compose([
            T.Resize((img_size, img_size), T.InterpolationMode.BICUBIC),
            # T.ToTensor(),
            # FastSVDNA(target_path="./transforms/NORMAL-36734-8.jpeg", img_size=img_size),
            T.Grayscale(channels),
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize([mean], [std])
        ])
