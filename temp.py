import torch
import torchvision
from torchvision.models import Swin_V2_T_Weights, Swin_V2_S_Weights

model = torchvision.models.swin_v2_t(weights=Swin_V2_T_Weights.IMAGENET1K_V1)
model.features[0][0] = torch.nn.Conv2d(1, 96, kernel_size=4, stride=4)
print(model)


# model = torchvision.models.resnet18()
# model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
# print(model)