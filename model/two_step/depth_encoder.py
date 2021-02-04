import torch
import torch.nn as nn
from torchvision.models import resnet18


class DepthEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.resnet = resnet18(pretrained=True)
        self.conv1 = nn.Conv2d(1, 3, 1, 1)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

    def forward(self, depth: torch.Tensor) -> (torch.Tensor, list):
        x = self.conv1(depth)
        global_features, perceptual_feature_list = self.extract_feature(x)

        return global_features, perceptual_feature_list

    def extract_feature(self, x):
        out = self.resnet.conv1(x)
        out = self.resnet.bn1(out)
        out = self.resnet.relu(out)
        out = self.resnet.maxpool(out)

        l1_out = self.resnet.layer1(out)
        l2_out = self.resnet.layer2(l1_out)
        l3_out = self.resnet.layer3(l2_out)
        l4_out = self.resnet.layer4(l3_out)

        out = self.avgpool(l4_out)
        global_features = out.view(out.size(0), -1)
        perceptual_feature_list = [l1_out, l2_out, l3_out, l4_out]

        return global_features, perceptual_feature_list
