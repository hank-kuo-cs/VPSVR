import torch
import torch.nn as nn
from torchvision.models import resnet18


class DepthEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.resnet = resnet18(pretrained=True)
        self.conv1 = nn.Conv2d(1, 3, 1, 1)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        self._initialize_weights()

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

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
