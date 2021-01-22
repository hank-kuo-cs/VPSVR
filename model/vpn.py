"""
Volumetric Primitive Prediction Network:
Assemble volumetric primitives (spheres, cuboids, and cones) to reconstruct 3D mesh.
Theses volumetric primitives will be represented by volume, quaternion, and center.
Volume (3D) represents the shape of volumetric primitives (width, height, ...)
Quaternion (4D) represents the direction of volumetric primitives (direction vector (3D) and rotation angle (1D))
Center (3D) represents the absolute position of volumetric primitives (x, y, z).
We use ResNet18 to extract features from single depth map,
and regress it to volume, quaternion, center by 3 fully connected network.
"""
import torch
import torch.nn as nn
from torchvision.models import resnet18


sigmoid = nn.Sigmoid()
tanh = nn.Tanh()


class VolumetricPrimitiveNet(nn.Module):
    def __init__(self, vp_num=16, volume_restrict=None, volume_eps=0.1):
        super().__init__()

        self.vp_num = vp_num
        self.volume_eps = volume_eps
        self.volume_restrict = volume_restrict if volume_restrict is not None else [8, 10, 12]

        self.resnet = resnet18()
        self.conv1 = nn.Conv2d(1, 3, 1, 1)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        self.volume_fc = self._make_linear(3 * vp_num)
        self.rotate_fc = self._make_linear(4 * vp_num)
        self.translate_fc = self._make_linear(3 * vp_num)

    def forward(self, depth):
        x = self.conv1(depth)
        global_features, perceptual_features = self.extract_feature(x)

        volumes = self.volume_fc(global_features)
        rotates = self.rotate_fc(global_features)
        translates = self.translate_fc(global_features)

        volumes, rotates, translates = self.restrict_range(volumes, rotates, translates)

        volumes = list(volumes.split(3, dim=1))
        rotates = list(rotates.split(4, dim=1))
        translates = list(translates.split(3, dim=1))

        volumes = self.restrict_volumes(volumes)

        return volumes, rotates, translates, perceptual_features, global_features

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
        perceptual_features = [l1_out, l2_out, l3_out, l4_out]

        return global_features, perceptual_features

    def restrict_range(self, volumes, rotates, translates):
        volumes = sigmoid(volumes) + self.volume_eps
        rotates = sigmoid(rotates)
        translates = tanh(translates)

        return volumes, rotates, translates

    def restrict_volumes(self, volumes):
        for i in range(len(volumes)):
            for j in range(3):
                volumes[i][:, j] = torch.div(volumes[i][:, j], self.volume_restrict[j])
        return volumes

    @staticmethod
    def _make_linear(output_dim: int):
        return nn.Sequential(
            nn.Linear(512, 1024),
            nn.Linear(1024, 1024),
            nn.Linear(1024, 1024),
            nn.Linear(1024, 512),
            nn.Linear(512, output_dim),
        )
