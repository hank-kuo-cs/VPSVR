"""
Depth Estimation Network:
Assemble the ResNet18 and reverse ResNet18 with U-Net structure to estimate depth map from single RGB input.
We modify the code provided by GenRe (https://github.com/xiumingzhang/GenRe-ShapeHD)
"""
import torch
import torch.nn as nn
from torchvision.models import resnet18


class DepthEstimationUNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder
        resnet = resnet18(pretrained=True)
        init_conv = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)

        self.encoder = nn.ModuleList(
            [init_conv, resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4])

        # Decoder
        rev_resnet = ReverseResNet18(RevBasicBlock, [2, 2, 2, 2], [256, 128, 64, 64],
                                     inplanes=[512, 512, 256, 128, 128], out_planes=1)
        final_deconv = nn.Sequential(rev_resnet.deconv1, rev_resnet.bn1, rev_resnet.relu, rev_resnet.deconv2)

        self.decoder = nn.ModuleList(
            [rev_resnet.layer1, rev_resnet.layer2, rev_resnet.layer3, rev_resnet.layer4, final_deconv])

    def forward(self, x):
        feature_maps = self.encode(x)
        pred_depth = self.decode(feature_maps)
        return pred_depth

    def encode(self, x):
        feature_maps = []

        for layer in self.encoder:
            x = layer(x)
            feature_maps.append(x)

        return feature_maps

    def decode(self, feature_maps):
        x = feature_maps[-1]
        for idx, layer in enumerate(self.decoder):
            x = layer(x)
            if idx != len(self.decoder) - 1:
                x = torch.cat((x, feature_maps[-(idx + 2)]), dim=1)

        pred_depth = torch.clamp(x, min=0.0, max=1.0)
        return pred_depth


class RevBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, upsample=None):
        super().__init__()
        self.deconv1 = self.deconv3x3(inplanes, planes, stride=1)
        # Note that in ResNet, the stride is on the second layer
        # Here we put it on the first layer as the mirrored block
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.deconv2 = self.deconv3x3(planes, planes, stride=stride, output_padding=1 if stride > 1 else 0)
        self.bn2 = nn.BatchNorm2d(planes)
        self.upsample = upsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.deconv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.deconv2(out)
        out = self.bn2(out)
        if self.upsample is not None:
            residual = self.upsample(x)
        out += residual
        out = self.relu(out)
        return out

    @staticmethod
    def deconv3x3(in_planes, out_planes, stride=1, output_padding=0):
        return nn.ConvTranspose2d(
            in_planes,
            out_planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
            output_padding=output_padding
        )


class RevBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, upsample=None):
        super().__init__()
        bottleneck_planes = int(inplanes / 4)
        self.deconv1 = nn.ConvTranspose2d(
            inplanes,
            bottleneck_planes,
            kernel_size=1,
            bias=False,
            stride=1
        )  # conv and deconv are the same when kernel size is 1
        self.bn1 = nn.BatchNorm2d(bottleneck_planes)
        self.deconv2 = nn.ConvTranspose2d(
            bottleneck_planes,
            bottleneck_planes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(bottleneck_planes)
        self.deconv3 = nn.ConvTranspose2d(
            bottleneck_planes,
            planes,
            kernel_size=1,
            bias=False,
            stride=stride,
            output_padding=1 if stride > 0 else 0
        )
        self.bn3 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.upsample = upsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.deconv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.deconv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.deconv3(out)
        out = self.bn3(out)
        if self.upsample is not None:
            residual = self.upsample(x)
        out += residual
        out = self.relu(out)
        return out


class ReverseResNet18(nn.Module):
    def __init__(self, block, layers, planes, inplanes=None, out_planes=5):
        """
        planes: # output channels for each block
        inplanes: # input channels for the input at each layer
            If missing, it will be inferred.
        """
        if inplanes is None:
            inplanes = [512]
        self.inplanes = inplanes[0]
        super().__init__()
        inplanes_after_blocks = inplanes[4] if len(inplanes) > 4 else planes[3]
        self.deconv1 = nn.ConvTranspose2d(
            inplanes_after_blocks,
            planes[3],
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1
        )
        self.deconv2 = nn.ConvTranspose2d(
            planes[3],
            out_planes,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
            output_padding=1
        )
        self.bn1 = nn.BatchNorm2d(planes[3])
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, planes[0], layers[0], stride=2)
        if len(inplanes) > 1:
            self.inplanes = inplanes[1]
        self.layer2 = self._make_layer(block, planes[1], layers[1], stride=2)
        if len(inplanes) > 2:
            self.inplanes = inplanes[2]
        self.layer3 = self._make_layer(block, planes[2], layers[2], stride=2)
        if len(inplanes) > 3:
            self.inplanes = inplanes[3]
        self.layer4 = self._make_layer(block, planes[3], layers[3])

    def _make_layer(self, block, planes, blocks, stride=1):
        upsample = None
        if stride != 1 or self.inplanes != planes:
            upsample = nn.Sequential(
                nn.ConvTranspose2d(
                    self.inplanes,
                    planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                    output_padding=1 if stride > 1 else 0
                ),
                nn.BatchNorm2d(planes),
            )

        layers = [block(self.inplanes, planes, stride, upsample)]
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.deconv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.deconv2(x)
        return x
