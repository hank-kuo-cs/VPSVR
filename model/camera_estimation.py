import torch.nn as nn
from torchvision.models import resnet18


sigmoid = nn.Sigmoid()
tanh = nn.Tanh()


class CameraEstimationNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = resnet18(pretrained=True)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(512, 512),
            nn.Linear(512, 256),
            nn.Linear(256, 64),
            nn.Linear(64, 2),
        )

    def forward(self, rgbs):
        features = self.extract_feature(rgbs)
        view_poses = self.fc(features)

        pred_elevs = tanh(view_poses[:, 0]).view(-1, 1)
        pred_azims = sigmoid(view_poses[:, 1]).view(-1, 1)

        return pred_elevs, pred_azims

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
        features = out.view(out.size(0), -1)

        return features
