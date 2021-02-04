import torch
import torch.nn as nn

tanh = nn.Tanh()
sigmoid = nn.Sigmoid()


class TranslateDecoder(nn.Module):
    def __init__(self, vp_num=16):
        super().__init__()

        self.translate_fc = self._make_linear(3 * vp_num)

    def forward(self, global_features: torch.Tensor) -> list:
        translates = self.translate_fc(global_features)
        translates = tanh(translates)
        translates = list(translates.split(3, dim=1))

        return translates  # list[torch.Tensor(B, 3), torch.Tensor(B, 3), ..., ], len = vp_num

    @staticmethod
    def _make_linear(output_dim: int):
        return nn.Sequential(
            nn.Linear(512, 1024),
            nn.Linear(1024, 1024),
            nn.Linear(1024, 512),
            nn.Linear(512, output_dim),
        )
