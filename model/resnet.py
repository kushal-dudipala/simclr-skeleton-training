import torch
import torch.nn as nn
from .loss import nt_xent_loss
from .heads import SimCLRProjectionHeadV2

valid_backbone_names = ["resnet18"]


class SimCLR(nn.Module):
    def __init__(self, backbone: str = "resnet18", output_dim: int = 128):
        super().__init__()
        assert (
            backbone in valid_backbone_names
        ), f"Invalid base model. Choose from {valid_backbone_names}"
        self.backbone = backbone
        self.projection_head = SimCLRProjectionHeadV2(
            input_dim=512, output_dim=output_dim, hidden_dim=128
        )

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(x)
        return z
