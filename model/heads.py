import torch.nn as nn
from torch import Tensor


class SimCLRProjectionHeadV2(nn.Module):
    """
    Projection head for SimCLR.

    z = W₃ · ReLU(W₂ · ReLU(W₁ · h))
    SimCLR v2, 2020, https://arxiv.org/abs/2006.10029
    """

    def __init__(
        self,
        input_dim: int = 2048,
        hidden_dim: int = 2048,
        output_dim: int = 128,
        batch_norm: bool = True,
    ) -> None:
        super().__init__()

        self.net = nn.Sequential(
            # layer 1
            nn.Linear(input_dim, hidden_dim, bias=not batch_norm),
            nn.BatchNorm1d(hidden_dim) if batch_norm else nn.Identity(),
            nn.ReLU(),
            # layer 2
            nn.Linear(hidden_dim, hidden_dim, bias=not batch_norm),
            nn.BatchNorm1d(hidden_dim) if batch_norm else nn.Identity(),
            nn.ReLU(),
            # layer 3
            nn.Linear(hidden_dim, output_dim, bias=True),
            nn.BatchNorm1d(output_dim) if batch_norm else nn.Identity(),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the projection head.

        """
        return self.net(x)
