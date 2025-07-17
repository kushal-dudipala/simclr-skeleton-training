import torch
import torch.nn as nn
import torchvision.models as models
from loss import nt_xent_loss 

valid_backbone_names = [
    'resnet18'
]

class SimCLRModel(nn.Module):
    def __init__(self, 
                 backbone: str = 'resnet18', 
                 out_dim: int = 128, 
                 temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature
        
        assert backbone in valid_backbone_names, f"Invalid base model. Choose from {valid_backbone_names}"

        model = getattr(models, backbone)(pretrained=False)
        # Remove final FC
        self.encoder = nn.Sequential(*list(model.children())[:-1])  
        dim_mlp = model.fc.in_features

        # Projection head (2-layer MLP)
        self.projector = nn.Sequential(
            nn.Linear(dim_mlp, dim_mlp),
            nn.ReLU(),
            nn.Linear(dim_mlp, out_dim)
        )

    def forward(self, x1, x2=None):
        """
        If only x1 is given, return projection.
        If x1 and x2 are given, return SimCLR loss.
        """
        h1 = torch.flatten(self.encoder(x1), start_dim=1)
        z1 = self.projector(h1)

        if x2 is None:
            return z1  

        h2 = torch.flatten(self.encoder(x2), start_dim=1)
        z2 = self.projector(h2)

        loss = nt_xent_loss(z1, z2, temperature=self.temperature)
        return loss
