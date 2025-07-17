import torch
import torch.nn.functional as F



"""
NT-Xent loss implementation for contrastive learning.
Based on "Improved Deep Metric Learning with Multi-class N-pair Loss Objective"
Part of Advances in Neural Information Processing Systems 29 (NIPS 2016)
"""


def nt_xent_loss(z1, z2, temperature=0.5):
    """
    z1, z2: torch.Tensor of shape (N, D)
    Returns: scalar loss averaged over 2N examples
    """
    N = z1.size(0)
    # (2N, D)
    z = torch.cat([z1, z2], dim=0) 
    # L2 normalization              
    z = F.normalize(z, dim=1)                    

    sim_matrix = torch.matmul(z, z.T)
    # (2N, 2N)           
    sim_matrix = sim_matrix / temperature

    # Create positive pair indices
    labels = torch.arange(N, device=z.device)
    labels = torch.cat([labels + N, labels], dim=0)

    # Mask to zero out self-similarity
    mask = torch.eye(2 * N, device=z.device).bool()
    sim_matrix = sim_matrix.masked_fill(mask, float('-inf'))

    # Cross-entropy between true positive and all others
    loss = F.cross_entropy(sim_matrix, labels)
    return loss
