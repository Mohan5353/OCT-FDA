import torch
import torch.nn as nn
import torch.nn.functional as F

class DistributionDisruptionModule(nn.Module):
    """
    DDM: Disrupts domain-specific distributions by mixing channel-wise 
    statistics (mean/std) between source and target features.
    """
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def get_stats(self, x):
        """Extract channel-wise mean and std."""
        B, C, H, W = x.shape
        mu = x.mean(dim=(2, 3), keepdim=True)
        var = x.var(dim=(2, 3), keepdim=True)
        sig = (var + 1e-6).sqrt()
        return mu, sig

    def forward(self, x_s, x_t):
        """
        x_s: Source features [B, C, H, W]
        x_t: Target features [B, C, H, W]
        """
        if not self.training or torch.rand(1).item() > self.p:
            return x_s, x_t

        mu_s, sig_s = self.get_stats(x_s)
        mu_t, sig_t = self.get_stats(x_t)

        # Disruption: Swap statistics or mix them
        # Here we perform a Stochastic Mix: 
        # Source features get Target statistics and vice versa
        x_s_disrupted = ((x_s - mu_s) / sig_s) * sig_t + mu_t
        x_t_disrupted = ((x_t - mu_t) / sig_t) * sig_t + mu_s # Cross-reinforcement

        return x_s_disrupted, x_t_disrupted

def disruption_consistency_loss(pred_orig, pred_disrupted):
    """
    Ensures that the segmentation prediction remains consistent 
    even when the low-level distributions are disrupted.
    """
    return F.mse_loss(torch.softmax(pred_orig, dim=1), torch.softmax(pred_disrupted, dim=1))
