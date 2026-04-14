import torch
import torch.nn as nn
import torch.nn.functional as F

class MIEstimator(nn.Module):
    """
    Mutual Information Neural Estimator (MINE) or simplified InfoNCE style estimator.
    Takes Phase and Amplitude and estimates their dependency.
    """
    def __init__(self, channels, feature_size):
        super().__init__()
        self.conv_amp = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.conv_pha = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, amp, pha):
        # Normalize amp and pha to improve stability
        # amp can have very large values, so we use log1p
        amp = torch.log1p(amp)
        
        amp_feat = self.conv_amp(amp).view(amp.size(0), -1)
        pha_feat = self.conv_pha(pha).view(pha.size(0), -1)
        combined = torch.cat([amp_feat, pha_feat], dim=1)
        return self.fc(combined)

def mi_loss(estimator, amp, pha):
    """
    Minimizes MI by penalizing the estimator's output on joint distributions.
    We want the features to be disentangled, so we minimize the correlation.
    Uses log-sum-exp trick for stability.
    """
    # Joint distribution
    joint = estimator(amp, pha)
    
    # Marginal distribution (shuffling amplitude)
    indices = torch.randperm(amp.size(0))
    marginal = estimator(amp[indices], pha)
    
    # MINE lower bound: MI >= E[joint] - log(E[exp(marginal)])
    # For stability: log(mean(exp(x))) = logsumexp(x) - log(n)
    n = marginal.size(0)
    log_mean_exp_marginal = torch.logsumexp(marginal, dim=0) - torch.log(torch.tensor(n, dtype=torch.float32, device=marginal.device))
    
    mi_est = torch.mean(joint) - log_mean_exp_marginal
    return mi_est

class PhysicsAttenuationLoss(nn.Module):
    """
    Enforces the Beer-Lambert law: I(z) = I0 * exp(-mu * z).
    Simplified: ensures vertical intensity profile is monotonic or respects a gradient.
    """
    def __init__(self, mu=0.01):
        super().__init__()
        self.mu = mu

    def forward(self, features):
        # features: [B, C, H, W]
        # Calculate mean vertical profile across channels and width
        profile = torch.mean(features, dim=(1, 3)) # [B, H]
        
        # dI/dz should be negative (attenuation)
        # We penalize positive gradients (increasing intensity with depth)
        diff = profile[:, 1:] - profile[:, :-1]
        penalty = torch.relu(diff).mean() # Only penalize where diff > 0
        
        return penalty

class TopologicalLoss(nn.Module):
    """
    Proxy for Betti Matching. Preserves connectivity.
    Uses a soft-pooling approach to count connected components or 
    penalize fragmentation of the predicted masks.
    """
    def __init__(self):
        super().__init__()

    def forward(self, pred_mask, target_mask):
        """
        Ensures that the number of connected components in prediction matches target.
        Approximated via the Euler Characteristic or a simple consistency loss.
        """
        # Soft-Euler Characteristic approximation (simplified)
        # Pred_mask: [B, 4, H, W] (logits)
        probs = torch.softmax(pred_mask, dim=1)
        # Focus on fluid classes (indices 1, 2, 3)
        fluid_probs = probs[:, 1:].sum(dim=1, keepdim=True)
        
        # Simple fragmentation penalty: total variation on fluid masks
        # High TV means more edges/fragmentation
        tv_h = torch.pow(fluid_probs[:, :, 1:, :] - fluid_probs[:, :, :-1, :], 2).sum()
        tv_w = torch.pow(fluid_probs[:, :, :, 1:] - fluid_probs[:, :, :, :-1], 2).sum()
        
        return (tv_h + tv_w) / fluid_probs.numel()
