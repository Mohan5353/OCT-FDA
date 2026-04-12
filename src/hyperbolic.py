import torch
import torch.nn as nn
import torch.nn.functional as F

class PoincareManifold:
    def __init__(self, c=1.0):
        self.c = c

    def mobius_add(self, x, y):
        # x, y: [..., D]
        x2 = torch.sum(x * x, dim=-1, keepdim=True)
        y2 = torch.sum(y * y, dim=-1, keepdim=True)
        xy = torch.sum(x * y, dim=-1, keepdim=True)
        num = (1 + 2 * self.c * xy + self.c * y2) * x + (1 - self.c * x2) * y
        denom = 1 + 2 * self.c * xy + self.c**2 * x2 * y2
        return num / (denom + 1e-15)

    def expmap0(self, u):
        """Exponential map from tangent space at origin to the manifold."""
        norm_u = torch.norm(u, p=2, dim=-1, keepdim=True)
        sqrt_c = self.c**0.5
        res = torch.tanh(sqrt_c * norm_u) * u / (sqrt_c * norm_u + 1e-15)
        return self.project(res)

    def project(self, x):
        """Project back into the ball for stability."""
        norm = torch.norm(x, p=2, dim=-1, keepdim=True)
        maxnorm = (1 - 1e-5) / (self.c**0.5)
        cond = norm > maxnorm
        projected = x / (norm + 1e-15) * maxnorm
        return torch.where(cond, projected, x)

    def dist(self, x, y):
        """Hyperbolic distance between x and y."""
        sqrt_c = self.c**0.5
        # dist = (2/sqrt(c)) * atanh(sqrt(c) * ||-x + y|| / (1 - c <x,y>))
        # More stable version using mobius_add
        mob = self.mobius_add(-x, y)
        dist_c = torch.norm(mob, p=2, dim=-1)
        res = (2 / sqrt_c) * torch.atanh(torch.clamp(sqrt_c * dist_c, -1 + 1e-7, 1 - 1e-7))
        return res

class HyperbolicCrossEntropyLoss(nn.Module):
    def __init__(self, num_classes, embedding_dim, c=1.0, ignore_index=255, weight=None):
        super().__init__()
        self.manifold = PoincareManifold(c)
        self.ignore_index = ignore_index
        self.weight = weight
        
        # Euclidean prototypes that will be mapped to hyperbolic space
        self.prototypes = nn.Parameter(torch.randn(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.prototypes)

    def get_probs(self, x):
        """
        Returns softmax probabilities based on hyperbolic distances to prototypes.
        """
        B, C, H, W = x.shape
        x_flat = x.permute(0, 2, 3, 1).reshape(-1, C)
        x_hyp = self.manifold.expmap0(x_flat)
        proto_hyp = self.manifold.expmap0(self.prototypes)
        dists = self.manifold.dist(x_hyp.unsqueeze(1), proto_hyp.unsqueeze(0))
        logits = -dists
        probs = F.softmax(logits, dim=1)
        return probs.view(B, H, W, -1).permute(0, 3, 1, 2)

    def forward(self, x, target):
        """
        Args:
            x: Euclidean features [B, C, H, W]
            target: [B, H, W]
        """
        B, C, H, W = x.shape
        
        # 1. Flatten and map pixel embeddings to Hyperbolic space
        x_flat = x.permute(0, 2, 3, 1).reshape(-1, C) # [N, C]
        x_hyp = self.manifold.expmap0(x_flat) # [N, C]
        
        # 2. Map class prototypes to Hyperbolic space
        proto_hyp = self.manifold.expmap0(self.prototypes) # [K, C]
        
        # 3. Calculate hyperbolic distances
        # x_hyp: [N, 1, C], proto_hyp: [1, K, C]
        dists = self.manifold.dist(x_hyp.unsqueeze(1), proto_hyp.unsqueeze(0)) # [N, K]
        
        # 4. Probabilities are inversely proportional to distance
        logits = -dists
        logits = logits.view(B, H, W, -1).permute(0, 3, 1, 2) # [B, K, H, W]
        
        return F.cross_entropy(logits, target, ignore_index=self.ignore_index, weight=self.weight)

def hyperbolic_radius_loss(x, c=1.0):
    """
    Maximizes the hyperbolic radius of embeddings.
    Used for target domain uncertainty reduction.
    """
    manifold = PoincareManifold(c)
    B, C, H, W = x.shape
    x_flat = x.permute(0, 2, 3, 1).reshape(-1, C)
    x_hyp = manifold.expmap0(x_flat)
    
    radius = torch.norm(x_hyp, p=2, dim=1)
    # Target embeddings usually cluster near origin. 
    # Moving them to the boundary increases confidence.
    return torch.mean(1.0 - radius)
