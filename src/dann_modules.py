import torch
import torch.nn as nn
from torch.autograd import Function

class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

def grad_reverse(x, alpha=1.0):
    return GradientReversalFunction.apply(x, alpha)

class DomainDiscriminator(nn.Module):
    """
    Discriminator for DANN. 
    Input: Flattened features from the encoder bottleneck.
    """
    def __init__(self, input_dim=2048, hidden_dim=512):
        super(DomainDiscriminator, self).__init__()
        self.ad_layer1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        self.ad_layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        self.ad_layer3 = nn.Linear(hidden_dim, 1)

    def forward(self, x, alpha=1.0):
        x = grad_reverse(x, alpha)
        x = self.ad_layer1(x)
        x = self.ad_layer2(x)
        x = self.ad_layer3(x)
        return x
