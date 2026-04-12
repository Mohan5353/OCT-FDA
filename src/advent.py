import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Discriminator(nn.Module):
    """
    Standard pixel-level discriminator for ADVENT.
    Takes entropy maps as input and predicts if they are from Source or Target.
    """
    def __init__(self, num_classes, ndf=64):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(num_classes, ndf, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(ndf, ndf*2, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(ndf*4, ndf*8, kernel_size=4, stride=2, padding=1)
        self.classifier = nn.Conv2d(ndf*8, 1, kernel_size=4, stride=2, padding=1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x = self.leaky_relu(self.conv1(x))
        x = self.leaky_relu(self.conv2(x))
        x = self.leaky_relu(self.conv3(x))
        x = self.leaky_relu(self.conv4(x))
        x = self.classifier(x)
        return x

def prob_2_entropy(prob):
    """ 
    Calculates the normalized entropy map of the probability output.
    prob: [B, C, H, W]
    Returns: [B, C, H, W] entropy map
    """
    n, c, h, w = prob.size()
    # Normalize by log2(c) so entropy is in [0, 1]
    return -torch.mul(prob, torch.log2(prob + 1e-30)) / np.log2(c)
