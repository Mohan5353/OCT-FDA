import torch
import torch.nn as nn

class DomainSpecificBatchNorm2d(nn.Module):
    """
    Domain-Specific Batch Normalization (DSBN) layer.
    Maintains multiple BN layers, one for each domain.
    """
    def __init__(self, num_features, num_domains=2, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super(DomainSpecificBatchNorm2d, self).__init__()
        self.bns = nn.ModuleList([
            nn.BatchNorm2d(num_features, eps, momentum, affine, track_running_stats)
            for _ in range(num_domains)
        ])
        self.num_domains = num_domains
        self.current_domain = 0

    def set_domain(self, domain_idx):
        if domain_idx >= self.num_domains:
            raise ValueError(f"Domain index {domain_idx} out of range (max {self.num_domains-1})")
        self.current_domain = domain_idx

    def forward(self, x):
        return self.bns[self.current_domain](x)

def convert_dsbn(model, num_domains=2):
    """
    Recursively replaces all nn.BatchNorm2d with DomainSpecificBatchNorm2d.
    """
    for name, child in model.named_children():
        if isinstance(child, nn.BatchNorm2d):
            # Create DSBN with same parameters
            dsbn = DomainSpecificBatchNorm2d(
                child.num_features, 
                num_domains=num_domains, 
                eps=child.eps, 
                momentum=child.momentum, 
                affine=child.affine, 
                track_running_stats=child.track_running_stats
            )
            # Copy weights from the original BN to all new BNs
            if child.affine:
                with torch.no_grad():
                    for i in range(num_domains):
                        dsbn.bns[i].weight.copy_(child.weight)
                        dsbn.bns[i].bias.copy_(child.bias)
            setattr(model, name, dsbn)
        else:
            convert_dsbn(child, num_domains)

def set_model_domain(model, domain_idx):
    """
    Sets the active domain for all DSBN layers in the model.
    """
    for m in model.modules():
        if isinstance(m, DomainSpecificBatchNorm2d):
            m.set_domain(domain_idx)
