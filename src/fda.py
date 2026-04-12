import torch
import torch.fft as fft
import numpy as np

def FDA_source_to_target(src_img, trg_img, L=0.01):
    """
    Fourier Domain Adaptation: Swap low frequency amplitude components.
    """
    if src_img.dim() == 3: src_img = src_img.unsqueeze(0)
    if trg_img.dim() == 3: trg_img = trg_img.unsqueeze(0)
        
    fft_src = fft.fft2(src_img, dim=(-2, -1))
    fft_trg = fft.fft2(trg_img, dim=(-2, -1))
    
    amp_src, pha_src = torch.abs(fft_src), torch.angle(fft_src)
    amp_trg = torch.abs(fft_trg)
    
    amp_src_shift = fft.fftshift(amp_src, dim=(-2, -1))
    amp_trg_shift = fft.fftshift(amp_trg, dim=(-2, -1))
    
    B, C, H, W = amp_src.shape
    b = int(np.floor(min(H, W) * L))
    
    if b > 0:
        cy, cx = H // 2, W // 2
        amp_src_shift[:, :, cy-b:cy+b, cx-b:cx+b] = amp_trg_shift[:, :, cy-b:cy+b, cx-b:cx+b]
    
    amp_src_mutated = fft.ifftshift(amp_src_shift, dim=(-2, -1))
    fft_src_mutated = amp_src_mutated * torch.exp(1j * pha_src)
    src_in_trg_style = fft.ifft2(fft_src_mutated, dim=(-2, -1))
    
    return torch.real(src_in_trg_style)

def Fourier_Mixup(src_img, trg_img, L=0.01, lam=0.5):
    """
    Fourier Mixup: Blends amplitude spectrum between source and target.
    """
    if src_img.dim() == 3: src_img = src_img.unsqueeze(0)
    if trg_img.dim() == 3: trg_img = trg_img.unsqueeze(0)
    
    fft_src = fft.fft2(src_img, dim=(-2, -1))
    fft_trg = fft.fft2(trg_img, dim=(-2, -1))
    
    amp_src, pha_src = torch.abs(fft_src), torch.angle(fft_src)
    amp_trg = torch.abs(fft_trg)
    
    amp_src_shift = fft.fftshift(amp_src, dim=(-2, -1))
    amp_trg_shift = fft.fftshift(amp_trg, dim=(-2, -1))
    
    B, C, H, W = amp_src.shape
    b = int(np.floor(min(H, W) * L))
    if b > 0:
        cy, cx = H // 2, W // 2
        amp_src_shift[:, :, cy-b:cy+b, cx-b:cx+b] = \
            (1 - lam) * amp_src_shift[:, :, cy-b:cy+b, cx-b:cx+b] + \
            lam * amp_trg_shift[:, :, cy-b:cy+b, cx-b:cx+b]
            
    amp_mixed = fft.ifftshift(amp_src_shift, dim=(-2, -1))
    fft_mixed = amp_mixed * torch.exp(1j * pha_src)
    out = fft.ifft2(fft_mixed, dim=(-2, -1))
    
    return torch.real(out)

def Feature_FDA(src_feat, trg_feat, L=0.05):
    """
    Feature-Space Fourier Domain Adaptation: Swaps low frequency amplitude components 
    of deep feature embeddings without clamping to [0, 1].
    """
    fft_src = fft.fft2(src_feat, dim=(-2, -1))
    fft_trg = fft.fft2(trg_feat, dim=(-2, -1))
    
    amp_src, pha_src = torch.abs(fft_src), torch.angle(fft_src)
    amp_trg = torch.abs(fft_trg)
    
    amp_src_shift = fft.fftshift(amp_src, dim=(-2, -1))
    amp_trg_shift = fft.fftshift(amp_trg, dim=(-2, -1))
    
    B, C, H, W = amp_src.shape
    b = int(np.floor(min(H, W) * L))
    
    if b > 0:
        cy, cx = H // 2, W // 2
        amp_src_shift[:, :, cy-b:cy+b, cx-b:cx+b] = amp_trg_shift[:, :, cy-b:cy+b, cx-b:cx+b]
    
    amp_src_mutated = fft.ifftshift(amp_src_shift, dim=(-2, -1))
    fft_src_mutated = amp_src_mutated * torch.exp(1j * pha_src)
    out = fft.ifft2(fft_src_mutated, dim=(-2, -1))
    
    return torch.real(out)
