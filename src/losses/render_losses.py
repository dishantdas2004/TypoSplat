"""
TypoSplat Rendering Losses
==========================
Computes 2D image-space losses between the gsplat render and the ground truth RGB.
Expects input tensors in PyTorch channel-first format: [B, 3, H, W] bounded in [0, 1].
"""

import torch
import torch.nn.functional as F
import torchvision.models as models

def _check_shape(tensor, name="Tensor"):
    """Guards against gsplat's [B, H, W, C] output bleeding into PyTorch CNNs."""
    assert tensor.dim() == 4, f"{name} must be 4D, got {tensor.shape}"
    assert tensor.shape[1] == 3, f"{name} must be channel-first [B, 3, H, W], got {tensor.shape}. Did you forget to permute(0, 3, 1, 2)?"

def compute_l1_rgb_loss(pred_rgb, gt_rgb, mask=None):
    """Standard L1 pixel-wise loss, with optional spatial masking."""
    _check_shape(pred_rgb, "pred_rgb")
    _check_shape(gt_rgb, "gt_rgb")
    
    if mask is not None:
        assert mask.shape[2:] == pred_rgb.shape[2:], f"Mask shape {mask.shape} doesn't match image {pred_rgb.shape}"
        abs_err = torch.abs(pred_rgb - gt_rgb) * mask
        return abs_err.sum() / (mask.sum() + 1e-6)
        
    return F.l1_loss(pred_rgb, gt_rgb)

def compute_sobel_edge_loss(pred_rgb, gt_rgb, mask=None):
    """
    Computes L1 loss on the Sobel-filtered gradients.
    If a mask is provided, it isolates the loss to the typography regions,
    ignoring background wall textures.
    """
    _check_shape(pred_rgb, "pred_rgb")
    _check_shape(gt_rgb, "gt_rgb")
    device = pred_rgb.device
    
    # 1. Convert to Grayscale [B, 1, H, W]
    weight = torch.tensor([0.2989, 0.5870, 0.1140], device=device).view(1, 3, 1, 1)
    pred_gray = (pred_rgb * weight).sum(dim=1, keepdim=True)
    gt_gray = (gt_rgb * weight).sum(dim=1, keepdim=True)
    
    # 2. Define Sobel kernels
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=device).view(1, 1, 3, 3)
    
    # 3. Apply convolutions
    pred_grad_x = F.conv2d(pred_gray, sobel_x, padding=1)
    pred_grad_y = F.conv2d(pred_gray, sobel_y, padding=1)
    
    gt_grad_x = F.conv2d(gt_gray, sobel_x, padding=1)
    gt_grad_y = F.conv2d(gt_gray, sobel_y, padding=1)
    
    # 4. Compute gradient magnitudes
    pred_mag = torch.sqrt(pred_grad_x**2 + pred_grad_y**2 + 1e-6)
    gt_mag = torch.sqrt(gt_grad_x**2 + gt_grad_y**2 + 1e-6)
    
    # 5. Masking
    if mask is not None:
        assert mask.shape[2:] == pred_rgb.shape[2:], f"Mask shape {mask.shape} doesn't match image {pred_rgb.shape}"
        # FIX: Compute the absolute difference BEFORE masking
        abs_err = torch.abs(pred_mag - gt_mag) * mask
        return abs_err.sum() / (mask.sum() + 1e-6)
        
    return F.l1_loss(pred_mag, gt_mag)

class ShallowPerceptualLoss(torch.nn.Module):
    """
    Lightweight LPIPS alternative using VGG-16 sliced up to relu3_3.
    """
    def __init__(self, device):
        super().__init__()
        # weights_only=True is modern PyTorch best practice for security
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        
        self.vgg_slice = torch.nn.Sequential()
        for i in range(16):
            self.vgg_slice.add_module(str(i), vgg[i])
            
        self.vgg_slice.eval()
        self.vgg_slice.to(device)
        
        for param in self.parameters():
            param.requires_grad = False
            
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device))

    def forward(self, pred, gt, mask=None):
        _check_shape(pred, "pred_rgb")
        _check_shape(gt, "gt_rgb")
        
        pred_norm = (pred - self.mean) / self.std
        gt_norm = (gt - self.mean) / self.std
        
        pred_feat = self.vgg_slice(pred_norm)
        gt_feat = self.vgg_slice(gt_norm)
        
        if mask is not None:
            # Mask in feature space to avoid generating artificial edges that VGG will detect
            mask_feat = F.interpolate(mask, size=pred_feat.shape[-2:], mode='nearest')
            abs_err = torch.abs(pred_feat - gt_feat) * mask_feat
            return abs_err.sum() / (mask_feat.sum() + 1e-6)
        
        return F.l1_loss(pred_feat, gt_feat)