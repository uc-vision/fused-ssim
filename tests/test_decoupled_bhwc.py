from math import exp

import pytest
import torch
import torch.nn.functional as F

from fused_ssim_bhwc import decoupled_fused_ssim


def gaussian(window_size: int, sigma: float) -> torch.Tensor:
    values = [exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2)) for x in range(window_size)]
    weight = torch.tensor(values, dtype=torch.float32)
    return weight / weight.sum()


def create_window(channel: int, device: torch.device) -> torch.Tensor:
    window_1d = gaussian(11, 1.5).unsqueeze(1)
    window_2d = window_1d.mm(window_1d.t()).unsqueeze(0).unsqueeze(0)
    return window_2d.expand(channel, 1, 11, 11).contiguous().to(device=device)


def decoupled_reference(
    img1: torch.Tensor,
    img2: torch.Tensor,
    img3: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    channel = img1.shape[-1]
    window = create_window(channel, img1.device)
    img1_nchw = img1.permute(0, 3, 1, 2).contiguous()
    img2_nchw = img2.permute(0, 3, 1, 2).contiguous()
    img3_nchw = img3.permute(0, 3, 1, 2).contiguous()

    mu1 = F.conv2d(img1_nchw, window, padding=5, groups=channel)
    mu2 = F.conv2d(img2_nchw, window, padding=5, groups=channel)
    mu3 = F.conv2d(img3_nchw, window, padding=5, groups=channel)

    mu1_sq = mu1.square()
    mu2_sq = mu2.square()
    mu3_sq = mu3.square()
    sigma1_sq = F.conv2d(img1_nchw * img1_nchw, window, padding=5, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2_nchw * img2_nchw, window, padding=5, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1_nchw * img2_nchw, window, padding=5, groups=channel) - mu1 * mu2

    sigma1_sq = sigma1_sq.clamp(min=0.0)
    sigma2_sq = sigma2_sq.clamp(min=0.0)
    luminance = (2.0 * mu1 * mu3 + 0.01**2) / (mu1_sq + mu3_sq + 0.01**2)
    contrast_structure = (2.0 * sigma12 + 0.03**2) / (sigma1_sq + sigma2_sq + 0.03**2)
    return luminance.permute(0, 2, 3, 1), contrast_structure.permute(0, 2, 3, 1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA extension test")
def test_decoupled_fused_ssim_bhwc_matches_reference():
    torch.manual_seed(10)
    img1 = torch.rand(2, 23, 19, 3, device="cuda", requires_grad=True)
    img2 = torch.rand(2, 23, 19, 3, device="cuda", requires_grad=True)
    img3 = torch.rand(2, 23, 19, 3, device="cuda", requires_grad=True)

    reference_inputs = [image.detach().clone().requires_grad_(True) for image in (img1, img2, img3)]
    luminance_ref, contrast_structure_ref = decoupled_reference(*reference_inputs)
    loss_ref = (
        0.2 * luminance_ref + 0.4 * contrast_structure_ref + 0.8 * luminance_ref * contrast_structure_ref
    ).mean()
    grads_ref = torch.autograd.grad(loss_ref, reference_inputs)

    luminance, contrast_structure = decoupled_fused_ssim(img1, img2, img3)
    loss = (0.2 * luminance + 0.4 * contrast_structure + 0.8 * luminance * contrast_structure).mean()
    grads = torch.autograd.grad(loss, (img1, img2, img3))

    torch.testing.assert_close(luminance, luminance_ref, rtol=1e-3, atol=1e-5)
    torch.testing.assert_close(contrast_structure, contrast_structure_ref, rtol=1e-3, atol=1e-5)
    for grad, grad_ref in zip(grads, grads_ref, strict=True):
        torch.testing.assert_close(grad, grad_ref, rtol=1e-3, atol=1e-5)
