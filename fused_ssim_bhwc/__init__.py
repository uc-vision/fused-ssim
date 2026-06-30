from typing import Literal

import torch

Padding = Literal["same", "valid"]

if torch.cuda.is_available():
    from fused_ssim_bhwc_cuda import (
        decoupled_fusedssim,
        decoupled_fusedssim_backward,
        fusedssim,
        fusedssim_backward,
    )

allowed_padding: list[Padding] = ["same", "valid"]


class FusedSSIMMap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, C1, C2, img1, img2, padding: Padding = "same", train=True, spatial_dims=2):
        img1 = img1.contiguous()
        img2 = img2.contiguous()
        ssim_map, dm_dmu1, dm_dsigma1_sq, dm_dsigma12 = fusedssim(C1, C2, img1, img2, train)

        if padding == "valid":
            ssim_map = ssim_map[:, 5:-5, 5:-5, :]

        ctx.save_for_backward(img1.detach(), img2, dm_dmu1, dm_dsigma1_sq, dm_dsigma12)
        ctx.C1 = C1
        ctx.C2 = C2
        ctx.padding = padding

        return ssim_map

    @staticmethod
    def backward(ctx, opt_grad):
        img1, img2, dm_dmu1, dm_dsigma1_sq, dm_dsigma12 = ctx.saved_tensors
        C1, C2, padding = ctx.C1, ctx.C2, ctx.padding

        dL_dmap = opt_grad
        if padding == "valid":
            dL_dmap = torch.zeros_like(img1)
            dL_dmap[:, 5:-5, 5:-5, :] = opt_grad

        grad = fusedssim_backward(C1, C2, img1, img2, dL_dmap, dm_dmu1, dm_dsigma1_sq, dm_dsigma12)
        return None, None, grad, None, None, None, None


def fused_ssim(img1, img2, padding: Padding = "same", train=True):
    return fused_ssim_map(img1, img2, padding, train).mean()


def fused_ssim_map(img1, img2, padding: Padding = "same", train=True):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    assert padding in allowed_padding

    img1 = img1.contiguous()
    return FusedSSIMMap.apply(C1, C2, img1, img2, padding, train, 2)


class DecoupledFusedSSIMMap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, C1, C2, img1, img2, img3, padding: Padding = "same", train=True, spatial_dims=2):
        img1 = img1.contiguous()
        img2 = img2.contiguous()
        img3 = img3.contiguous()
        (
            luminance_map,
            contrast_structure_map,
            dl_dmu1,
            dl_dmu3,
            dcs_dmu1,
            dcs_dmu2,
            dcs_dsigma1_sq,
            dcs_dsigma12,
        ) = decoupled_fusedssim(C1, C2, img1, img2, img3, train)

        if padding == "valid":
            luminance_map = luminance_map[:, 5:-5, 5:-5, :]
            contrast_structure_map = contrast_structure_map[:, 5:-5, 5:-5, :]

        ctx.save_for_backward(
            img1.detach(),
            img2,
            img3,
            dl_dmu1,
            dl_dmu3,
            dcs_dmu1,
            dcs_dmu2,
            dcs_dsigma1_sq,
            dcs_dsigma12,
        )
        ctx.C1 = C1
        ctx.C2 = C2
        ctx.padding = padding

        return luminance_map, contrast_structure_map

    @staticmethod
    def backward(ctx, grad_luminance, grad_contrast_structure):
        (
            img1,
            img2,
            img3,
            dl_dmu1,
            dl_dmu3,
            dcs_dmu1,
            dcs_dmu2,
            dcs_dsigma1_sq,
            dcs_dsigma12,
        ) = ctx.saved_tensors
        C1, C2, padding = ctx.C1, ctx.C2, ctx.padding

        dL_dluminance = grad_luminance
        dL_dcontrast_structure = grad_contrast_structure
        if padding == "valid":
            dL_dluminance = torch.zeros_like(img1)
            dL_dcontrast_structure = torch.zeros_like(img1)
            dL_dluminance[:, 5:-5, 5:-5, :] = grad_luminance
            dL_dcontrast_structure[:, 5:-5, 5:-5, :] = grad_contrast_structure

        empty = dl_dmu1.new_empty((0,))
        if not ctx.needs_input_grad[2]:
            dl_dmu1 = empty
            dcs_dmu1 = empty
        if not ctx.needs_input_grad[3]:
            dcs_dmu2 = empty
        if not ctx.needs_input_grad[4]:
            dl_dmu3 = empty

        grad_img1, grad_img2, grad_img3 = decoupled_fusedssim_backward(
            C1,
            C2,
            img1,
            img2,
            img3,
            dL_dluminance,
            dL_dcontrast_structure,
            dl_dmu1,
            dl_dmu3,
            dcs_dmu1,
            dcs_dmu2,
            dcs_dsigma1_sq,
            dcs_dsigma12,
        )

        if not ctx.needs_input_grad[2]:
            grad_img1 = None
        if not ctx.needs_input_grad[3]:
            grad_img2 = None
        if not ctx.needs_input_grad[4]:
            grad_img3 = None

        return None, None, grad_img1, grad_img2, grad_img3, None, None, None


def decoupled_fused_ssim(img1, img2, img3, padding: Padding = "same", train=True):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    assert padding in allowed_padding

    return DecoupledFusedSSIMMap.apply(C1, C2, img1, img2, img3, padding, train, 2)
