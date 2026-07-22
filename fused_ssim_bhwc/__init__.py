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
        need_img1 = train and ctx.needs_input_grad[2]
        need_img2 = train and ctx.needs_input_grad[3]
        need_img3 = train and ctx.needs_input_grad[4]
        (
            luminance_map,
            contrast_structure_map,
            dl_dmu1,
            dl_dmu3,
            dcs_dmu1,
            dcs_dmu2,
            dcs_dsigma1_sq,
            dcs_dsigma12,
        ) = decoupled_fusedssim(C1, C2, img1, img2, img3, need_img1, need_img2, need_img3)

        if padding == "valid":
            luminance_map = luminance_map[:, 5:-5, 5:-5, :]
            contrast_structure_map = contrast_structure_map[:, 5:-5, 5:-5, :]

        derivative_maps = (
            (dl_dmu1, need_img1),
            (dl_dmu3, need_img3),
            (dcs_dmu1, need_img1),
            (dcs_dmu2, need_img2),
            (dcs_dsigma1_sq, need_img1 or need_img2),
            (dcs_dsigma12, need_img1 or need_img2),
        )
        ctx.save_for_backward(
            img1.detach(),
            img2,
            img3,
            *(derivative for derivative, needed in derivative_maps if needed),
        )
        ctx.image_gradients = need_img1, need_img2, need_img3
        ctx.C1 = C1
        ctx.C2 = C2
        ctx.padding = padding

        return luminance_map, contrast_structure_map

    @staticmethod
    def backward(ctx, grad_luminance, grad_contrast_structure):
        saved_tensors = iter(ctx.saved_tensors)
        img1 = next(saved_tensors)
        img2 = next(saved_tensors)
        img3 = next(saved_tensors)
        need_img1, need_img2, need_img3 = ctx.image_gradients
        C1, C2, padding = ctx.C1, ctx.C2, ctx.padding

        empty = img1.new_empty((0,))
        dl_dmu1 = next(saved_tensors) if need_img1 else empty
        dl_dmu3 = next(saved_tensors) if need_img3 else empty
        dcs_dmu1 = next(saved_tensors) if need_img1 else empty
        dcs_dmu2 = next(saved_tensors) if need_img2 else empty
        dcs_dsigma1_sq = next(saved_tensors) if need_img1 or need_img2 else empty
        dcs_dsigma12 = next(saved_tensors) if need_img1 or need_img2 else empty

        grad_img1, grad_img2, grad_img3 = decoupled_fusedssim_backward(
            C1,
            C2,
            img1,
            img2,
            img3,
            grad_luminance,
            grad_contrast_structure,
            dl_dmu1,
            dl_dmu3,
            dcs_dmu1,
            dcs_dmu2,
            dcs_dsigma1_sq,
            dcs_dsigma12,
            5 if padding == "valid" else 0,
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
