import torch

if torch.cuda.is_available():
    from fused_ssim_cuda import fusedssim, fusedssim_backward

allowed_padding = ["same", "valid"]


def _to_bchw(img: torch.Tensor) -> torch.Tensor:
    return img.permute(0, 3, 1, 2).contiguous()


def _to_bhwc(img: torch.Tensor) -> torch.Tensor:
    return img.permute(0, 2, 3, 1)


class FusedSSIMMap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, C1, C2, img1, img2, padding="same", train=True, spatial_dims=2):
        img1_bchw = _to_bchw(img1)
        img2_bchw = _to_bchw(img2)
        ssim_map, dm_dmu1, dm_dsigma1_sq, dm_dsigma12 = fusedssim(C1, C2, img1_bchw, img2_bchw, train)

        ssim_map = _to_bhwc(ssim_map)
        if padding == "valid":
            ssim_map = ssim_map[:, 5:-5, 5:-5, :]

        ctx.save_for_backward(img1_bchw.detach(), img2_bchw, dm_dmu1, dm_dsigma1_sq, dm_dsigma12)
        ctx.C1 = C1
        ctx.C2 = C2
        ctx.padding = padding

        return ssim_map

    @staticmethod
    def backward(ctx, opt_grad):
        img1, img2, dm_dmu1, dm_dsigma1_sq, dm_dsigma12 = ctx.saved_tensors
        C1, C2, padding = ctx.C1, ctx.C2, ctx.padding

        dL_dmap = _to_bchw(opt_grad)
        if padding == "valid":
            dL_dmap_full = torch.zeros_like(img1)
            dL_dmap_full[:, :, 5:-5, 5:-5] = dL_dmap
            dL_dmap = dL_dmap_full

        grad = fusedssim_backward(C1, C2, img1, img2, dL_dmap, dm_dmu1, dm_dsigma1_sq, dm_dsigma12)
        grad = _to_bhwc(grad)

        return None, None, grad, None, None, None, None


def fused_ssim(img1, img2, padding="same", train=True):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    assert padding in allowed_padding

    ssim_map = FusedSSIMMap.apply(C1, C2, img1, img2, padding, train, 2)
    return ssim_map.mean()
