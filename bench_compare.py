"""Compare BHWC (new) vs BCHW (installed fused-ssim) kernel performance."""
import os
import torch
from torch.utils.cpp_extension import load

os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "7.5 8.9 12.0")
src_dir = os.path.dirname(os.path.abspath(__file__))

print("JIT compiling BHWC kernel...")
bhwc = load(
    name="fused_ssim_bhwc_cuda",
    sources=[os.path.join(src_dir, "ext.cpp"), os.path.join(src_dir, "ssim.cu"),
             os.path.join(src_dir, "ssim3d.cu")],
    extra_cuda_cflags=["-O3", "-DFUSED_SSIM_CUDA", "--maxrregcount=32", "--use_fast_math"],
    extra_cflags=["-O3", "-DFUSED_SSIM_CUDA"],
    verbose=False,
)

print("Loading installed BCHW kernel (fused_ssim)...")
import fused_ssim_cuda as bchw
print("Done.\n")

c1 = 0.01**2
c2 = 0.03**2


class AutogradBHWC(torch.autograd.Function):
    @staticmethod
    def forward(ctx, img1, img2):
        img1 = img1.contiguous()
        img2 = img2.contiguous()
        ssim_map, dm1, ds1, ds12 = bhwc.fusedssim(c1, c2, img1, img2, True)
        ctx.save_for_backward(img1.detach(), img2, dm1, ds1, ds12)
        return ssim_map

    @staticmethod
    def backward(ctx, grad):
        img1, img2, dm1, ds1, ds12 = ctx.saved_tensors
        g = bhwc.fusedssim_backward(c1, c2, img1, img2, grad.contiguous(), dm1, ds1, ds12)
        return g, None


class AutogradBCHW(torch.autograd.Function):
    @staticmethod
    def forward(ctx, img1, img2):
        img1 = img1.contiguous()
        img2 = img2.contiguous()
        ssim_map, dm1, ds1, ds12 = bchw.fusedssim(c1, c2, img1, img2, True)
        ctx.save_for_backward(img1.detach(), img2, dm1, ds1, ds12)
        return ssim_map

    @staticmethod
    def backward(ctx, grad):
        img1, img2, dm1, ds1, ds12 = ctx.saved_tensors
        g = bchw.fusedssim_backward(c1, c2, img1, img2, grad.contiguous(), dm1, ds1, ds12)
        return g, None


def bench(fn, img1, img2, iters=200, warmup=50):
    # Forward only
    for _ in range(warmup):
        fn(img1, img2)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn(img1, img2)
    end.record()
    torch.cuda.synchronize()
    fwd_ms = start.elapsed_time(end) / iters

    # Forward + backward
    img1.requires_grad_(True)
    for _ in range(warmup):
        fn(img1, img2).mean().backward()
        img1.grad = None
    torch.cuda.synchronize()

    start.record()
    for _ in range(iters):
        fn(img1, img2).mean().backward()
        img1.grad = None
    end.record()
    torch.cuda.synchronize()
    fwd_bwd_ms = start.elapsed_time(end) / iters
    img1.requires_grad_(False)

    return fwd_ms, fwd_bwd_ms


print("=== Cross-validation: BHWC vs BCHW ===")
for h, w in [(64, 64), (128, 128), (480, 640), (1080, 1920)]:
    ch = 3
    img1_bhwc = torch.rand(1, h, w, ch, device="cuda")
    img2_bhwc = torch.rand(1, h, w, ch, device="cuda")
    img1_bchw = img1_bhwc.permute(0, 3, 1, 2).contiguous()
    img2_bchw = img2_bhwc.permute(0, 3, 1, 2).contiguous()

    bhwc_map, _, _, _ = bhwc.fusedssim(c1, c2, img1_bhwc.contiguous(), img2_bhwc.contiguous(), False)
    bchw_map, _, _, _ = bchw.fusedssim(c1, c2, img1_bchw, img2_bchw, False)
    bchw_as_bhwc = bchw_map.permute(0, 2, 3, 1)

    max_err = (bhwc_map - bchw_as_bhwc).abs().max().item()
    mean_err = (bhwc_map - bchw_as_bhwc).abs().mean().item()
    print(f"  {h}x{w}x{ch}: max_err={max_err:.2e}  mean_err={mean_err:.2e}  "
          f"{'PASS' if max_err < 1e-4 else 'FAIL'}")
print()

print(f"GPU: {torch.cuda.get_device_name()}\n")
print(f"{'Resolution':>15}  {'BCHW fwd':>10}  {'BHWC fwd':>10}  {'speedup':>8}  "
      f"{'BCHW f+b':>10}  {'BHWC f+b':>10}  {'speedup':>8}")
print(f"{'-'*15}  {'-'*10}  {'-'*10}  {'-'*8}  {'-'*10}  {'-'*10}  {'-'*8}")

for h, w in [(480, 640), (720, 1280), (1080, 1920), (1440, 2560)]:
    ch = 3
    img1_bhwc = torch.rand(1, h, w, ch, device="cuda")
    img2_bhwc = torch.rand(1, h, w, ch, device="cuda")
    bhwc_fwd, bhwc_fb = bench(AutogradBHWC.apply, img1_bhwc, img2_bhwc)

    img1_bchw = img1_bhwc.detach().permute(0, 3, 1, 2).contiguous()
    img2_bchw = img2_bhwc.permute(0, 3, 1, 2).contiguous()
    bchw_fwd, bchw_fb = bench(AutogradBCHW.apply, img1_bchw, img2_bchw)

    print(f"  {h}x{w}x{ch}  {bchw_fwd:10.3f}  {bhwc_fwd:10.3f}  {bchw_fwd/bhwc_fwd:7.2f}x  "
          f"{bchw_fb:10.3f}  {bhwc_fb:10.3f}  {bchw_fb/bhwc_fb:7.2f}x")
