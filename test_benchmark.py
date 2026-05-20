"""Test correctness and benchmark fused_ssim_bhwc kernel.

JIT-compiles the CUDA extension from source so we can iterate
without building wheels.
"""
import os
import time
import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load

os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "7.5 8.9 12.0")

print("JIT compiling fused_ssim_bhwc_cuda...")
src_dir = os.path.dirname(os.path.abspath(__file__))
fused_ssim_bhwc_cuda = load(
    name="fused_ssim_bhwc_cuda",
    sources=[os.path.join(src_dir, "ext.cpp"), os.path.join(src_dir, "ssim.cu"),
             os.path.join(src_dir, "ssim3d.cu")],
    extra_cuda_cflags=["-O3", "-DFUSED_SSIM_CUDA", "--maxrregcount=32", "--use_fast_math"],
    extra_cflags=["-O3", "-DFUSED_SSIM_CUDA"],
    verbose=False,
)
print("Done.")


def fused_ssim_bhwc(img1: torch.Tensor, img2: torch.Tensor, train: bool = True):
    c1 = 0.01**2
    c2 = 0.03**2
    img1 = img1.contiguous()
    img2 = img2.contiguous()
    return fused_ssim_bhwc_cuda.fusedssim(c1, c2, img1, img2, train)


class FusedSSIMMap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, c1, c2, img1, img2, train=True):
        img1 = img1.contiguous()
        img2 = img2.contiguous()
        ssim_map, dm_dmu1, dm_dsigma1_sq, dm_dsigma12 = fused_ssim_bhwc_cuda.fusedssim(c1, c2, img1, img2, train)
        ctx.save_for_backward(img1.detach(), img2, dm_dmu1, dm_dsigma1_sq, dm_dsigma12)
        ctx.c1 = c1
        ctx.c2 = c2
        return ssim_map

    @staticmethod
    def backward(ctx, opt_grad):
        img1, img2, dm_dmu1, dm_dsigma1_sq, dm_dsigma12 = ctx.saved_tensors
        dL_dmap = opt_grad.contiguous()
        grad = fused_ssim_bhwc_cuda.fusedssim_backward(ctx.c1, ctx.c2, img1, img2, dL_dmap, dm_dmu1, dm_dsigma1_sq, dm_dsigma12)
        return None, None, grad, None, None


def fused_ssim_with_grad(img1, img2, padding="same"):
    c1 = 0.01**2
    c2 = 0.03**2
    ssim_map = FusedSSIMMap.apply(c1, c2, img1, img2, True)
    if padding == "valid":
        ssim_map = ssim_map[:, 5:-5, 5:-5, :]
    return ssim_map.mean()


def reference_ssim_bhwc(img1: torch.Tensor, img2: torch.Tensor):
    """Pure-PyTorch SSIM on BHWC tensors for correctness reference."""
    c1 = 0.01**2
    c2 = 0.03**2
    # Convert to BCHW for F.conv2d
    x = img1.permute(0, 3, 1, 2)
    y = img2.permute(0, 3, 1, 2)
    ch = x.shape[1]

    # 11x1 Gaussian kernel
    gauss = torch.tensor([
        0.001028380123898387, 0.0075987582094967365, 0.036000773310661316,
        0.10936068743467331, 0.21300552785396576, 0.26601171493530273,
        0.21300552785396576, 0.10936068743467331, 0.036000773310661316,
        0.0075987582094967365, 0.001028380123898387,
    ], device=x.device, dtype=x.dtype)

    kernel_h = gauss.view(1, 1, 1, 11).expand(ch, 1, 1, 11)
    kernel_v = gauss.view(1, 1, 11, 1).expand(ch, 1, 11, 1)

    def blur(t):
        t = F.conv2d(t, kernel_h, padding=(0, 5), groups=ch)
        t = F.conv2d(t, kernel_v, padding=(5, 0), groups=ch)
        return t

    mu1 = blur(x)
    mu2 = blur(y)
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    sigma1_sq = blur(x * x) - mu1_sq
    sigma2_sq = blur(y * y) - mu2_sq
    sigma12 = blur(x * y) - mu1 * mu2

    ssim_map = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )
    # Back to BHWC
    return ssim_map.permute(0, 2, 3, 1)


def test_correctness():
    print("\n=== Correctness Tests ===")
    torch.manual_seed(42)

    for h, w, ch in [(64, 64, 3), (128, 128, 3), (256, 256, 1), (100, 200, 3)]:
        img1 = torch.rand(1, h, w, ch, device="cuda")
        img2 = torch.rand(1, h, w, ch, device="cuda")

        ssim_map, _, _, _ = fused_ssim_bhwc(img1, img2, train=False)
        ref = reference_ssim_bhwc(img1, img2)

        max_err = (ssim_map - ref).abs().max().item()
        mean_err = (ssim_map - ref).abs().mean().item()
        print(f"  {h}x{w}x{ch}: max_err={max_err:.2e}  mean_err={mean_err:.2e}  "
              f"{'PASS' if max_err < 1e-4 else 'FAIL'}")

    # Gradient test — use autograd wrapper
    print("\n  Gradient check (64x64x3):")
    img1 = torch.rand(1, 64, 64, 3, device="cuda", requires_grad=True)
    img2 = torch.rand(1, 64, 64, 3, device="cuda")

    loss = fused_ssim_with_grad(img1, img2)
    loss.backward()
    grad = img1.grad
    has_nan = torch.isnan(grad).any().item()
    print(f"    grad range: [{grad.min().item():.6f}, {grad.max().item():.6f}]  NaN: {has_nan}  "
          f"{'PASS' if not has_nan else 'FAIL'}")


def benchmark(h: int, w: int, ch: int, iters: int = 200, warmup: int = 50):
    img1 = torch.rand(1, h, w, ch, device="cuda")
    img2 = torch.rand(1, h, w, ch, device="cuda")

    # Warmup
    for _ in range(warmup):
        fused_ssim_bhwc(img1, img2, train=True)
    torch.cuda.synchronize()

    # Forward
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fused_ssim_bhwc(img1, img2, train=True)
    end.record()
    torch.cuda.synchronize()
    fwd_ms = start.elapsed_time(end) / iters

    # Forward + backward
    img1.requires_grad_(True)
    for _ in range(warmup):
        loss = fused_ssim_with_grad(img1, img2)
        loss.backward()
        img1.grad = None
    torch.cuda.synchronize()

    start.record()
    for _ in range(iters):
        loss = fused_ssim_with_grad(img1, img2)
        loss.backward()
        img1.grad = None
    end.record()
    torch.cuda.synchronize()
    fwd_bwd_ms = start.elapsed_time(end) / iters

    return fwd_ms, fwd_bwd_ms


def run_benchmarks():
    print("\n=== Benchmarks ===")
    print(f"  GPU: {torch.cuda.get_device_name()}")
    print(f"  {'Resolution':>15}  {'Fwd (ms)':>10}  {'Fwd+Bwd (ms)':>14}")
    print(f"  {'-' * 15}  {'-' * 10}  {'-' * 14}")

    for h, w, ch in [
        (480, 640, 3),
        (720, 1280, 3),
        (1080, 1920, 3),
        (1440, 2560, 3),
    ]:
        fwd, fwd_bwd = benchmark(h, w, ch)
        print(f"  {h}x{w}x{ch:>1}  {fwd:10.3f}  {fwd_bwd:14.3f}")


if __name__ == "__main__":
    test_correctness()
    run_benchmarks()
