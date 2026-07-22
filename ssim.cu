#include <torch/extension.h>
#include <cooperative_groups.h>
#include <algorithm>
#include <c10/cuda/CUDAGuard.h>

namespace cg = cooperative_groups;

__constant__ float cGauss[11] = {
    0.001028380123898387f,
    0.0075987582094967365f,
    0.036000773310661316f,
    0.10936068743467331f,
    0.21300552785396576f,
    0.26601171493530273f,
    0.21300552785396576f,
    0.10936068743467331f,
    0.036000773310661316f,
    0.0075987582094967365f,
    0.001028380123898387f
};

#define BLOCK_X 16
#define BLOCK_Y 16
#define HALO    5

#define SHARED_X (BLOCK_X + 2 * HALO)
#define SHARED_Y (BLOCK_Y + 2 * HALO)
#define CONV_X BLOCK_X
#define CONV_Y SHARED_Y

// BHWC pixel fetch with zero padding
__device__ __forceinline__ float get_pix_bhwc(
    const float* img, int b, int c, int y, int x,
    int CH, int H, int W
) {
    if (x < 0 || x >= W || y < 0 || y >= H) return 0.0f;
    return img[b * H * W * CH + y * W * CH + x * CH + c];
}

// BHWC gradient fetch. Valid-mode gradients omit the five-pixel border.
__device__ __forceinline__ float get_grad_bhwc(
    const float* grad, int b, int c, int y, int x,
    int CH, int H, int W, int padding
) {
    const int grad_y = y - padding;
    const int grad_x = x - padding;
    const int grad_h = H - 2 * padding;
    const int grad_w = W - 2 * padding;
    if (grad_x < 0 || grad_x >= grad_w || grad_y < 0 || grad_y >= grad_h) return 0.0f;
    return grad[b * grad_h * grad_w * CH + grad_y * grad_w * CH + grad_x * CH + c];
}

// BCHW pixel fetch with zero padding (for internal derivative maps)
__device__ __forceinline__ float get_pix_bchw(
    const float* img, int b, int c, int y, int x,
    int CH, int H, int W
) {
    if (x < 0 || x >= W || y < 0 || y >= H) return 0.0f;
    return img[b * CH * H * W + c * H * W + y * W + x];
}

__global__ void fusedssimCUDA(
    int H, int W, int CH,
    float C1, float C2,
    const float* __restrict__ img1,
    const float* __restrict__ img2,
    float* __restrict__ ssim_map,
    float* __restrict__ dm_dmu1,
    float* __restrict__ dm_dsigma1_sq,
    float* __restrict__ dm_dsigma12
) {
    auto block = cg::this_thread_block();
    const int bIdx   = block.group_index().z;
    const int pix_y  = block.group_index().y * BLOCK_Y + block.thread_index().y;
    const int pix_x  = block.group_index().x * BLOCK_X + block.thread_index().x;
    const int pix_id = pix_y * W + pix_x;
    const int num_pix = H * W;

    __shared__ float sTile[SHARED_Y][SHARED_X][2];
    __shared__ float xconv[CONV_Y][CONV_X][5];

    for (int c = 0; c < CH; ++c) {
        // 1) Load tile from BHWC input
        {
            const int tileSize = SHARED_Y * SHARED_X;
            const int threads  = BLOCK_X * BLOCK_Y;
            const int steps    = (tileSize + threads - 1) / threads;
            const int startY   = block.group_index().y * BLOCK_Y;
            const int startX   = block.group_index().x * BLOCK_X;

            for (int s = 0; s < steps; ++s) {
                int tid = s * threads + block.thread_rank();
                if (tid < tileSize) {
                    int ly = tid / SHARED_X;
                    int lx = tid % SHARED_X;
                    int gy = startY + ly - HALO;
                    int gx = startX + lx - HALO;
                    sTile[ly][lx][0] = get_pix_bhwc(img1, bIdx, c, gy, gx, CH, H, W);
                    sTile[ly][lx][1] = get_pix_bhwc(img2, bIdx, c, gy, gx, CH, H, W);
                }
            }
        }
        block.sync();

        // 2) Horizontal convolution
        {
            int ly = threadIdx.y;
            int lx = threadIdx.x + HALO;

            float sumX = 0.f, sumX2 = 0.f, sumY = 0.f, sumY2 = 0.f, sumXY = 0.f;

#pragma unroll
            for (int d = 1; d <= HALO; ++d) {
                float w  = cGauss[HALO - d];
                float Xl = sTile[ly][lx - d][0];
                float Yl = sTile[ly][lx - d][1];
                float Xr = sTile[ly][lx + d][0];
                float Yr = sTile[ly][lx + d][1];

                sumX  += (Xl + Xr) * w;
                sumX2 += (Xl * Xl + Xr * Xr) * w;
                sumY  += (Yl + Yr) * w;
                sumY2 += (Yl * Yl + Yr * Yr) * w;
                sumXY += (Xl * Yl + Xr * Yr) * w;
            }
            {
                float X = sTile[ly][lx][0];
                float Y = sTile[ly][lx][1];
                float wc = cGauss[HALO];
                sumX  += X * wc;
                sumX2 += (X * X) * wc;
                sumY  += Y * wc;
                sumY2 += (Y * Y) * wc;
                sumXY += (X * Y) * wc;
            }
            xconv[ly][threadIdx.x][0] = sumX;
            xconv[ly][threadIdx.x][1] = sumX2;
            xconv[ly][threadIdx.x][2] = sumY;
            xconv[ly][threadIdx.x][3] = sumY2;
            xconv[ly][threadIdx.x][4] = sumXY;

            int ly2 = ly + BLOCK_Y;
            if (ly2 < CONV_Y) {
                sumX = 0.f; sumX2 = 0.f; sumY = 0.f; sumY2 = 0.f; sumXY = 0.f;
#pragma unroll
                for (int d = 1; d <= HALO; ++d) {
                    float w  = cGauss[HALO - d];
                    float Xl = sTile[ly2][lx - d][0];
                    float Yl = sTile[ly2][lx - d][1];
                    float Xr = sTile[ly2][lx + d][0];
                    float Yr = sTile[ly2][lx + d][1];

                    sumX  += (Xl + Xr) * w;
                    sumX2 += (Xl * Xl + Xr * Xr) * w;
                    sumY  += (Yl + Yr) * w;
                    sumY2 += (Yl * Yl + Yr * Yr) * w;
                    sumXY += (Xl * Yl + Xr * Yr) * w;
                }
                {
                    float X = sTile[ly2][lx][0];
                    float Y = sTile[ly2][lx][1];
                    float wc = cGauss[HALO];
                    sumX  += X * wc;
                    sumX2 += (X * X) * wc;
                    sumY  += Y * wc;
                    sumY2 += (Y * Y) * wc;
                    sumXY += (X * Y) * wc;
                }
                xconv[ly2][threadIdx.x][0] = sumX;
                xconv[ly2][threadIdx.x][1] = sumX2;
                xconv[ly2][threadIdx.x][2] = sumY;
                xconv[ly2][threadIdx.x][3] = sumY2;
                xconv[ly2][threadIdx.x][4] = sumXY;
            }
        }
        block.sync();

        // 3) Vertical convolution + SSIM
        {
            int ly = threadIdx.y + HALO;
            int lx = threadIdx.x;

            float o0 = 0.f, o1 = 0.f, o2 = 0.f, o3 = 0.f, o4 = 0.f;

#pragma unroll
            for (int d = 1; d <= HALO; ++d) {
                float w = cGauss[HALO - d];
                float* top = xconv[ly - d][lx];
                float* bot = xconv[ly + d][lx];
                o0 += (top[0] + bot[0]) * w;
                o1 += (top[1] + bot[1]) * w;
                o2 += (top[2] + bot[2]) * w;
                o3 += (top[3] + bot[3]) * w;
                o4 += (top[4] + bot[4]) * w;
            }
            {
                float wC = cGauss[HALO];
                float* ctr = xconv[ly][lx];
                o0 += ctr[0] * wC;
                o1 += ctr[1] * wC;
                o2 += ctr[2] * wC;
                o3 += ctr[3] * wC;
                o4 += ctr[4] * wC;
            }

            if (pix_x < W && pix_y < H) {
                float mu1 = o0;
                float mu2 = o2;
                float mu1_sq = mu1 * mu1;
                float mu2_sq = mu2 * mu2;
                float sigma1_sq = fmaxf(0.0f, o1 - mu1_sq);
                float sigma2_sq = fmaxf(0.0f, o3 - mu2_sq);
                float sigma12   = o4 - mu1 * mu2;

                float A  = mu1_sq + mu2_sq + C1;
                float B  = sigma1_sq + sigma2_sq + C2;
                float C_ = 2.f * mu1 * mu2 + C1;
                float D_ = 2.f * sigma12 + C2;

                float val = (C_ * D_) / (A * B);
                bool clamped = (val < -1.0f || val > 1.0f);
                val = fmaxf(-1.0f, fminf(1.0f, val));

                int bhwc_idx = bIdx * num_pix * CH + pix_id * CH + c;
                ssim_map[bhwc_idx] = val;

                if (dm_dmu1) {
                    int bchw_idx = bIdx * CH * num_pix + c * num_pix + pix_id;
                    float d_mu1 = clamped ? 0.0f : (
                        (mu2 * 2.f * D_) / (A * B)
                        - (mu2 * 2.f * C_) / (A * B)
                        - (mu1 * 2.f * C_ * D_) / (A * A * B)
                        + (mu1 * 2.f * C_ * D_) / (A * B * B)
                    );
                    float d_s1  = clamped ? 0.0f : (-C_ * D_) / (A * B * B);
                    float d_s12 = clamped ? 0.0f : (2.f * C_) / (A * B);

                    dm_dmu1[bchw_idx]       = d_mu1;
                    dm_dsigma1_sq[bchw_idx] = d_s1;
                    dm_dsigma12[bchw_idx]   = d_s12;
                }
            }
        }
        block.sync();
    }
}

__global__ void fusedssim_backwardCUDA(
    int H, int W, int CH,
    float C1, float C2,
    const float* __restrict__ img1,
    const float* __restrict__ img2,
    const float* __restrict__ dL_dmap,
    float* __restrict__ dL_dimg1,
    const float* __restrict__ dm_dmu1,
    const float* __restrict__ dm_dsigma1_sq,
    const float* __restrict__ dm_dsigma12
) {
    auto block = cg::this_thread_block();
    const int pix_y  = block.group_index().y * BLOCK_Y + block.thread_index().y;
    const int pix_x  = block.group_index().x * BLOCK_X + block.thread_index().x;
    const int pix_id = pix_y * W + pix_x;
    const int num_pix = H * W;
    const int bIdx   = block.group_index().z;

    __shared__ float sData[3][SHARED_Y][SHARED_X];
    __shared__ float sScratch[CONV_Y][CONV_X][3];

    for (int c = 0; c < CH; ++c) {
        float p1 = 0.f, p2 = 0.f;
        if (pix_x < W && pix_y < H) {
            p1 = get_pix_bhwc(img1, bIdx, c, pix_y, pix_x, CH, H, W);
            p2 = get_pix_bhwc(img2, bIdx, c, pix_y, pix_x, CH, H, W);
        }

        // 1) Load tile: dL_dmap from BHWC, derivatives from BCHW (coalesced)
        {
            const int startY = block.group_index().y * BLOCK_Y;
            const int startX = block.group_index().x * BLOCK_X;

            int tid = threadIdx.y * blockDim.x + threadIdx.x;
            int warp_id  = tid / 32;
            int lane_id  = tid % 32;
            int num_warps = (BLOCK_X * BLOCK_Y + 31) / 32;

            for (int row = warp_id; row < SHARED_Y; row += num_warps) {
                int gy = startY + row - HALO;
                for (int col = lane_id; col < SHARED_X; col += 32) {
                    int gx = startX + col - HALO;

                    float chain = get_pix_bhwc(dL_dmap, bIdx, c, gy, gx, CH, H, W);
                    float vmu   = get_pix_bchw(dm_dmu1,       bIdx, c, gy, gx, CH, H, W);
                    float vs1   = get_pix_bchw(dm_dsigma1_sq, bIdx, c, gy, gx, CH, H, W);
                    float vs12  = get_pix_bchw(dm_dsigma12,   bIdx, c, gy, gx, CH, H, W);

                    sData[0][row][col] = vmu  * chain;
                    sData[1][row][col] = vs1  * chain;
                    sData[2][row][col] = vs12 * chain;
                }
            }
        }
        block.sync();

        // 2) Horizontal convolution
        {
            int ly = threadIdx.y;
            int lx = threadIdx.x + HALO;

            for (int pass = 0; pass < 2; ++pass) {
                int yy = ly + pass * BLOCK_Y;
                if (yy < CONV_Y) {
                    float a0 = 0.f, a1 = 0.f, a2 = 0.f;
#pragma unroll
                    for (int d = 1; d <= HALO; ++d) {
                        float w = cGauss[HALO - d];
                        a0 += (sData[0][yy][lx - d] + sData[0][yy][lx + d]) * w;
                        a1 += (sData[1][yy][lx - d] + sData[1][yy][lx + d]) * w;
                        a2 += (sData[2][yy][lx - d] + sData[2][yy][lx + d]) * w;
                    }
                    {
                        float wc = cGauss[HALO];
                        a0 += sData[0][yy][lx] * wc;
                        a1 += sData[1][yy][lx] * wc;
                        a2 += sData[2][yy][lx] * wc;
                    }
                    sScratch[yy][threadIdx.x][0] = a0;
                    sScratch[yy][threadIdx.x][1] = a1;
                    sScratch[yy][threadIdx.x][2] = a2;
                }
            }
        }
        block.sync();

        // 3) Vertical convolution + output in BHWC
        if (pix_x < W && pix_y < H) {
            int ly = threadIdx.y + HALO;
            int lx = threadIdx.x;

            float s0 = 0.f, s1 = 0.f, s2 = 0.f;
#pragma unroll
            for (int d = 1; d <= HALO; ++d) {
                float w = cGauss[HALO - d];
                s0 += (sScratch[ly - d][lx][0] + sScratch[ly + d][lx][0]) * w;
                s1 += (sScratch[ly - d][lx][1] + sScratch[ly + d][lx][1]) * w;
                s2 += (sScratch[ly - d][lx][2] + sScratch[ly + d][lx][2]) * w;
            }
            {
                float wc = cGauss[HALO];
                s0 += sScratch[ly][lx][0] * wc;
                s1 += sScratch[ly][lx][1] * wc;
                s2 += sScratch[ly][lx][2] * wc;
            }

            int bhwc_idx = bIdx * num_pix * CH + pix_id * CH + c;
            dL_dimg1[bhwc_idx] = s0 + (2.f * p1) * s1 + p2 * s2;
        }
        block.sync();
    }
}

template <bool need_img1, bool need_img2, bool need_img3>
__global__ void decoupled_fusedssimCUDA(
    int H, int W, int CH,
    float C1, float C2,
    const float* __restrict__ img1,
    const float* __restrict__ img2,
    const float* __restrict__ img3,
    float* __restrict__ luminance_map,
    float* __restrict__ contrast_structure_map,
    float* __restrict__ dl_dmu1,
    float* __restrict__ dl_dmu3,
    float* __restrict__ dcs_dmu1,
    float* __restrict__ dcs_dmu2,
    float* __restrict__ dcs_dsigma1_sq,
    float* __restrict__ dcs_dsigma12
) {
    auto block = cg::this_thread_block();
    const int bIdx   = block.group_index().z;
    const int pix_y  = block.group_index().y * BLOCK_Y + block.thread_index().y;
    const int pix_x  = block.group_index().x * BLOCK_X + block.thread_index().x;
    const int pix_id = pix_y * W + pix_x;
    const int num_pix = H * W;

    __shared__ float sTile[SHARED_Y][SHARED_X][3];
    __shared__ float xconv[CONV_Y][CONV_X][6];

    for (int c = 0; c < CH; ++c) {
        {
            const int tileSize = SHARED_Y * SHARED_X;
            const int threads  = BLOCK_X * BLOCK_Y;
            const int steps    = (tileSize + threads - 1) / threads;
            const int startY   = block.group_index().y * BLOCK_Y;
            const int startX   = block.group_index().x * BLOCK_X;

            for (int s = 0; s < steps; ++s) {
                int tid = s * threads + block.thread_rank();
                if (tid < tileSize) {
                    int ly = tid / SHARED_X;
                    int lx = tid % SHARED_X;
                    int gy = startY + ly - HALO;
                    int gx = startX + lx - HALO;
                    sTile[ly][lx][0] = get_pix_bhwc(img1, bIdx, c, gy, gx, CH, H, W);
                    sTile[ly][lx][1] = get_pix_bhwc(img2, bIdx, c, gy, gx, CH, H, W);
                    sTile[ly][lx][2] = get_pix_bhwc(img3, bIdx, c, gy, gx, CH, H, W);
                }
            }
        }
        block.sync();

        {
            int ly = threadIdx.y;
            int lx = threadIdx.x + HALO;

            float sumX = 0.f, sumX2 = 0.f, sumY = 0.f, sumY2 = 0.f, sumXY = 0.f, sumZ = 0.f;

#pragma unroll
            for (int d = 1; d <= HALO; ++d) {
                float w  = cGauss[HALO - d];
                float Xl = sTile[ly][lx - d][0];
                float Yl = sTile[ly][lx - d][1];
                float Zl = sTile[ly][lx - d][2];
                float Xr = sTile[ly][lx + d][0];
                float Yr = sTile[ly][lx + d][1];
                float Zr = sTile[ly][lx + d][2];

                sumX  += (Xl + Xr) * w;
                sumX2 += (Xl * Xl + Xr * Xr) * w;
                sumY  += (Yl + Yr) * w;
                sumY2 += (Yl * Yl + Yr * Yr) * w;
                sumXY += (Xl * Yl + Xr * Yr) * w;
                sumZ  += (Zl + Zr) * w;
            }
            {
                float X = sTile[ly][lx][0];
                float Y = sTile[ly][lx][1];
                float Z = sTile[ly][lx][2];
                float wc = cGauss[HALO];
                sumX  += X * wc;
                sumX2 += (X * X) * wc;
                sumY  += Y * wc;
                sumY2 += (Y * Y) * wc;
                sumXY += (X * Y) * wc;
                sumZ  += Z * wc;
            }
            xconv[ly][threadIdx.x][0] = sumX;
            xconv[ly][threadIdx.x][1] = sumX2;
            xconv[ly][threadIdx.x][2] = sumY;
            xconv[ly][threadIdx.x][3] = sumY2;
            xconv[ly][threadIdx.x][4] = sumXY;
            xconv[ly][threadIdx.x][5] = sumZ;

            int ly2 = ly + BLOCK_Y;
            if (ly2 < CONV_Y) {
                sumX = 0.f; sumX2 = 0.f; sumY = 0.f; sumY2 = 0.f; sumXY = 0.f; sumZ = 0.f;
#pragma unroll
                for (int d = 1; d <= HALO; ++d) {
                    float w  = cGauss[HALO - d];
                    float Xl = sTile[ly2][lx - d][0];
                    float Yl = sTile[ly2][lx - d][1];
                    float Zl = sTile[ly2][lx - d][2];
                    float Xr = sTile[ly2][lx + d][0];
                    float Yr = sTile[ly2][lx + d][1];
                    float Zr = sTile[ly2][lx + d][2];

                    sumX  += (Xl + Xr) * w;
                    sumX2 += (Xl * Xl + Xr * Xr) * w;
                    sumY  += (Yl + Yr) * w;
                    sumY2 += (Yl * Yl + Yr * Yr) * w;
                    sumXY += (Xl * Yl + Xr * Yr) * w;
                    sumZ  += (Zl + Zr) * w;
                }
                {
                    float X = sTile[ly2][lx][0];
                    float Y = sTile[ly2][lx][1];
                    float Z = sTile[ly2][lx][2];
                    float wc = cGauss[HALO];
                    sumX  += X * wc;
                    sumX2 += (X * X) * wc;
                    sumY  += Y * wc;
                    sumY2 += (Y * Y) * wc;
                    sumXY += (X * Y) * wc;
                    sumZ  += Z * wc;
                }
                xconv[ly2][threadIdx.x][0] = sumX;
                xconv[ly2][threadIdx.x][1] = sumX2;
                xconv[ly2][threadIdx.x][2] = sumY;
                xconv[ly2][threadIdx.x][3] = sumY2;
                xconv[ly2][threadIdx.x][4] = sumXY;
                xconv[ly2][threadIdx.x][5] = sumZ;
            }
        }
        block.sync();

        {
            int ly = threadIdx.y + HALO;
            int lx = threadIdx.x;

            float o0 = 0.f, o1 = 0.f, o2 = 0.f, o3 = 0.f, o4 = 0.f, o5 = 0.f;

#pragma unroll
            for (int d = 1; d <= HALO; ++d) {
                float w = cGauss[HALO - d];
                float* top = xconv[ly - d][lx];
                float* bot = xconv[ly + d][lx];
                o0 += (top[0] + bot[0]) * w;
                o1 += (top[1] + bot[1]) * w;
                o2 += (top[2] + bot[2]) * w;
                o3 += (top[3] + bot[3]) * w;
                o4 += (top[4] + bot[4]) * w;
                o5 += (top[5] + bot[5]) * w;
            }
            {
                float wC = cGauss[HALO];
                float* ctr = xconv[ly][lx];
                o0 += ctr[0] * wC;
                o1 += ctr[1] * wC;
                o2 += ctr[2] * wC;
                o3 += ctr[3] * wC;
                o4 += ctr[4] * wC;
                o5 += ctr[5] * wC;
            }

            if (pix_x < W && pix_y < H) {
                float mu1 = o0;
                float mu2 = o2;
                float mu3 = o5;
                float mu1_sq = mu1 * mu1;
                float mu2_sq = mu2 * mu2;
                float mu3_sq = mu3 * mu3;
                float sigma1_sq = fmaxf(0.0f, o1 - mu1_sq);
                float sigma2_sq = fmaxf(0.0f, o3 - mu2_sq);
                float sigma12   = o4 - mu1 * mu2;

                float A  = mu1_sq + mu3_sq + C1;
                float B  = sigma1_sq + sigma2_sq + C2;
                float C_ = 2.f * mu1 * mu3 + C1;
                float D_ = 2.f * sigma12 + C2;

                int bhwc_idx = bIdx * num_pix * CH + pix_id * CH + c;
                luminance_map[bhwc_idx] = C_ / A;
                contrast_structure_map[bhwc_idx] = D_ / B;

                if constexpr (need_img1 || need_img2 || need_img3) {
                    int bchw_idx = bIdx * CH * num_pix + c * num_pix + pix_id;
                    if constexpr (need_img1) {
                        dl_dmu1[bchw_idx]  = (2.f * mu3) / A - (2.f * mu1 * C_) / (A * A);
                        dcs_dmu1[bchw_idx] = (-2.f * mu2) / B + (2.f * mu1 * D_) / (B * B);
                    }
                    if constexpr (need_img2) {
                        dcs_dmu2[bchw_idx] = (-2.f * mu1) / B + (2.f * mu2 * D_) / (B * B);
                    }
                    if constexpr (need_img3) {
                        dl_dmu3[bchw_idx] = (2.f * mu1) / A - (2.f * mu3 * C_) / (A * A);
                    }
                    if constexpr (need_img1 || need_img2) {
                        dcs_dsigma1_sq[bchw_idx] = -D_ / (B * B);
                        dcs_dsigma12[bchw_idx]   = 2.f / B;
                    }
                }
            }
        }
        block.sync();
    }
}

struct DecoupledForwardArguments {
    dim3 grid;
    dim3 block;
    cudaStream_t stream;
    int H;
    int W;
    int CH;
    float C1;
    float C2;
    const float* img1;
    const float* img2;
    const float* img3;
    float* luminance_map;
    float* contrast_structure_map;
    float* dl_dmu1;
    float* dl_dmu3;
    float* dcs_dmu1;
    float* dcs_dmu2;
    float* dcs_dsigma1_sq;
    float* dcs_dsigma12;
};

template <bool need_img1, bool need_img2, bool need_img3>
void launch_decoupled_fusedssim(const DecoupledForwardArguments &arguments) {
    decoupled_fusedssimCUDA<need_img1, need_img2, need_img3>
        <<<arguments.grid, arguments.block, 0, arguments.stream>>>(
        arguments.H, arguments.W, arguments.CH, arguments.C1, arguments.C2,
        arguments.img1, arguments.img2, arguments.img3,
        arguments.luminance_map, arguments.contrast_structure_map,
        arguments.dl_dmu1, arguments.dl_dmu3,
        arguments.dcs_dmu1, arguments.dcs_dmu2,
        arguments.dcs_dsigma1_sq, arguments.dcs_dsigma12
    );
}

template <bool use_luminance, bool use_contrast>
__global__ void decoupled_fusedssim_backwardCUDA(
    int H, int W, int CH, int gradient_padding,
    const float* __restrict__ img1,
    const float* __restrict__ img2,
    const float* __restrict__ dL_dluminance_map,
    const float* __restrict__ dL_dcontrast_structure_map,
    float* __restrict__ dL_dimg1,
    const float* __restrict__ dl_dmu1,
    const float* __restrict__ dcs_dmu1,
    const float* __restrict__ dcs_dsigma1_sq,
    const float* __restrict__ dcs_dsigma12
) {
    auto block = cg::this_thread_block();
    const int pix_y  = block.group_index().y * BLOCK_Y + block.thread_index().y;
    const int pix_x  = block.group_index().x * BLOCK_X + block.thread_index().x;
    const int pix_id = pix_y * W + pix_x;
    const int num_pix = H * W;
    const int bIdx   = block.group_index().z;

    __shared__ float sData[use_contrast ? 3 : 1][SHARED_Y][SHARED_X];
    __shared__ float sScratch[CONV_Y][CONV_X][use_contrast ? 3 : 1];

    for (int c = 0; c < CH; ++c) {
        float p1 = 0.f, p2 = 0.f;
        if constexpr (use_contrast) {
            if (pix_x < W && pix_y < H) {
                p1 = get_pix_bhwc(img1, bIdx, c, pix_y, pix_x, CH, H, W);
                p2 = get_pix_bhwc(img2, bIdx, c, pix_y, pix_x, CH, H, W);
            }
        }

        {
            const int startY = block.group_index().y * BLOCK_Y;
            const int startX = block.group_index().x * BLOCK_X;

            int tid = threadIdx.y * blockDim.x + threadIdx.x;
            int warp_id  = tid / 32;
            int lane_id  = tid % 32;
            int num_warps = (BLOCK_X * BLOCK_Y + 31) / 32;

            for (int row = warp_id; row < SHARED_Y; row += num_warps) {
                int gy = startY + row - HALO;
                for (int col = lane_id; col < SHARED_X; col += 32) {
                    int gx = startX + col - HALO;

                    float first_order = 0.f;
                    if constexpr (use_luminance) {
                        const float chain_l = get_grad_bhwc(
                            dL_dluminance_map, bIdx, c, gy, gx, CH, H, W, gradient_padding
                        );
                        first_order = chain_l * get_pix_bchw(dl_dmu1, bIdx, c, gy, gx, CH, H, W);
                    }
                    if constexpr (use_contrast) {
                        const float chain_cs = get_grad_bhwc(
                            dL_dcontrast_structure_map, bIdx, c, gy, gx, CH, H, W, gradient_padding
                        );
                        first_order += chain_cs * get_pix_bchw(dcs_dmu1, bIdx, c, gy, gx, CH, H, W);
                        sData[1][row][col] = chain_cs * get_pix_bchw(
                            dcs_dsigma1_sq, bIdx, c, gy, gx, CH, H, W
                        );
                        sData[2][row][col] = chain_cs * get_pix_bchw(
                            dcs_dsigma12, bIdx, c, gy, gx, CH, H, W
                        );
                    }
                    sData[0][row][col] = first_order;
                }
            }
        }
        block.sync();

        {
            int ly = threadIdx.y;
            int lx = threadIdx.x + HALO;

            for (int pass = 0; pass < 2; ++pass) {
                int yy = ly + pass * BLOCK_Y;
                if (yy < CONV_Y) {
                    float a0 = 0.f;
                    float a1 = 0.f;
                    float a2 = 0.f;
#pragma unroll
                    for (int d = 1; d <= HALO; ++d) {
                        float w = cGauss[HALO - d];
                        a0 += (sData[0][yy][lx - d] + sData[0][yy][lx + d]) * w;
                        if constexpr (use_contrast) {
                            a1 += (sData[1][yy][lx - d] + sData[1][yy][lx + d]) * w;
                            a2 += (sData[2][yy][lx - d] + sData[2][yy][lx + d]) * w;
                        }
                    }
                    {
                        float wc = cGauss[HALO];
                        a0 += sData[0][yy][lx] * wc;
                        if constexpr (use_contrast) {
                            a1 += sData[1][yy][lx] * wc;
                            a2 += sData[2][yy][lx] * wc;
                        }
                    }
                    sScratch[yy][threadIdx.x][0] = a0;
                    if constexpr (use_contrast) {
                        sScratch[yy][threadIdx.x][1] = a1;
                        sScratch[yy][threadIdx.x][2] = a2;
                    }
                }
            }
        }
        block.sync();

        if (pix_x < W && pix_y < H) {
            int ly = threadIdx.y + HALO;
            int lx = threadIdx.x;

            float s0 = 0.f;
            float s1 = 0.f;
            float s2 = 0.f;
#pragma unroll
            for (int d = 1; d <= HALO; ++d) {
                float w = cGauss[HALO - d];
                s0 += (sScratch[ly - d][lx][0] + sScratch[ly + d][lx][0]) * w;
                if constexpr (use_contrast) {
                    s1 += (sScratch[ly - d][lx][1] + sScratch[ly + d][lx][1]) * w;
                    s2 += (sScratch[ly - d][lx][2] + sScratch[ly + d][lx][2]) * w;
                }
            }
            {
                float wc = cGauss[HALO];
                s0 += sScratch[ly][lx][0] * wc;
                if constexpr (use_contrast) {
                    s1 += sScratch[ly][lx][1] * wc;
                    s2 += sScratch[ly][lx][2] * wc;
                }
            }

            int bhwc_idx = bIdx * num_pix * CH + pix_id * CH + c;
            float gradient = s0;
            if constexpr (use_contrast) {
                gradient += (2.f * p1) * s1 + p2 * s2;
            }
            dL_dimg1[bhwc_idx] = gradient;
        }
        block.sync();
    }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
fusedssim(
    float C1, float C2,
    torch::Tensor &img1, torch::Tensor &img2,
    bool train
) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(img1));
    auto stream = at::cuda::getCurrentCUDAStream();
    int B  = img1.size(0);
    int H  = img1.size(1);
    int W  = img1.size(2);
    int CH = img1.size(3);

    dim3 grid((W + BLOCK_X - 1) / BLOCK_X,
              (H + BLOCK_Y - 1) / BLOCK_Y, B);
    dim3 block(BLOCK_X, BLOCK_Y);

    auto ssim_map = torch::empty_like(img1);

    auto dm_dmu1       = train ? torch::empty({B, CH, H, W}, img1.options()) : torch::empty({0}, img1.options());
    auto dm_dsigma1_sq = train ? torch::empty({B, CH, H, W}, img1.options()) : torch::empty({0}, img1.options());
    auto dm_dsigma12   = train ? torch::empty({B, CH, H, W}, img1.options()) : torch::empty({0}, img1.options());

    fusedssimCUDA<<<grid, block, 0, stream>>>(
        H, W, CH, C1, C2,
        img1.data_ptr<float>(),
        img2.data_ptr<float>(),
        ssim_map.data_ptr<float>(),
        train ? dm_dmu1.data_ptr<float>()       : nullptr,
        train ? dm_dsigma1_sq.data_ptr<float>() : nullptr,
        train ? dm_dsigma12.data_ptr<float>()   : nullptr
    );

    return std::make_tuple(ssim_map, dm_dmu1, dm_dsigma1_sq, dm_dsigma12);
}

torch::Tensor
fusedssim_backward(
    float C1, float C2,
    torch::Tensor &img1, torch::Tensor &img2,
    torch::Tensor &dL_dmap,
    torch::Tensor &dm_dmu1,
    torch::Tensor &dm_dsigma1_sq,
    torch::Tensor &dm_dsigma12
) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(img1));
    auto stream = at::cuda::getCurrentCUDAStream();
    int B  = img1.size(0);
    int H  = img1.size(1);
    int W  = img1.size(2);
    int CH = img1.size(3);

    auto dL_dimg1 = torch::empty_like(img1);

    dim3 grid((W + BLOCK_X - 1) / BLOCK_X,
              (H + BLOCK_Y - 1) / BLOCK_Y, B);
    dim3 block(BLOCK_X, BLOCK_Y);

    fusedssim_backwardCUDA<<<grid, block, 0, stream>>>(
        H, W, CH, C1, C2,
        img1.contiguous().data_ptr<float>(),
        img2.contiguous().data_ptr<float>(),
        dL_dmap.contiguous().data_ptr<float>(),
        dL_dimg1.data_ptr<float>(),
        dm_dmu1.contiguous().data_ptr<float>(),
        dm_dsigma1_sq.contiguous().data_ptr<float>(),
        dm_dsigma12.contiguous().data_ptr<float>()
    );

    return dL_dimg1;
}

std::tuple<
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
decoupled_fusedssim(
    float C1, float C2,
    torch::Tensor &img1, torch::Tensor &img2, torch::Tensor &img3,
    bool need_img1, bool need_img2, bool need_img3
) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(img1));
    auto stream = at::cuda::getCurrentCUDAStream();
    int B  = img1.size(0);
    int H  = img1.size(1);
    int W  = img1.size(2);
    int CH = img1.size(3);

    dim3 grid((W + BLOCK_X - 1) / BLOCK_X,
              (H + BLOCK_Y - 1) / BLOCK_Y, B);
    dim3 block(BLOCK_X, BLOCK_Y);

    auto luminance_map = torch::empty_like(img1);
    auto contrast_structure_map = torch::empty_like(img1);

    auto derivative_shape = std::vector<int64_t>{B, CH, H, W};
    auto empty            = torch::empty({0}, img1.options());
    const bool need_contrast = need_img1 || need_img2;
    auto dl_dmu1          = need_img1 ? torch::empty(derivative_shape, img1.options()) : empty;
    auto dl_dmu3          = need_img3 ? torch::empty(derivative_shape, img1.options()) : empty;
    auto dcs_dmu1         = need_img1 ? torch::empty(derivative_shape, img1.options()) : empty;
    auto dcs_dmu2         = need_img2 ? torch::empty(derivative_shape, img1.options()) : empty;
    auto dcs_dsigma1_sq   = need_contrast ? torch::empty(derivative_shape, img1.options()) : empty;
    auto dcs_dsigma12     = need_contrast ? torch::empty(derivative_shape, img1.options()) : empty;

    const int derivative_mask =
        (need_img1 ? 1 : 0) |
        (need_img2 ? 2 : 0) |
        (need_img3 ? 4 : 0);
    const DecoupledForwardArguments arguments{
        grid,
        block,
        stream,
        H,
        W,
        CH,
        C1,
        C2,
        img1.data_ptr<float>(),
        img2.data_ptr<float>(),
        img3.data_ptr<float>(),
        luminance_map.data_ptr<float>(),
        contrast_structure_map.data_ptr<float>(),
        need_img1 ? dl_dmu1.data_ptr<float>() : nullptr,
        need_img3 ? dl_dmu3.data_ptr<float>() : nullptr,
        need_img1 ? dcs_dmu1.data_ptr<float>() : nullptr,
        need_img2 ? dcs_dmu2.data_ptr<float>() : nullptr,
        need_contrast ? dcs_dsigma1_sq.data_ptr<float>() : nullptr,
        need_contrast ? dcs_dsigma12.data_ptr<float>() : nullptr,
    };
    switch (derivative_mask) {
        case 0:
            launch_decoupled_fusedssim<false, false, false>(arguments);
            break;
        case 1:
            launch_decoupled_fusedssim<true, false, false>(arguments);
            break;
        case 2:
            launch_decoupled_fusedssim<false, true, false>(arguments);
            break;
        case 3:
            launch_decoupled_fusedssim<true, true, false>(arguments);
            break;
        case 4:
            launch_decoupled_fusedssim<false, false, true>(arguments);
            break;
        case 5:
            launch_decoupled_fusedssim<true, false, true>(arguments);
            break;
        case 6:
            launch_decoupled_fusedssim<false, true, true>(arguments);
            break;
        case 7:
            launch_decoupled_fusedssim<true, true, true>(arguments);
            break;
    }

    return std::make_tuple(
        luminance_map,
        contrast_structure_map,
        dl_dmu1,
        dl_dmu3,
        dcs_dmu1,
        dcs_dmu2,
        dcs_dsigma1_sq,
        dcs_dsigma12
    );
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
decoupled_fusedssim_backward(
    float C1,
    float C2,
    torch::Tensor &img1,
    torch::Tensor &img2,
    torch::Tensor &img3,
    torch::Tensor &dL_dluminance_map,
    torch::Tensor &dL_dcontrast_structure_map,
    torch::Tensor &dl_dmu1,
    torch::Tensor &dl_dmu3,
    torch::Tensor &dcs_dmu1,
    torch::Tensor &dcs_dmu2,
    torch::Tensor &dcs_dsigma1_sq,
    torch::Tensor &dcs_dsigma12,
    int gradient_padding
) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(img1));
    auto stream = at::cuda::getCurrentCUDAStream();
    int B  = img1.size(0);
    int H  = img1.size(1);
    int W  = img1.size(2);
    int CH = img1.size(3);

    const bool need_img1 = (dl_dmu1.numel() != 0) || (dcs_dmu1.numel() != 0);
    const bool need_img2 = dcs_dmu2.numel() != 0;
    const bool need_img3 = dl_dmu3.numel() != 0;

    auto dL_dimg1 = need_img1 ? torch::empty_like(img1) : torch::empty({0}, img1.options());
    auto dL_dimg2 = need_img2 ? torch::empty_like(img2) : torch::empty({0}, img2.options());
    auto dL_dimg3 = need_img3 ? torch::empty_like(img3) : torch::empty({0}, img3.options());

    dim3 grid((W + BLOCK_X - 1) / BLOCK_X,
              (H + BLOCK_Y - 1) / BLOCK_Y, B);
    dim3 block(BLOCK_X, BLOCK_Y);

    if (need_img1) {
        decoupled_fusedssim_backwardCUDA<true, true><<<grid, block, 0, stream>>>(
            H, W, CH, gradient_padding,
            img1.contiguous().data_ptr<float>(),
            img2.contiguous().data_ptr<float>(),
            dL_dluminance_map.contiguous().data_ptr<float>(),
            dL_dcontrast_structure_map.contiguous().data_ptr<float>(),
            dL_dimg1.data_ptr<float>(),
            dl_dmu1.contiguous().data_ptr<float>(),
            dcs_dmu1.contiguous().data_ptr<float>(),
            dcs_dsigma1_sq.contiguous().data_ptr<float>(),
            dcs_dsigma12.contiguous().data_ptr<float>()
        );
    }

    if (need_img2) {
        decoupled_fusedssim_backwardCUDA<false, true><<<grid, block, 0, stream>>>(
            H, W, CH, gradient_padding,
            img2.contiguous().data_ptr<float>(),
            img1.contiguous().data_ptr<float>(),
            nullptr,
            dL_dcontrast_structure_map.contiguous().data_ptr<float>(),
            dL_dimg2.data_ptr<float>(),
            nullptr,
            dcs_dmu2.contiguous().data_ptr<float>(),
            dcs_dsigma1_sq.contiguous().data_ptr<float>(),
            dcs_dsigma12.contiguous().data_ptr<float>()
        );
    }

    if (need_img3) {
        decoupled_fusedssim_backwardCUDA<true, false><<<grid, block, 0, stream>>>(
            H, W, CH, gradient_padding,
            img3.contiguous().data_ptr<float>(),
            nullptr,
            dL_dluminance_map.contiguous().data_ptr<float>(),
            nullptr,
            dL_dimg3.data_ptr<float>(),
            dl_dmu3.contiguous().data_ptr<float>(),
            nullptr,
            nullptr,
            nullptr
        );
    }

    return std::make_tuple(dL_dimg1, dL_dimg2, dL_dimg3);
}
