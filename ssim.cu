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

template<int CH>
__global__ void fusedssimCUDA(
    int H, int W,
    float C1, float C2,
    const float* __restrict__ img1,
    const float* __restrict__ img2,
    float* __restrict__ ssim_map,
    float* __restrict__ dm_dmu1,
    float* __restrict__ dm_dsigma1_sq,
    float* __restrict__ dm_dsigma12
) {
    auto block = cg::this_thread_block();
    const int bIdx  = block.group_index().z;
    const int pix_y = block.group_index().y * BLOCK_Y + block.thread_index().y;
    const int pix_x = block.group_index().x * BLOCK_X + block.thread_index().x;
    const int pix_id = pix_y * W + pix_x;
    const int num_pix = H * W;

    __shared__ float sTile[SHARED_Y][SHARED_X][2][CH];
    __shared__ float xconv[CONV_Y][CONV_X][5][CH];

    // 1) Load tile + halo — all channels coalesced
    {
        const int tileSize = SHARED_Y * SHARED_X;
        const int threads  = BLOCK_X * BLOCK_Y;
        const int steps    = (tileSize + threads - 1) / threads;
        const int startY   = block.group_index().y * BLOCK_Y;
        const int startX   = block.group_index().x * BLOCK_X;
        const float* base1 = img1 + bIdx * num_pix * CH;
        const float* base2 = img2 + bIdx * num_pix * CH;

        for (int s = 0; s < steps; ++s) {
            int tid = s * threads + block.thread_rank();
            if (tid < tileSize) {
                int ly = tid / SHARED_X;
                int lx = tid % SHARED_X;
                int gy = startY + ly - HALO;
                int gx = startX + lx - HALO;
                bool valid = (gx >= 0 && gx < W && gy >= 0 && gy < H);
                int idx = gy * W * CH + gx * CH;

                #pragma unroll
                for (int c = 0; c < CH; ++c) {
                    sTile[ly][lx][0][c] = valid ? base1[idx + c] : 0.0f;
                    sTile[ly][lx][1][c] = valid ? base2[idx + c] : 0.0f;
                }
            }
        }
    }
    block.sync();

    // 2) Horizontal convolution — channel innermost
    {
        int ly = threadIdx.y;
        int lx = threadIdx.x + HALO;

        float sumX[CH], sumX2[CH], sumY[CH], sumY2[CH], sumXY[CH];
        #pragma unroll
        for (int c = 0; c < CH; ++c) {
            sumX[c] = 0.f; sumX2[c] = 0.f;
            sumY[c] = 0.f; sumY2[c] = 0.f;
            sumXY[c] = 0.f;
        }

        #pragma unroll
        for (int d = 1; d <= HALO; ++d) {
            float w = cGauss[HALO - d];
            #pragma unroll
            for (int c = 0; c < CH; ++c) {
                float Xl = sTile[ly][lx - d][0][c];
                float Yl = sTile[ly][lx - d][1][c];
                float Xr = sTile[ly][lx + d][0][c];
                float Yr = sTile[ly][lx + d][1][c];
                sumX[c]  += (Xl + Xr) * w;
                sumX2[c] += (Xl*Xl + Xr*Xr) * w;
                sumY[c]  += (Yl + Yr) * w;
                sumY2[c] += (Yl*Yl + Yr*Yr) * w;
                sumXY[c] += (Xl*Yl + Xr*Yr) * w;
            }
        }
        {
            float wc = cGauss[HALO];
            #pragma unroll
            for (int c = 0; c < CH; ++c) {
                float X = sTile[ly][lx][0][c];
                float Y = sTile[ly][lx][1][c];
                sumX[c]  += X * wc;
                sumX2[c] += X*X * wc;
                sumY[c]  += Y * wc;
                sumY2[c] += Y*Y * wc;
                sumXY[c] += X*Y * wc;
            }
        }

        #pragma unroll
        for (int c = 0; c < CH; ++c) {
            xconv[ly][threadIdx.x][0][c] = sumX[c];
            xconv[ly][threadIdx.x][1][c] = sumX2[c];
            xconv[ly][threadIdx.x][2][c] = sumY[c];
            xconv[ly][threadIdx.x][3][c] = sumY2[c];
            xconv[ly][threadIdx.x][4][c] = sumXY[c];
        }

        int ly2 = ly + BLOCK_Y;
        if (ly2 < CONV_Y) {
            #pragma unroll
            for (int c = 0; c < CH; ++c) {
                sumX[c] = 0.f; sumX2[c] = 0.f;
                sumY[c] = 0.f; sumY2[c] = 0.f;
                sumXY[c] = 0.f;
            }
            #pragma unroll
            for (int d = 1; d <= HALO; ++d) {
                float w = cGauss[HALO - d];
                #pragma unroll
                for (int c = 0; c < CH; ++c) {
                    float Xl = sTile[ly2][lx - d][0][c];
                    float Yl = sTile[ly2][lx - d][1][c];
                    float Xr = sTile[ly2][lx + d][0][c];
                    float Yr = sTile[ly2][lx + d][1][c];
                    sumX[c]  += (Xl + Xr) * w;
                    sumX2[c] += (Xl*Xl + Xr*Xr) * w;
                    sumY[c]  += (Yl + Yr) * w;
                    sumY2[c] += (Yl*Yl + Yr*Yr) * w;
                    sumXY[c] += (Xl*Yl + Xr*Yr) * w;
                }
            }
            {
                float wc = cGauss[HALO];
                #pragma unroll
                for (int c = 0; c < CH; ++c) {
                    float X = sTile[ly2][lx][0][c];
                    float Y = sTile[ly2][lx][1][c];
                    sumX[c]  += X * wc;
                    sumX2[c] += X*X * wc;
                    sumY[c]  += Y * wc;
                    sumY2[c] += Y*Y * wc;
                    sumXY[c] += X*Y * wc;
                }
            }
            #pragma unroll
            for (int c = 0; c < CH; ++c) {
                xconv[ly2][threadIdx.x][0][c] = sumX[c];
                xconv[ly2][threadIdx.x][1][c] = sumX2[c];
                xconv[ly2][threadIdx.x][2][c] = sumY[c];
                xconv[ly2][threadIdx.x][3][c] = sumY2[c];
                xconv[ly2][threadIdx.x][4][c] = sumXY[c];
            }
        }
    }
    block.sync();

    // 3) Vertical convolution + SSIM — channel innermost
    {
        int ly = threadIdx.y + HALO;
        int lx = threadIdx.x;

        float out[5][CH];
        #pragma unroll
        for (int c = 0; c < CH; ++c) {
            out[0][c] = 0.f; out[1][c] = 0.f; out[2][c] = 0.f;
            out[3][c] = 0.f; out[4][c] = 0.f;
        }

        #pragma unroll
        for (int d = 1; d <= HALO; ++d) {
            float w = cGauss[HALO - d];
            #pragma unroll
            for (int c = 0; c < CH; ++c) {
                #pragma unroll
                for (int k = 0; k < 5; ++k)
                    out[k][c] += (xconv[ly - d][lx][k][c] + xconv[ly + d][lx][k][c]) * w;
            }
        }
        {
            float wC = cGauss[HALO];
            #pragma unroll
            for (int c = 0; c < CH; ++c) {
                #pragma unroll
                for (int k = 0; k < 5; ++k)
                    out[k][c] += xconv[ly][lx][k][c] * wC;
            }
        }

        if (pix_x < W && pix_y < H) {
            int base_idx = bIdx * num_pix * CH + pix_id * CH;

            #pragma unroll
            for (int c = 0; c < CH; ++c) {
                float mu1 = out[0][c];
                float mu2 = out[2][c];
                float mu1_sq = mu1 * mu1;
                float mu2_sq = mu2 * mu2;
                float sigma1_sq = fmaxf(0.0f, out[1][c] - mu1_sq);
                float sigma2_sq = fmaxf(0.0f, out[3][c] - mu2_sq);
                float sigma12   = out[4][c] - mu1 * mu2;

                float A  = mu1_sq + mu2_sq + C1;
                float B  = sigma1_sq + sigma2_sq + C2;
                float C_ = 2.f * mu1 * mu2 + C1;
                float D_ = 2.f * sigma12 + C2;

                float val = (C_ * D_) / (A * B);
                bool clamped = (val < -1.0f || val > 1.0f);
                val = fmaxf(-1.0f, fminf(1.0f, val));

                ssim_map[base_idx + c] = val;

                if (dm_dmu1) {
                    float d_mu1 = clamped ? 0.0f : (
                        (mu2 * 2.f * D_) / (A * B)
                        - (mu2 * 2.f * C_) / (A * B)
                        - (mu1 * 2.f * C_ * D_) / (A * A * B)
                        + (mu1 * 2.f * C_ * D_) / (A * B * B)
                    );
                    float d_s1  = clamped ? 0.0f : (-C_ * D_) / (A * B * B);
                    float d_s12 = clamped ? 0.0f : (2.f * C_) / (A * B);

                    dm_dmu1[base_idx + c]       = d_mu1;
                    dm_dsigma1_sq[base_idx + c] = d_s1;
                    dm_dsigma12[base_idx + c]   = d_s12;
                }
            }
        }
    }
}

template<int CH>
__global__ void fusedssim_backwardCUDA(
    int H, int W,
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

    __shared__ float sData[SHARED_Y][SHARED_X][3][CH];
    __shared__ float sScratch[CONV_Y][CONV_X][3][CH];

    float p1[CH], p2[CH];
    if (pix_x < W && pix_y < H) {
        const float* b1 = img1 + bIdx * num_pix * CH + pix_id * CH;
        const float* b2 = img2 + bIdx * num_pix * CH + pix_id * CH;
        #pragma unroll
        for (int c = 0; c < CH; ++c) { p1[c] = b1[c]; p2[c] = b2[c]; }
    } else {
        #pragma unroll
        for (int c = 0; c < CH; ++c) { p1[c] = 0.f; p2[c] = 0.f; }
    }

    // 1) Load + fuse multiplication — all channels coalesced
    {
        const int startY = block.group_index().y * BLOCK_Y;
        const int startX = block.group_index().x * BLOCK_X;
        int tid = threadIdx.y * blockDim.x + threadIdx.x;
        int warp_id = tid / 32;
        int lane_id = tid % 32;
        int totalThreads = BLOCK_X * BLOCK_Y;
        int num_warps = (totalThreads + 31) / 32;

        const float* base_dL  = dL_dmap       + bIdx * num_pix * CH;
        const float* base_mu  = dm_dmu1       + bIdx * num_pix * CH;
        const float* base_s1  = dm_dsigma1_sq + bIdx * num_pix * CH;
        const float* base_s12 = dm_dsigma12   + bIdx * num_pix * CH;

        for (int row = warp_id; row < SHARED_Y; row += num_warps) {
            int gy = startY + row - HALO;
            for (int col = lane_id; col < SHARED_X; col += 32) {
                int gx = startX + col - HALO;
                bool valid = (gx >= 0 && gx < W && gy >= 0 && gy < H);
                int idx = gy * W * CH + gx * CH;

                #pragma unroll
                for (int c = 0; c < CH; ++c) {
                    float chain = valid ? base_dL[idx + c]  : 0.0f;
                    float vmu   = valid ? base_mu[idx + c]  : 0.0f;
                    float vs1   = valid ? base_s1[idx + c]  : 0.0f;
                    float vs12  = valid ? base_s12[idx + c] : 0.0f;
                    sData[row][col][0][c] = vmu  * chain;
                    sData[row][col][1][c] = vs1  * chain;
                    sData[row][col][2][c] = vs12 * chain;
                }
            }
        }
    }
    block.sync();

    // 2) Horizontal pass — channel innermost
    {
        int ly = threadIdx.y;
        int lx = threadIdx.x + HALO;

        for (int pass = 0; pass < 2; ++pass) {
            int yy = ly + pass * BLOCK_Y;
            if (yy < CONV_Y) {
                float accum[3][CH];
                #pragma unroll
                for (int c = 0; c < CH; ++c) {
                    accum[0][c] = 0.f; accum[1][c] = 0.f; accum[2][c] = 0.f;
                }

                #pragma unroll
                for (int d = 1; d <= HALO; ++d) {
                    float w = cGauss[HALO - d];
                    #pragma unroll
                    for (int c = 0; c < CH; ++c) {
                        #pragma unroll
                        for (int k = 0; k < 3; ++k)
                            accum[k][c] += (sData[yy][lx - d][k][c] + sData[yy][lx + d][k][c]) * w;
                    }
                }
                {
                    float wc = cGauss[HALO];
                    #pragma unroll
                    for (int c = 0; c < CH; ++c) {
                        #pragma unroll
                        for (int k = 0; k < 3; ++k)
                            accum[k][c] += sData[yy][lx][k][c] * wc;
                    }
                }

                #pragma unroll
                for (int c = 0; c < CH; ++c) {
                    sScratch[yy][threadIdx.x][0][c] = accum[0][c];
                    sScratch[yy][threadIdx.x][1][c] = accum[1][c];
                    sScratch[yy][threadIdx.x][2][c] = accum[2][c];
                }
            }
        }
    }
    block.sync();

    // 3) Vertical pass + output — channel innermost
    if (pix_x < W && pix_y < H) {
        int ly = threadIdx.y + HALO;
        int lx = threadIdx.x;

        float sum[3][CH];
        #pragma unroll
        for (int c = 0; c < CH; ++c) {
            sum[0][c] = 0.f; sum[1][c] = 0.f; sum[2][c] = 0.f;
        }

        #pragma unroll
        for (int d = 1; d <= HALO; ++d) {
            float w = cGauss[HALO - d];
            #pragma unroll
            for (int c = 0; c < CH; ++c) {
                #pragma unroll
                for (int k = 0; k < 3; ++k)
                    sum[k][c] += (sScratch[ly - d][lx][k][c] + sScratch[ly + d][lx][k][c]) * w;
            }
        }
        {
            float wc = cGauss[HALO];
            #pragma unroll
            for (int c = 0; c < CH; ++c) {
                #pragma unroll
                for (int k = 0; k < 3; ++k)
                    sum[k][c] += sScratch[ly][lx][k][c] * wc;
            }
        }

        int base_idx = bIdx * num_pix * CH + pix_id * CH;
        #pragma unroll
        for (int c = 0; c < CH; ++c) {
            dL_dimg1[base_idx + c] = sum[0][c] + (2.f * p1[c]) * sum[1][c] + p2[c] * sum[2][c];
        }
    }
}

// Dispatch helper
#define LAUNCH_FWD(CH_VAL) \
    fusedssimCUDA<CH_VAL><<<grid, block, 0, stream>>>( \
        H, W, C1, C2, \
        img1.contiguous().data_ptr<float>(), \
        img2.contiguous().data_ptr<float>(), \
        ssim_map.data_ptr<float>(), \
        train ? dm_dmu1.data_ptr<float>()       : nullptr, \
        train ? dm_dsigma1_sq.data_ptr<float>() : nullptr, \
        train ? dm_dsigma12.data_ptr<float>()   : nullptr)

#define LAUNCH_BWD(CH_VAL) \
    fusedssim_backwardCUDA<CH_VAL><<<grid, block, 0, stream>>>( \
        H, W, C1, C2, \
        img1.contiguous().data_ptr<float>(), \
        img2.contiguous().data_ptr<float>(), \
        dL_dmap.contiguous().data_ptr<float>(), \
        dL_dimg1.data_ptr<float>(), \
        dm_dmu1.contiguous().data_ptr<float>(), \
        dm_dsigma1_sq.contiguous().data_ptr<float>(), \
        dm_dsigma12.contiguous().data_ptr<float>())

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

    auto ssim_map      = torch::zeros_like(img1).contiguous();
    auto dm_dmu1       = train ? torch::zeros_like(img1) : torch::empty({0}, img1.options());
    auto dm_dsigma1_sq = train ? torch::zeros_like(img1) : torch::empty({0}, img1.options());
    auto dm_dsigma12   = train ? torch::zeros_like(img1) : torch::empty({0}, img1.options());

    switch (CH) {
        case 1: LAUNCH_FWD(1); break;
        case 3: LAUNCH_FWD(3); break;
        case 4: LAUNCH_FWD(4); break;
        default: TORCH_CHECK(false, "fused_ssim_bhwc: unsupported channel count ", CH);
    }

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

    auto dL_dimg1 = torch::zeros_like(img1);

    dim3 grid((W + BLOCK_X - 1) / BLOCK_X,
              (H + BLOCK_Y - 1) / BLOCK_Y, B);
    dim3 block(BLOCK_X, BLOCK_Y);

    switch (CH) {
        case 1: LAUNCH_BWD(1); break;
        case 3: LAUNCH_BWD(3); break;
        case 4: LAUNCH_BWD(4); break;
        default: TORCH_CHECK(false, "fused_ssim_bhwc: unsupported channel count ", CH);
    }

    return dL_dimg1;
}
