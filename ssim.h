#pragma once
#include <torch/extension.h>
#include <cstdio>
#include <tuple>
#include <string>

std::tuple<torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor>
fusedssim(
    float C1,
    float C2,
    torch::Tensor &img1,
    torch::Tensor &img2,
    bool train
);

torch::Tensor
fusedssim_backward(
    float C1,
    float C2,
    torch::Tensor &img1,
    torch::Tensor &img2,
    torch::Tensor &dL_dmap,
    torch::Tensor &dm_dmu1,
    torch::Tensor &dm_dsigma1_sq,
    torch::Tensor &dm_dsigma12
);

std::tuple<
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
decoupled_fusedssim(
    float C1,
    float C2,
    torch::Tensor &img1,
    torch::Tensor &img2,
    torch::Tensor &img3,
    bool train
);

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
    torch::Tensor &dcs_dsigma12
);
