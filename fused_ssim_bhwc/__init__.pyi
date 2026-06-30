from typing import Literal

import torch

Padding = Literal["same", "valid"]

def fused_ssim(
  img1: torch.Tensor,
  img2: torch.Tensor,
  padding: Padding = "same",
  train: bool = True,
) -> torch.Tensor:
  ...

def fused_ssim_map(
  img1: torch.Tensor,
  img2: torch.Tensor,
  padding: Padding = "same",
  train: bool = True,
) -> torch.Tensor:
  ...

def decoupled_fused_ssim(
  img1: torch.Tensor,
  img2: torch.Tensor,
  img3: torch.Tensor,
  padding: Padding = "same",
  train: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
  ...
