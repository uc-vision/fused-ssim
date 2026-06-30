#include <torch/extension.h>
#include "ssim.h"

// Only include 3D SSIM for CUDA builds
#ifdef FUSED_SSIM_CUDA
#include "ssim3d.h"
#endif

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // Release the GIL — kernel dispatch doesn't touch Python objects, so
  // multi-threaded callers can overlap their dispatches.
  using pybind11::call_guard;
  using pybind11::gil_scoped_release;

  // 2D SSIM (available on all backends)
  m.def("fusedssim", &fusedssim, call_guard<gil_scoped_release>());
  m.def("fusedssim_backward", &fusedssim_backward, call_guard<gil_scoped_release>());
  m.def("decoupled_fusedssim", &decoupled_fusedssim, call_guard<gil_scoped_release>());
  m.def("decoupled_fusedssim_backward", &decoupled_fusedssim_backward, call_guard<gil_scoped_release>());

  // 3D SSIM (CUDA only for now)
#ifdef FUSED_SSIM_CUDA
  m.def("fusedssim3d", &fusedssim3d, call_guard<gil_scoped_release>());
  m.def("fusedssim_backward3d", &fusedssim_backward3d, call_guard<gil_scoped_release>());
#endif
}
