#include <iostream>
#include <stdio.h>
#include <vector>

#include <adiak.hpp>
#include <caliper/cali-manager.h>
#include <caliper/cali.h>




/**
 * @brief Implemenation using global memory for image and constant memory for
 * filter
 * @param a input image
 * @param b output image
 * @param nx image width
 * @param nx image length
 */
__global__ void filter_constant(unsigned char *a, unsigned char *b, int nx,
                                int ny) {}

/**
 * @brief Implemenation using global memory fir filter and image
 * @param a input image
 * @param b output image
 * @param c filter
 * @param nx image width
 * @param nx image length
 */
__global__ void filter_global(unsigned char *a, unsigned char *b, int nx,
                              int ny, float *c) {}

/**
 * @brief CPU implementation for the filter
 * @param a input image
 * @param b output image
 * @param c filter
 * @param nx image width
 * @param ny image length
 */
void filter_CPU(const std::vector<unsigned char> &a,
                std::vector<unsigned char> &b, int nx, int ny,
                const std::vector<float> &c) {
  auto idx = [&nx](int y, int x) { return y * nx + x; };

  for (int y = 0; y < ny; ++y) {
    for (int x = 0; x < nx; ++x) {
      int xl = std::max(0, x - 1);
      int yl = std::max(0, y - 1);
      int xh = std::min(nx - 1, x + 1);
      int yh = std::min(ny - 1, y + 1);

      float v =
          c[0] * a[idx(yl, xl)] + c[1] * a[idx(yl, x)] + c[2] * a[idx(yl, xh)] +
          c[3] * a[idx(y, xl)] + c[4] * a[idx(y, x)] + c[5] * a[idx(y, xh)] +
          c[6] * a[idx(yh, xl)] + c[7] * a[idx(yh, x)] + c[8] * a[idx(yh, xh)];

      uint f = (uint)(v + 0.5f);
      b[idx(y, x)] =
          (unsigned char)std::min(255, std::max(0, static_cast<int>(f)));
    }
  }
}

int main() {
  CALI_CXX_MARK_FUNCTION;

  // Create caliper ConfigManager object
  cali::ConfigManager mgr;
  mgr.start();

  // Image size
  int nx = 1024;
  int ny = 1024;
  int size = nx * ny;

  // TODO: Allocate host memory

  // TODO: Initialize input image

  // TODO: Allocate device memory

  // TODO: Define block and grid sizes

  // TODO: Launch filter kernel

  // TODO: Copy result back to host

  // TODO: Check result

  // Flush Caliper output
  mgr.stop();
  mgr.flush();

  std::cout << "End" << "\n";
  return 0;
}
