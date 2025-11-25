#include <iostream>
#include <stdio.h>
#include <vector>

#include <adiak.hpp>
#include <caliper/cali.h>
#include <caliper/cali-manager.h>


__constant__ float d_filter_constant[25];

/**
 * @brief Implemenation using global memory for image and constant memory for
 * filter
 * @param a input image
 * @param b output image
 * @param nx image width
 * @param nx image length
 */
__global__ void filter_constant(unsigned char *a, unsigned char *b, int nx, int ny) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int idy = blockIdx.y * blockDim.y + threadIdx.y;

  if (idx < nx && idy < ny) {
    float v = 0.0f;
    int filter_idx = 0;
    
    // 5x5 filter: iterate from -2 to +2 in both directions
    for (int fy = -2; fy <= 2; fy++) {
      for (int fx = -2; fx <= 2; fx++) {
        int nx_clamped = min(max(idx + fx, 0), nx - 1);
        int ny_clamped = min(max(idy + fy, 0), ny - 1);
        v += d_filter_constant[filter_idx] * a[ny_clamped * nx + nx_clamped];
        filter_idx++;
      }
    }

    uint f = (uint)(v + 0.5f);
    b[idy * nx + idx] = (unsigned char)min(255, max(0, static_cast<int>(f)));
  }
}

/**
 * @brief Implemenation using global memory for filter and image
 * @param a input image
 * @param b output image
 * @param c filter
 * @param nx image width
 * @param nx image length
 */
__global__ void filter_global(unsigned char *a, unsigned char *b, int nx, int ny, float *c) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int idy = blockIdx.y * blockDim.y + threadIdx.y;

  if (idx < nx && idy < ny) {
    float v = 0.0f;
    int filter_idx = 0;
    
    // 5x5 filter: iterate from -2 to +2 in both directions
    for (int fy = -2; fy <= 2; fy++) {
      for (int fx = -2; fx <= 2; fx++) {
        int nx_clamped = min(max(idx + fx, 0), nx - 1);
        int ny_clamped = min(max(idy + fy, 0), ny - 1);
        v += c[filter_idx] * a[ny_clamped * nx + nx_clamped];
        filter_idx++;
      }
    }

    uint f = (uint)(v + 0.5f);
    b[idy * nx + idx] = (unsigned char)min(255, max(0, static_cast<int>(f)));
  }
}

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

  for (int y = 0; y < ny; ++y) {
    for (int x = 0; x < nx; ++x) {
      float v = 0.0f;
      int filter_idx = 0;
      
      // 5x5 filter: iterate from -2 to +2 in both directions
      for (int fy = -2; fy <= 2; fy++) {
        for (int fx = -2; fx <= 2; fx++) {
          int nx_clamped = std::min(std::max(x + fx, 0), nx - 1);
          int ny_clamped = std::min(std::max(y + fy, 0), ny - 1);
          v += c[filter_idx] * a[ny_clamped * nx + nx_clamped];
          filter_idx++;
        }
      }

      uint f = (uint)(v + 0.5f);
      b[y * nx + x] = (unsigned char)std::min(255, std::max(0, static_cast<int>(f)));
    }
  }
}

int main(int argc, char* argv[]) {
  CALI_MARK_BEGIN("main");

  if (argc != 2) {
    std::cout<<"One argument required for program imgsize"<<std::endl;
    return -1;
  }
  
  int imgsize = atoi(argv[1]);
  std::cout<<"Image size: " << imgsize << std::endl;

  adiak::init(nullptr);
  adiak::value("image_size", imgsize);
  adiak::value("filter_size", 5);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  float t_global = 0.0f;
  float t_constant = 0.0f;
  float t_memcpy_h2d = 0.0f;
  float t_memcpy_d2h = 0.0f;
  // Create caliper ConfigManager object
  cali::ConfigManager mgr;
  mgr.start();

  // Image size
  int nx = imgsize;
  int ny = imgsize;
  int size = nx * ny;

  // TODO: Allocate host memory
  std::vector<unsigned char> input_img(size);
  std::vector<unsigned char> output_img_global(size);
  std::vector<unsigned char> output_img_constant(size);
  std::vector<unsigned char> output_img_ref(size);
  // 5x5 averaging filter (each value = 1/25 = 0.04)
  std::vector<float> five_filter = {
    0.04f, 0.04f, 0.04f, 0.04f, 0.04f,
    0.04f, 0.04f, 0.04f, 0.04f, 0.04f,
    0.04f, 0.04f, 0.04f, 0.04f, 0.04f,
    0.04f, 0.04f, 0.04f, 0.04f, 0.04f,
    0.04f, 0.04f, 0.04f, 0.04f, 0.04f
  };
  // TODO: Initialize input image
  for (int i = 0; i < size; ++i) {
    input_img[i] = static_cast<unsigned char>(i % 256);
  }
  // TODO: Allocate device memory
  unsigned char *d_input_img = nullptr;
  unsigned char *d_output_img = nullptr;
  float *d_filter = nullptr;
  // TODO: Define block and grid sizes
  dim3 blockSize(16, 16);
  dim3 gridSize((nx + blockSize.x - 1) / blockSize.x,
                (ny + blockSize.y - 1) / blockSize.y);
  // TODO: Launch filter kernel
  cudaMalloc((void**)&d_filter, five_filter.size() * sizeof(float));
  cudaMalloc((void**)&d_input_img, size * sizeof(unsigned char));
  cudaMalloc((void**)&d_output_img, size * sizeof(unsigned char));
  
  CALI_MARK_BEGIN("cudaMemcpy_host_to_device");
  cudaEventRecord(start);
  cudaMemcpy(d_input_img, input_img.data(), size * sizeof(unsigned char), cudaMemcpyHostToDevice);
  cudaMemcpy(d_filter, five_filter.data(), five_filter.size() * sizeof(float), cudaMemcpyHostToDevice);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&t_memcpy_h2d, start, stop);
  CALI_MARK_END("cudaMemcpy_host_to_device");

  // Global Memory Kernel
  CALI_MARK_BEGIN("kernel_global");
  cudaEventRecord(start);
  filter_global<<<gridSize, blockSize>>>(d_input_img, d_output_img, nx, ny, d_filter);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&t_global, start, stop);
  CALI_MARK_END("kernel_global");

  // Copy global memory result back to host (timed separately for global kernel)
  CALI_MARK_BEGIN("cudaMemcpy_device_to_host");
  cudaEventRecord(start);
  cudaMemcpy(output_img_global.data(), d_output_img, size * sizeof(unsigned char), cudaMemcpyDeviceToHost);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&t_memcpy_d2h, start, stop);
  CALI_MARK_END("cudaMemcpy_device_to_host");

  // Constant Memory Kernel
  cudaMemcpyToSymbol(d_filter_constant, five_filter.data(), five_filter.size() * sizeof(float), 0, cudaMemcpyHostToDevice);
  cudaMemcpy(d_input_img, input_img.data(), size * sizeof(unsigned char), cudaMemcpyHostToDevice);
  
  CALI_MARK_BEGIN("kernel_constant");
  cudaEventRecord(start);
  filter_constant<<<gridSize, blockSize>>>(d_input_img, d_output_img, nx, ny);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&t_constant, start, stop);
  CALI_MARK_END("kernel_constant");

  // Copy constant memory result back to host
  cudaMemcpy(output_img_constant.data(), d_output_img, size * sizeof(unsigned char), cudaMemcpyDeviceToHost);
  // Compute CPU reference
  filter_CPU(input_img, output_img_ref, nx, ny, five_filter);
  
  // Check global memory result against CPU
  bool match_global = true;
  for (int i = 0; i < size; ++i) {
    if (output_img_global[i] != output_img_ref[i]) {
      match_global = false;
      std::cout << "Global Memory Mismatch at index " << i << ": GPU result = "
                << static_cast<int>(output_img_global[i]) << ", CPU result = "
                << static_cast<int>(output_img_ref[i]) << std::endl;
      break;
    }
  }
  if (match_global) {
    std::cout << "Global Memory: Results match CPU!" << std::endl;
  } else {
    std::cout << "Global Memory: Results do not match CPU!" << std::endl;
  }
  
  // Check constant memory result against CPU
  bool match_constant = true;
  for (int i = 0; i < size; ++i) {
    if (output_img_constant[i] != output_img_ref[i]) {
      match_constant = false;
      std::cout << "Constant Memory Mismatch at index " << i << ": GPU result = "
                << static_cast<int>(output_img_constant[i]) << ", CPU result = "
                << static_cast<int>(output_img_ref[i]) << std::endl;
      break;
    }
  }
  if (match_constant) {
    std::cout << "Constant Memory: Results match CPU!" << std::endl;
  } else {
    std::cout << "Constant Memory: Results do not match CPU!" << std::endl;
  }

  // Effective Bandwidth Calculation
  long long bytes_transferred = 2LL * size * sizeof(unsigned char); // input + output image
  
  // Time conversion from ms to s in bandwidth calculation
  double bw_global = (bytes_transferred / (t_global / 1000.0)) / 1e9;
  double bw_constant = (bytes_transferred / (t_constant / 1000.0)) / 1e9;

  // Print results
  std::cout << "\n========== Timing Results ==========" << std::endl;
  std::cout << "cudaMemcpy Host to Device: " << t_memcpy_h2d << " ms" << std::endl;
  std::cout << "cudaMemcpy Device to Host: " << t_memcpy_d2h << " ms" << std::endl;
  std::cout << "Kernel Global Time: " << t_global << " ms" << std::endl;
  std::cout << "Kernel Constant Time: " << t_constant << " ms" << std::endl;
  std::cout << "\n========== Effective Bandwidth ==========" << std::endl;
  std::cout << "Global Memory Kernel Bandwidth: " << bw_global << " GB/s" << std::endl;
  std::cout << "Constant Memory Kernel Bandwidth: " << bw_constant << " GB/s" << std::endl;

  // Record timing data with Adiak
  adiak::value("cudaMemcpy_host_to_device_ms", t_memcpy_h2d);
  adiak::value("cudaMemcpy_device_to_host_ms", t_memcpy_d2h);
  adiak::value("kernel_global_time_ms", t_global);
  adiak::value("kernel_constant_time_ms", t_constant);
  adiak::value("bandwidth_global_GBs", bw_global);
  adiak::value("bandwidth_constant_GBs", bw_constant);

  // Flush Caliper output
  adiak::fini();
  mgr.stop();
  mgr.flush();

  // Free device memory
  cudaFree(d_input_img);
  cudaFree(d_output_img);
  cudaFree(d_filter);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  std::cout << "End" << "\n";
  CALI_MARK_END("main");
  return 0;
}
