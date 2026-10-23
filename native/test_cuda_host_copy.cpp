// Exercise real CUDA transfers with CPU-only Open3D; no camera services are used.
#include <cuda_runtime.h>
#include <unistd.h>

#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

#include <opencv2/imgcodecs.hpp>
#include <yaml-cpp/yaml.h>

#include "core/common/cuda_image.hpp"
#include "utils/camera/capture_writer.hpp"

namespace {
void Require(bool condition, const char *message) {
  if (!condition) throw std::runtime_error(message);
}

void Cuda(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

template <typename Operation>
void MustReject(Operation operation) {
  try {
    operation();
  } catch (const std::exception &) {
    return;
  }
  throw std::runtime_error("Invalid image input was accepted");
}

struct DeviceBuffer {
  int device;
  float *address = nullptr;
  explicit DeviceBuffer(int id, size_t bytes) : device(id) {
    Cuda(cudaSetDevice(device));
    Cuda(cudaMalloc(reinterpret_cast<void **>(&address), bytes));
  }
  ~DeviceBuffer() {
    int current = 0;
    cudaGetDevice(&current);
    cudaSetDevice(device);
    cudaFree(address);
    cudaSetDevice(current);
  }
};

struct TemporaryCapture {
  std::filesystem::path path;
  TemporaryCapture() {
    auto pattern = (std::filesystem::temp_directory_path() / "bin-host-copy-XXXXXX").string();
    const auto created = mkdtemp(pattern.data());
    if (!created) throw std::runtime_error("Cannot create temporary capture directory");
    path = created;
  }
  ~TemporaryCapture() {
    std::error_code error;
    std::filesystem::remove_all(path, error);
  }
};
}  // namespace

int main() {
  try {
    using ebots_perception::core::common::CopyCudaFloatImageToHost;
    float placeholder = 0;
    MustReject([&] { CopyCudaFloatImageToHost(nullptr, 2, 3, false); });
    MustReject([&] { CopyCudaFloatImageToHost(&placeholder, 0, 3, true); });
    MustReject([&] { CopyCudaFloatImageToHost(&placeholder, 2, 0, false); });
    MustReject([&] { CopyCudaFloatImageToHost(&placeholder, UINT32_MAX, UINT32_MAX, false); });

    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
      std::cout << "SKIP: CUDA device required for synthetic buffer-copy regression\n";
      return 77;
    }
    int original_device = 0;
    Cuda(cudaGetDevice(&original_device));
    constexpr uint32_t rows = 2, cols = 4;
    std::array<float, rows * cols> values{
        1234.1234130859375f, 0.125f, -0.0f, 0.0f, -3.5f, 65535.5f,
        std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity()};
    std::vector<open3d::core::Tensor> depth, spotlight;
    for (int sensor = 0; sensor < 2; ++sensor) {
      const int source_device = sensor % devices;
      const int calling_device = (source_device + 1) % devices;
      {
        DeviceBuffer source(source_device, sizeof(values));
        Cuda(cudaMemcpy(source.address, values.data(), sizeof(values), cudaMemcpyHostToDevice));
        Cuda(cudaSetDevice(calling_device));
        depth.push_back(CopyCudaFloatImageToHost(source.address, rows, cols, false, source_device));
        spotlight.push_back(CopyCudaFloatImageToHost(source.address, rows, cols, true, source_device));
        int after_copy = -1;
        Cuda(cudaGetDevice(&after_copy));
        Require(after_copy == calling_device, "Copy changed the caller's current CUDA device");
        Cuda(cudaSetDevice(source_device));
        Cuda(cudaMemset(source.address, 0, sizeof(values)));
        Cuda(cudaSetDevice(calling_device));
      }
      // Camera shared memory can be reused immediately; saved tensors must own
      // their values after the source allocation is overwritten and freed.
      for (const auto &image : {depth.back(), spotlight.back()}) {
        Require(image.GetDevice() == open3d::core::Device("CPU:0"), "Capture tensor must reside on CPU");
        Require(image.GetDtype() == open3d::core::Float32, "Capture dtype must remain Float32");
        Require(image.IsContiguous(), "Capture tensor must be contiguous");
        Require(std::memcmp(image.GetDataPtr<float>(), values.data(), sizeof(values)) == 0,
                "GPU transfer altered values or retained borrowed storage");
      }
      Require(depth.back().GetShape() == open3d::core::SizeVector({rows, cols, 1}), "Wrong depth shape");
      Require(spotlight.back().GetShape() == open3d::core::SizeVector({1, 1, rows, cols}), "Wrong spotlight shape");
    }
    Cuda(cudaSetDevice(original_device));

    TemporaryCapture output;
    {
      ebots_perception::node::utils::CaptureWriter writer(output.path.string(), "test", "capture", 4, 0, true, "A1");
      writer.SaveImagePair("spotlight", spotlight, true);
      writer.SaveImagePair("depth", depth, false);
      writer.Complete();
    }
    Require(YAML::LoadFile((output.path / "capture_manifest.yaml").string())["status"].as<std::string>() == "complete",
            "Writer did not complete capture evidence");
    for (const auto *kind : {"spotlight", "depth"}) {
      for (int sensor = 0; sensor < 2; ++sensor) {
        const auto file = output.path / kind / ("Frame" + std::to_string(sensor) + ".exr");
        const auto image = cv::imread(file.string(), cv::IMREAD_UNCHANGED);
        Require(image.type() == CV_32FC1 && image.rows == rows && image.cols == cols,
                "EXR changed image dimensions or dtype");
        for (size_t pixel = 0; pixel < values.size(); ++pixel) {
          const float actual = image.ptr<float>()[pixel];
          Require(std::isnan(values[pixel]) ? std::isnan(actual) :
                    std::memcmp(&actual, &values[pixel], sizeof(float)) == 0,
                  "EXR round trip changed a measurement");
        }
      }
    }
    std::cout << "PASS: owning CPU Float32 copies, both layouts, CUDA device restoration, "
              << "shared-buffer reuse and four lossless EXRs (" << devices << " CUDA devices)\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
