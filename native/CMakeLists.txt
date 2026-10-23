cmake_minimum_required(VERSION 3.20)
project(bin_capture_native LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
option(BUILD_TESTING "Build native capture regression checks" OFF)
set(PERCEPTION_SOURCE_WORKSPACE "${CMAKE_CURRENT_LIST_DIR}/../../ebots_ros2_perception/workspace"
    CACHE PATH "Workspace containing the shared capture implementation")
set(node "${PERCEPTION_SOURCE_WORKSPACE}/perception_node")
set(core "${PERCEPTION_SOURCE_WORKSPACE}/perception_core")

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(ebots_ros2_common REQUIRED)
find_package(Open3D REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgcodecs imgproc)
find_package(CUDAToolkit REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(Threads REQUIRED)

# Compile the same acquisition and manifest writer used by the capture service.
# No Perception node, pose pipeline, model loader, ONNX or inference library links.
add_executable(bin_capture
  "${node}/src/tools/bin_capture.cpp"
  "${node}/src/utils/camera/bin_capture_engine.cpp"
  "${node}/src/utils/camera/capture_acquisition.cpp"
  "${node}/src/utils/camera/capture_writer.cpp"
  "${node}/src/utils/camera/denali_images.cpp"
  "${core}/src/common/cuda_image.cpp"
  "${core}/src/image/float_depth_io.cpp")
target_include_directories(bin_capture PRIVATE "${node}/include" "${core}/include" ${OpenCV_INCLUDE_DIRS})
ament_target_dependencies(bin_capture rclcpp ebots_ros2_common)
target_link_libraries(bin_capture Open3D::Open3D CUDA::cudart yaml-cpp ${OpenCV_LIBS}
  ebots_ros2_common::ebots_common_lib Threads::Threads)
install(TARGETS bin_capture RUNTIME DESTINATION .)

# Optional synthetic GPU-memory regression. It never requests camera images.
if(BUILD_TESTING)
  enable_testing()
  find_package(ament_cmake_gtest REQUIRED)
  # Metadata/ROS lifetime checks use fake Eyes on domain 198 and no GPU access.
  ament_add_gtest(test_depth_request_lifetime
    "${node}/test/unit_tests/utils/camera/test_depth_request_lifetime.cpp"
    "${node}/src/utils/camera/capture_acquisition.cpp" TIMEOUT 30)
  target_include_directories(test_depth_request_lifetime PRIVATE "${node}/include" ${OpenCV_INCLUDE_DIRS})
  ament_target_dependencies(test_depth_request_lifetime rclcpp ebots_ros2_common)
  target_link_libraries(test_depth_request_lifetime ebots_ros2_common::ebots_common_lib
    CUDA::cudart yaml-cpp ${OpenCV_LIBS} Threads::Threads)
  add_executable(test_cuda_host_copy
    "${CMAKE_CURRENT_LIST_DIR}/test_cuda_host_copy.cpp"
    "${core}/src/common/cuda_image.cpp"
    "${core}/src/image/float_depth_io.cpp"
    "${node}/src/utils/camera/capture_writer.cpp")
  target_include_directories(test_cuda_host_copy PRIVATE "${node}/include" "${core}/include" ${OpenCV_INCLUDE_DIRS})
  ament_target_dependencies(test_cuda_host_copy rclcpp ebots_ros2_common)
  target_link_libraries(test_cuda_host_copy Open3D::Open3D CUDA::cudart yaml-cpp ${OpenCV_LIBS}
    ebots_ros2_common::ebots_common_lib Threads::Threads)
  add_test(NAME cuda_host_copy COMMAND test_cuda_host_copy)
  set_tests_properties(cuda_host_copy PROPERTIES SKIP_RETURN_CODE 77)
endif()
