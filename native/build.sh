#!/usr/bin/env bash
# Build in the existing camera/Eyes environment. This does not start any ROS node.
set -euo pipefail
native_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
capture_root=$(cd -- "$native_dir/../.." && pwd)
if [ "$#" -lt 1 ] || [ "$#" -gt 2 ] || [ "$1" = --help ] || [ "$1" = -h ]; then
  printf 'Usage: %s CAMERA_CONTAINER [CONTAINER_OUTPUT_DIRECTORY]\n' "$0"
  printf 'Choose the running container with the station Eyes build environment; its name is unrestricted.\n'
  if [ "$#" -ge 1 ] && { [ "$1" = --help ] || [ "$1" = -h ]; }; then exit 0; fi
  exit 2
fi
capture_container=$1
capture_destination=${2:-/root/bin_filter_offline/native}
capture_user="$(id -u):$(id -g)"
capture_stage=$(mktemp -d /tmp/bin-capture-source.XXXXXX)
trap 'rm -rf -- "$capture_stage"' EXIT
capture_remote="/tmp/bin-capture-build-$(id -u)-$$"

mkdir -p "$capture_stage/native"
cp "$native_dir/CMakeLists.txt" "$capture_stage/native/"
cp "$native_dir/test_cuda_host_copy.cpp" "$capture_stage/native/"
for capture_file in \
  perception_node/src/tools/bin_capture.cpp \
  perception_node/src/utils/camera/bin_capture_engine.cpp \
  perception_node/src/utils/camera/capture_acquisition.cpp \
  perception_node/src/utils/camera/capture_writer.cpp \
  perception_node/src/utils/camera/denali_images.cpp \
  perception_node/include/utils/camera/bin_capture_engine.hpp \
  perception_node/include/utils/camera/capture_acquisition.hpp \
  perception_node/include/utils/camera/depth_request_buffer.hpp \
  perception_node/include/utils/camera/capture_writer.hpp \
  perception_node/include/utils/camera/image_utils.hpp \
  perception_node/test/unit_tests/utils/camera/test_depth_request_lifetime.cpp \
  perception_core/src/common/cuda_image.cpp \
  perception_core/src/image/float_depth_io.cpp \
  perception_core/include/core/common/cuda_image.hpp \
  perception_core/include/core/image/utils.hpp; do
  mkdir -p "$capture_stage/workspace/$(dirname -- "$capture_file")"
  cp "$capture_root/ebots_ros2_perception/workspace/$capture_file" "$capture_stage/workspace/$capture_file"
done
chmod -R a+rX "$capture_stage"
docker cp "$capture_stage/." "$capture_container:$capture_remote"
docker exec "$capture_container" chown -R "$capture_user" "$capture_remote"
docker exec "$capture_container" bash -c '
  set -eo pipefail
  source /opt/ros/humble/setup.bash
  for setup_file in /root/workspace/install/setup.bash /root/eyes/workspace/install/setup.bash; do
    if [ -f "$setup_file" ]; then source "$setup_file"; fi
  done
  cmake -S "$1/native" -B "$1/build" -DPERCEPTION_SOURCE_WORKSPACE="$1/workspace" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=/usr/bin/g++-14
  cmake --build "$1/build" --parallel 2
  mkdir -p "$2"
  install -m 0755 "$1/build/bin_capture" "$2/bin_capture.new"
  mv -f "$2/bin_capture.new" "$2/bin_capture"
  chown "$3" "$2" "$2/bin_capture"
  printf "Standalone camera helper: %s/bin_capture\n" "$2"
' _ "$capture_remote" "$capture_destination" "$capture_user"
