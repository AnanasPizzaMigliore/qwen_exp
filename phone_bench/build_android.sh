#!/usr/bin/env bash
# Cross-compile llama.cpp's llama-mtmd-cli for Android, as used by the on-device benchmark.
#
#   LLAMA_CPP=/path/to/llama.cpp NDK=/path/to/android-ndk-r27d VULKAN_SDK=/path/to/vulkan-sdk/x86_64 \
#     bash build_android.sh
#
# Tested with llama.cpp commit d05fe1d, NDK r27d, Vulkan SDK 1.4.341. Outputs, stripped, into ./bin:
#   llama-mtmd-cli          arm64, armv8.2-a+dotprod+fp16        any phone since ~2018
#   llama-mtmd-cli-i8mm     arm64, + i8mm                        Snapdragon 8 Gen 1+, Dimensity 9000+, Tensor G3+
#   llama-mtmd-cli-vulkan   arm64, Vulkan GPU backend            needs the phone's libvulkan.so
#   llama-mtmd-cli-x86_64   x86_64, CPU                          Android emulator (functional tests only)
# All are static apart from Android system libraries. -DGGML_OPENMP=OFF because the NDK's OpenMP
# runtime is not reliably available on-device.
set -euo pipefail
: "${LLAMA_CPP:?set LLAMA_CPP to the llama.cpp checkout}"
: "${NDK:?set NDK to the Android NDK root}"
OUT="$(cd "$(dirname "$0")" && pwd)/bin"
mkdir -p "$OUT"
TC=$NDK/build/cmake/android.toolchain.cmake
STRIP=$NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-strip
COMMON=(-DANDROID_PLATFORM=android-28 -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF
        -DGGML_OPENMP=OFF -DGGML_LLAMAFILE=OFF -DLLAMA_CURL=OFF)

build () {  # $1 build dir, $2 output name, rest: extra cmake args
  local dir=$1 name=$2; shift 2
  cmake -S "$LLAMA_CPP" -B "$LLAMA_CPP/$dir" -DCMAKE_TOOLCHAIN_FILE="$TC" "${COMMON[@]}" "$@"
  cmake --build "$LLAMA_CPP/$dir" --config Release -j "$(nproc)" --target llama-mtmd-cli
  "$STRIP" -o "$OUT/$name" "$LLAMA_CPP/$dir/bin/llama-mtmd-cli"
}

ARM=(-DANDROID_ABI=arm64-v8a)
build build-android      llama-mtmd-cli      "${ARM[@]}" \
  -DCMAKE_C_FLAGS="-march=armv8.2-a+dotprod+fp16" -DCMAKE_CXX_FLAGS="-march=armv8.2-a+dotprod+fp16"
build build-android-i8mm llama-mtmd-cli-i8mm "${ARM[@]}" \
  -DCMAKE_C_FLAGS="-march=armv8.2-a+dotprod+fp16+i8mm" -DCMAKE_CXX_FLAGS="-march=armv8.2-a+dotprod+fp16+i8mm"

if [ -n "${VULKAN_SDK:-}" ]; then
  # The NDK ships vulkan.h and libvulkan.so but not vulkan.hpp, so take headers and glslc from the SDK.
  SYS=$NDK/toolchains/llvm/prebuilt/linux-x86_64/sysroot
  build build-android-vulkan llama-mtmd-cli-vulkan "${ARM[@]}" \
    -DCMAKE_C_FLAGS="-march=armv8.2-a+dotprod+fp16" -DCMAKE_CXX_FLAGS="-march=armv8.2-a+dotprod+fp16" \
    -DGGML_VULKAN=ON \
    -DVulkan_INCLUDE_DIR="$VULKAN_SDK/include" \
    -DVulkan_LIBRARY="$SYS/usr/lib/aarch64-linux-android/28/libvulkan.so" \
    -DVulkan_GLSLC_EXECUTABLE="$VULKAN_SDK/bin/glslc"
else
  echo "VULKAN_SDK not set - skipping the Vulkan build"
fi

build build-android-x86_64 llama-mtmd-cli-x86_64 -DANDROID_ABI=x86_64 -DGGML_NATIVE=OFF

ls -la "$OUT"
