#!/bin/sh
# tests/run_tests.sh — build and run backend kernel comparison tests.
#
# Builds test_kernels against CPU and CUDA, runs them with identical
# synthetic inputs, then checks that CUDA outputs are within abs_tol of CPU.
#
# Usage: sh tests/run_tests.sh

set -e
cd "$(dirname "$0")/.."

ABS_TOL="1e-4"   # absolute tolerance for element-wise float comparison

# ---- helpers ----------------------------------------------------------------

need_rebuild() {
    # need_rebuild <target> <dep...>  — exit 0 (true) if rebuild needed
    tgt="$1"; shift
    [ ! -f "$tgt" ] && return 0
    for dep in "$@"; do
        [ -f "$dep" ] && [ "$dep" -nt "$tgt" ] && return 0
    done
    return 1
}

SHARED_HDRS="backends/backend.h backends/model_types.h tests/test_kernels.c"

# ---- build compare_floats ---------------------------------------------------

if need_rebuild tests/compare_floats tests/compare_floats.c; then
    echo "Building compare_floats ..."
    gcc -O2 -o tests/compare_floats tests/compare_floats.c -lm
fi

# ---- CPU reference ----------------------------------------------------------

if need_rebuild backends/cpu/backend.o \
        backends/cpu/backend.c backends/backend.h backends/model_types.h; then
    echo "Compiling CPU backend ..."
    gcc -O3 -march=native -I. -c backends/cpu/backend.c -o backends/cpu/backend.o
fi

if need_rebuild tests/test_kernels_cpu $SHARED_HDRS backends/cpu/backend.o; then
    echo "Building test_kernels_cpu ..."
    gcc -O3 -march=native -I. -o tests/test_kernels_cpu \
        tests/test_kernels.c backends/cpu/backend.o -lm
fi

echo
echo "Running CPU reference ..."
./tests/test_kernels_cpu tests/ref_cpu.bin

PASS=1

# ---- CUDA -------------------------------------------------------------------

if command -v nvcc >/dev/null 2>&1; then
    CUDA_PREFIX="$(dirname "$(dirname "$(command -v nvcc)")")"

    if need_rebuild backends/cuda/backend.o \
            backends/cuda/backend.cu backends/backend.h backends/model_types.h; then
        echo "Compiling CUDA backend ..."
        nvcc -O3 -arch=native -DBACKEND_CUDA -I. \
            -c backends/cuda/backend.cu -o backends/cuda/backend.o
    fi

    if need_rebuild tests/test_kernels_cuda $SHARED_HDRS backends/cuda/backend.o; then
        echo "Building test_kernels_cuda ..."
        gcc -O3 -march=native -DBACKEND_CUDA -I. \
            -o tests/test_kernels_cuda \
            tests/test_kernels.c backends/cuda/backend.o \
            -L"${CUDA_PREFIX}/lib64" -lcudart -lstdc++ -lm
    fi

    echo
    echo "Running CUDA test ..."
    ./tests/test_kernels_cuda tests/out_cuda.bin

    echo
    printf "Comparing CUDA vs CPU (abs_tol=%s) ...\n" "$ABS_TOL"
    if ./tests/compare_floats tests/ref_cpu.bin tests/out_cuda.bin "$ABS_TOL"; then
        echo "CUDA: PASS"
    else
        echo "CUDA: FAIL"
        PASS=0
    fi
else
    echo
    echo "nvcc not found — skipping CUDA test."
fi

# ---- summary ----------------------------------------------------------------

echo
if [ "$PASS" -eq 1 ]; then
    echo "All available backend tests PASSED."
    exit 0
else
    echo "One or more backend tests FAILED."
    exit 1
fi
