# CUDA Reduction Kernels
## Introduction
This repository contains CUDA implementations of 8 reduction kernels, showcasing a progression from basic approaches to optimized techniques for efficient reduction operations on NVIDIA GPUs.
## Kernels and Performance

#### HARDWARE: RTX 3070Ti ( Compute Capablity 8.6 )

Below kernels were bench on ~4M elements using Nvidia Nsight Compute.
| Kernel | Runtime | Relative Speedup | Absolute Speedup |
|--|--|--|--|
|1. Reduction Naive, adjacent numbers are summed, even threads at work only | 351.3 us | 1 x | 1 x|
|2. + Using shared memory to store the numbers and then operate on them | 333.7 us | 1.05 x | 1.05 x|
|3. Removing IF for checking working thread | 213.02 us | 1.56 x | 1.64 x|
|4. Instead of Adjacent elements, add elements which are 'stride' apart | 205.12 us | 1.03 x | 1.71 x|
|5. Since working by a stride, changing launch config to remove redundant warps | 114.7 us | 1.78 x | 3.06 x|
|6. Warp Unrolling | 69.51 us | 1.65 x | 5.05 x|
|7. Main loop unrolling using template constant | 66.72 us | 1.04 x | 5.26 x|
|8. Thread coarescening, each thread adding multiple elements (launching less number of grids) | 55.9 us | 1.19 x | 6.28 x|

## Usage
* Compile using nvcc

    <code>nvcc main.cu -o main.exe -lcublas</code>

* Run

    <code>main.exe</code>

* Tune parameters like BLOCK_SIZE and GRID_SIZE for your hardware.

## Acknowledgements
Mark Harris' amazing article on optimising CUDA reduction
https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf




