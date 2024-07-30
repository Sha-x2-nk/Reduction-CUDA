#pragma once

#include "errorCheckUtils.cuh"

#include <cuda_runtime.h>

#include <iostream>


/*
add adjacent elements.
*/ 
template<typename TP>
__global__ void reduce1(TP* A, TP* output, int size) {
    int bx = blockIdx.x;
    int tx = threadIdx.x;

    int idx = bx * blockDim.x + tx;
    
    if (idx < size) {

        for (int s = 1; s < blockDim.x; s *= 2) {
            if (tx % (2 * s) == 0 && idx + s< size)
                A[idx] = A[idx] + A[idx + s];
            __syncthreads();
        }


        if (tx == 0) {
            output[bx] = A[idx];
        }
    }
}

/*
using shared mem
*/
template<typename TP>
__global__ void reduce2(TP* A, TP* output, int size) {
    const int bx = blockIdx.x;
    const int tx = threadIdx.x;
    const int idx = bx * blockDim.x + tx;

    __shared__ TP s_A[256]; // block size is 256

    // loading in SHMEM
    s_A[tx] = (idx < size)?A[idx]:0;
    __syncthreads();
   

    for (int s = 1; s < blockDim.x; s *= 2) {
        if (tx % (2 * s) == 0) {
            s_A[tx] += s_A[tx + s];
        }
        __syncthreads();
    }


    if (tx == 0) 
        output[bx] = s_A[0];
}

/*
    removing if, % is costly
*/
template<typename TP>
__global__ void reduce3(TP* A, TP* output, int size) {
    int bx = blockIdx.x;
    int tx = threadIdx.x;
    int idx = bx * blockDim.x + tx;

    __shared__ TP s_A[256]; // block size is 256

    // load this block in SHMEM
    s_A[tx] = (idx < size)?A[idx]:0;

    __syncthreads();

    for (int s = 1; s < blockDim.x; s *= 2) {
        int index = 2 * s * tx;

        if (index< blockDim.x) 
            s_A[index] += s_A[index + s];
        __syncthreads();
    }


    if (tx == 0) 
        output[bx] = s_A[0];
    
}

/*
works on stride instead of adjacent elements
*/
template<typename TP>
__global__ void reduce4(TP *A, TP *output, int size){
    const int bx = blockIdx.x;
    const int tx = threadIdx.x;
    const int idx = bx * blockDim.x + tx;
    __shared__ TP s_A[256]; // BLOCK SIZE is 256

    s_A[tx] = (idx < size)?A[idx]:0;
    __syncthreads();

    for(int s = blockDim.x / 2; s > 0; s/= 2){
        if(tx < s){ // tx + s < blockDim.x ==> tx < blockDim.x - s  range(s) = (0, blockDim.x / 2]
            s_A[tx] += s_A[tx + s];
        }
        __syncthreads();
    }

    if(tx == 0){
        output[bx] = s_A[0];
    }
}

/*
in stride method, we only need to launch half the threads, since stride is BLOCK_SIZE.
reducing number of redundant warps
*/
template<typename TP>
__global__ void reduce5(TP *A, TP *output, int size){
    const int bx = blockIdx.x;
    const int tx = threadIdx.x;
    const int idx = bx * blockDim.x * 2 + tx;
    __shared__ TP s_A[256]; // BLOCK SIZE is 256

    s_A[tx] = ( (idx < size)?A[idx]:0 ) + ( (idx + blockDim.x < size)?A[idx + blockDim.x] : 0 );
    __syncthreads();

    for(int s = blockDim.x / 2; s > 0; s/= 2){
        if(tx < s){ // tx + s < blockDim.x ==> tx < blockDim.x - s  range(s) = (0, blockDim.x / 2]
            s_A[tx] += s_A[tx + s];
        }
        __syncthreads();
    }

    if(tx == 0){
        output[bx] = s_A[0];
    }
}

/*
Warp Reduction unrolled, since calls are synchronous in warp, no need for __syncthreads()
*/
template< typename TP >
__device__ void warpReduce6(volatile TP* s_A, int tid){ // warp reduce for kernel 6
    s_A[tid] += s_A[tid + 32];
    s_A[tid] += s_A[tid + 16];
    s_A[tid] += s_A[tid + 8];
    s_A[tid] += s_A[tid + 4];
    s_A[tid] += s_A[tid + 2];
    s_A[tid] += s_A[tid + 1];
}

template<typename TP>
__global__ void reduce6(TP *A, TP *output, int size){
    const int bx = blockIdx.x;
    const int tx = threadIdx.x;
    const int idx = bx * blockDim.x * 2 + tx;
    __shared__ TP s_A[256]; // BLOCK SIZE is 256

    s_A[tx] = ( (idx < size)?A[idx]:0 ) + ( (idx + blockDim.x < size)?A[idx + blockDim.x] : 0 );
    __syncthreads();

    for(int s = blockDim.x / 2; s > 32; s/= 2){
        if(tx < s){ // tx + s < blockDim.x ==> tx < blockDim.x - s  range(s) = (0, blockDim.x / 2]
            s_A[tx] += s_A[tx + s];
        }
        __syncthreads();
    }

    if(tx< 32) warpReduce6<TP>(s_A, tx);
 
    if(tx == 0)
        output[bx] = s_A[0];
}

/*
    since stride are reduced by a factor of 2, unrolling normal loop. we will pass stride(BLOCK_SIZE) in template function so that IFs are resolved compile time
*/
template< typename TP>
__device__ void warpReduce7(volatile TP* s_A, int tid){ // warp reduce for kernel 6
    s_A[tid] += s_A[tid + 32];
    s_A[tid] += s_A[tid + 16];
    s_A[tid] += s_A[tid + 8];
    s_A[tid] += s_A[tid + 4];
    s_A[tid] += s_A[tid + 2];
    s_A[tid] += s_A[tid + 1];
}

template<typename TP, int BLOCK_SIZE>
__global__ void reduce7(TP *A, TP *output, int size){
    const int bx = blockIdx.x;
    const int tx = threadIdx.x;
    const int idx = bx * blockDim.x * 2 + tx;
    __shared__ TP s_A[BLOCK_SIZE];


    s_A[tx] = ( (idx < size)?A[idx]:0 ) + ( (idx + BLOCK_SIZE < size)?A[idx + BLOCK_SIZE] : 0 );
    __syncthreads();

    // these IFs will be evaluated at compile time since BLOCK_SIZE is constant, passed in template
    if(BLOCK_SIZE > 511){
        if(tx < 256){
            s_A[tx] += s_A[tx + 256];
        } 
        __syncthreads();
    }

    if(BLOCK_SIZE > 255){
        if(tx < 128){
            s_A[tx] += s_A[tx + 128];
        }
        __syncthreads();
    }
    if(BLOCK_SIZE > 127){
        if(tx < 64){
            s_A[tx] += s_A[tx + 64];
        }
        __syncthreads();
    }

    if(tx< 32) warpReduce7<TP>(s_A, tx);
 
    if(tx == 0){
        output[bx] = s_A[0];
    }
}

/*
each thread does more work, less number of grids are launched.
*/
template< typename TP >
__device__ void warpReduce8(volatile TP* s_A, int tid){ // warp reduce for kernel 6
    s_A[tid] += s_A[tid + 32];
    s_A[tid] += s_A[tid + 16];
    s_A[tid] += s_A[tid + 8];
    s_A[tid] += s_A[tid + 4];
    s_A[tid] += s_A[tid + 2];
    s_A[tid] += s_A[tid + 1];
}

template<typename TP, int BLOCK_SIZE>
__global__ void reduce8(TP *A, TP *output, int size){
    const int bx = blockIdx.x;
    const int tx = threadIdx.x;
    int idx = bx * BLOCK_SIZE * 2 + tx;
    const int gridSize = BLOCK_SIZE * 2 * gridDim.x;
    __shared__ TP s_A[BLOCK_SIZE];
    
    s_A[tx] = 0;

    // assume only 1 grid is being launched
    while(idx< size){
        s_A[tx] += A[idx] + ( (idx + BLOCK_SIZE < size)?A[idx + BLOCK_SIZE] : 0 );
        idx += gridSize;
    }   
    __syncthreads();

    if(BLOCK_SIZE > 511){
        if(tx < 256){
            s_A[tx] += s_A[tx + 256];
        } 
        __syncthreads();
    }

    if(BLOCK_SIZE > 255){
        if(tx < 128){
            s_A[tx] += s_A[tx + 128];
        }
        __syncthreads();
    }
    if(BLOCK_SIZE > 127){
        if(tx < 64){
            s_A[tx] += s_A[tx + 64];
        }
        __syncthreads();
    }

    if(tx< 32) warpReduce8<TP>(s_A, tx);
    
    if(tx == 0){
        output[bx] = s_A[0];
    }
}