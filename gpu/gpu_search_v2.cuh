#pragma once

#include <vector>
#include <queue>
#include <iostream>
#include <iomanip>
#include <sys/time.h>
#include <cfloat>
#include "cuda_runtime.h"

#define CUDA_CHECK(err) { if (err != cudaSuccess) { std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " in file " << __FILE__ << " at line " << __LINE__ << std::endl; exit(EXIT_FAILURE); } }

struct GpuSearchResult {
    float dist;
    int id;
};

const int TILE_WIDTH = 16;
const int BLOCK_DIM_X_V2 = TILE_WIDTH;
const int BLOCK_DIM_Y_V2 = TILE_WIDTH;

__global__ void matrix_mul_shared_mem_kernel(const float* base, const float* query, float* results,
                                             int n, int m, int d) {
    __shared__ float tile_base[BLOCK_DIM_Y_V2][TILE_WIDTH];
    __shared__ float tile_query[TILE_WIDTH][BLOCK_DIM_X_V2];
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * blockDim.y + ty;
    int col = blockIdx.x * blockDim.x + tx;
    float sum = 0.0f;

    for (int t = 0; t < (d + TILE_WIDTH - 1) / TILE_WIDTH; ++t) {
        if (row < n && (t * TILE_WIDTH + tx) < d) {
            tile_base[ty][tx] = base[row * d + t * TILE_WIDTH + tx];
        } else {
            tile_base[ty][tx] = 0.0f;
        }
        if (col < m && (t * TILE_WIDTH + ty) < d) {
            tile_query[ty][tx] = query[col * d + t * TILE_WIDTH + ty];
        } else {
            tile_query[ty][tx] = 0.0f;
        }
        __syncthreads();
        for (int k = 0; k < TILE_WIDTH; ++k) {
            sum += tile_base[ty][k] * tile_query[k][tx];
        }
        __syncthreads();
    }
    if (row < n && col < m) {
        results[row * m + col] = sum;
    }
}


__global__ void find_topk_kernel_correct(const float* results_matrix, GpuSearchResult* topk_results,
                                         int n, int m, int k) {
    int query_idx = blockIdx.x;
    if (query_idx >= m) return;

    float local_dists[10];
    int local_ids[10];
    for (int i = 0; i < k; ++i) {
        local_dists[i] = -FLT_MAX;
        local_ids[i] = -1;
    }

    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float dist = results_matrix[i * m + query_idx];
        float min_dist = local_dists[0];
        int min_idx = 0;
        for (int j = 1; j < k; ++j) {
            if (local_dists[j] < min_dist) {
                min_dist = local_dists[j];
                min_idx = j;
            }
        }
        if (dist > min_dist) {
            local_dists[min_idx] = dist;
            local_ids[min_idx] = i;
        }
    }

    extern __shared__ float shared_mem[];
    float* shared_dists = shared_mem;
    int* shared_ids = (int*)&shared_dists[blockDim.x * k];

    for (int i = 0; i < k; ++i) {
        shared_dists[threadIdx.x * k + i] = local_dists[i];
        shared_ids[threadIdx.x * k + i] = local_ids[i];
    }
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            for (int i = 0; i < k; ++i) {
                float dist = shared_dists[(threadIdx.x + s) * k + i];
                int id = shared_ids[(threadIdx.x + s) * k + i];
                float min_dist = shared_dists[threadIdx.x * k + 0];
                int min_idx = 0;
                for (int j = 1; j < k; ++j) {
                    if (shared_dists[threadIdx.x * k + j] < min_dist) {
                        min_dist = shared_dists[threadIdx.x * k + j];
                        min_idx = j;
                    }
                }
                if (dist > min_dist) {
                    shared_dists[threadIdx.x * k + min_idx] = dist;
                    shared_ids[threadIdx.x * k + min_idx] = id;
                }
            }
        }
        __syncthreads();
    }

    if (threadIdx.x < k) {
        topk_results[query_idx * k + threadIdx.x].dist = shared_dists[threadIdx.x];
        topk_results[query_idx * k + threadIdx.x].id = shared_ids[threadIdx.x];
    }
}


std::vector<GpuSearchResult>
gpu_flat_search_v2(const float* base, const float* query, size_t n, size_t m, size_t d, size_t k) {
    cudaEvent_t start, stop; CUDA_CHECK(cudaEventCreate(&start)); CUDA_CHECK(cudaEventCreate(&stop));
    float time_H2D = 0, time_MatMul = 0, time_TopK = 0, time_D2H = 0;
    
    float *d_base, *d_query, *d_results; GpuSearchResult *d_topk_results;
    CUDA_CHECK(cudaMalloc((void**)&d_base, n * d * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_query, m * d * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_results, n * m * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_topk_results, m * k * sizeof(GpuSearchResult)));

    CUDA_CHECK(cudaEventRecord(start));
    CUDA_CHECK(cudaMemcpy(d_base, base, n * d * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_query, query, m * d * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(stop)); CUDA_CHECK(cudaEventSynchronize(stop)); CUDA_CHECK(cudaEventElapsedTime(&time_H2D, start, stop));

    dim3 blockDimMatMul(BLOCK_DIM_X_V2, BLOCK_DIM_Y_V2);
    dim3 gridDimMatMul((unsigned int)(m + blockDimMatMul.x - 1) / blockDimMatMul.x, (unsigned int)(n + blockDimMatMul.y - 1) / blockDimMatMul.y);
    CUDA_CHECK(cudaEventRecord(start));
    matrix_mul_shared_mem_kernel<<<gridDimMatMul, blockDimMatMul>>>(d_base, d_query, d_results, n, m, d);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaEventRecord(stop)); CUDA_CHECK(cudaEventSynchronize(stop)); CUDA_CHECK(cudaEventElapsedTime(&time_MatMul, start, stop));
    
    dim3 gridDimTopK((unsigned int)m);
    const int topk_block_size = 256;
    dim3 blockDimTopK(topk_block_size);
    size_t shared_mem_size = topk_block_size * k * (sizeof(float) + sizeof(int));
    CUDA_CHECK(cudaEventRecord(start));
    find_topk_kernel_correct<<<gridDimTopK, blockDimTopK, shared_mem_size>>>(d_results, d_topk_results, n, m, k);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaEventRecord(stop)); CUDA_CHECK(cudaEventSynchronize(stop)); CUDA_CHECK(cudaEventElapsedTime(&time_TopK, start, stop));
    
    std::vector<GpuSearchResult> h_topk_results(m * k);
    CUDA_CHECK(cudaEventRecord(start));
    CUDA_CHECK(cudaMemcpy(h_topk_results.data(), d_topk_results, m * k * sizeof(GpuSearchResult), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(stop)); CUDA_CHECK(cudaEventSynchronize(stop)); CUDA_CHECK(cudaEventElapsedTime(&time_D2H, start, stop));
    
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "Performance Breakdown (v2 - Shared Mem + GPU Top-K):" << std::endl;
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "1. Host to Device (H2D) Transfer: " << time_H2D << " ms" << std::endl;
    std::cout << "2. MatMul Kernel (Shared Mem):    " << time_MatMul << " ms" << std::endl;
    std::cout << "3. Top-K Kernel:                  " << time_TopK << " ms" << std::endl;
    std::cout << "4. Device to Host (D2H) Transfer: " << time_D2H << " ms" << std::endl;
    std::cout << "----------------------------------------" << std::endl;

    CUDA_CHECK(cudaFree(d_base)); CUDA_CHECK(cudaFree(d_query)); CUDA_CHECK(cudaFree(d_results)); CUDA_CHECK(cudaFree(d_topk_results));
    CUDA_CHECK(cudaEventDestroy(start)); CUDA_CHECK(cudaEventDestroy(stop));
    return h_topk_results;
}