#pragma once

#include <vector>
#include <iostream>
#include <iomanip>
#include <sys/time.h>
#include <cfloat>
#include "cuda_runtime.h"

#define CUDA_CHECK(err) { if (err != cudaSuccess) { std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " in file " << __FILE__ << " at line " << __LINE__ << std::endl; exit(EXIT_FAILURE); } }

struct GpuSearchResult_v3 {
    float dist;
    int id;
};

const int TILE_WIDTH_V3 = 16;

__global__ void matrix_mul_shared_mem_kernel(const float* base, const float* query, float* results, int n, int m, int d) {
    __shared__ float tile_base[TILE_WIDTH_V3][TILE_WIDTH_V3];
    __shared__ float tile_query[TILE_WIDTH_V3][TILE_WIDTH_V3];
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * blockDim.y + ty;
    int col = blockIdx.x * blockDim.x + tx;
    float sum = 0.0f;

    for (int t = 0; t < (d + TILE_WIDTH_V3 - 1) / TILE_WIDTH_V3; ++t) {
        if (row < n && (t * TILE_WIDTH_V3 + tx) < d) {
            tile_base[ty][tx] = base[row * d + t * TILE_WIDTH_V3 + tx];
        } else {
            tile_base[ty][tx] = 0.0f;
        }
        if (col < m && (t * TILE_WIDTH_V3 + ty) < d) {
            tile_query[ty][tx] = query[col * d + t * TILE_WIDTH_V3 + ty];
        } else {
            tile_query[ty][tx] = 0.0f;
        }
        __syncthreads();
        for (int k_tile = 0; k_tile < TILE_WIDTH_V3; ++k_tile) {
            sum += tile_base[ty][k_tile] * tile_query[k_tile][tx];
        }
        __syncthreads();
    }
    if (row < n && col < m) {
        results[row * m + col] = sum;
    }
}

__global__ void find_topk_kernel_efficient(const float* results_matrix, GpuSearchResult_v3* topk_results,
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

    if (threadIdx.x == 0) {
        float final_dists[10];
        int final_ids[10];
        for (int i = 0; i < k; ++i) {
            final_dists[i] = shared_dists[i];
            final_ids[i] = shared_ids[i];
        }
        for (int i = k; i < blockDim.x * k; ++i) {
            float dist = shared_dists[i];
            int id = shared_ids[i];
            float min_dist = final_dists[0];
            int min_idx = 0;
            for (int j = 1; j < k; ++j) {
                if (final_dists[j] < min_dist) {
                    min_dist = final_dists[j];
                    min_idx = j;
                }
            }
            if (dist > min_dist) {
                final_dists[min_idx] = dist;
                final_ids[min_idx] = id;
            }
        }
        for (int i = 0; i < k; ++i) {
            topk_results[query_idx * k + i].dist = final_dists[i];
            topk_results[query_idx * k + i].id = final_ids[i];
        }
    }
}

std::vector<GpuSearchResult_v3>
gpu_flat_search_v3(const float* base, const float* query, size_t n, size_t m, size_t d, size_t k) {
    cudaEvent_t start, stop; CUDA_CHECK(cudaEventCreate(&start)); CUDA_CHECK(cudaEventCreate(&stop));
    float time_H2D = 0, time_MatMul = 0, time_TopK = 0, time_D2H = 0;
    
    float *d_base, *d_query, *d_results; GpuSearchResult_v3 *d_topk_results;
    CUDA_CHECK(cudaMalloc((void**)&d_base, n * d * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_query, m * d * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_results, n * m * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_topk_results, m * k * sizeof(GpuSearchResult_v3)));

    CUDA_CHECK(cudaEventRecord(start));
    CUDA_CHECK(cudaMemcpy(d_base, base, n * d * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_query, query, m * d * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(stop)); CUDA_CHECK(cudaEventSynchronize(stop)); CUDA_CHECK(cudaEventElapsedTime(&time_H2D, start, stop));

    dim3 blockDimMatMul(TILE_WIDTH_V3, TILE_WIDTH_V3);
    dim3 gridDimMatMul((unsigned int)(m + blockDimMatMul.x - 1) / blockDimMatMul.x, (unsigned int)(n + blockDimMatMul.y - 1) / blockDimMatMul.y);
    CUDA_CHECK(cudaEventRecord(start));
    matrix_mul_shared_mem_kernel<<<gridDimMatMul, blockDimMatMul>>>(d_base, d_query, d_results, n, m, d);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaEventRecord(stop)); CUDA_CHECK(cudaEventSynchronize(stop)); CUDA_CHECK(cudaEventElapsedTime(&time_MatMul, start, stop));
    
    dim3 gridDimTopK((unsigned int)m);
    const int topk_block_size = 256;
    dim3 blockDimTopK(topk_block_size);
    size_t shared_mem_size = topk_block_size * k * (sizeof(float) + sizeof(int));
    CUDA_CHECK(cudaEventRecord(start));
    find_topk_kernel_efficient<<<gridDimTopK, blockDimTopK, shared_mem_size>>>(d_results, d_topk_results, n, m, k);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaEventRecord(stop)); CUDA_CHECK(cudaEventSynchronize(stop)); CUDA_CHECK(cudaEventElapsedTime(&time_TopK, start, stop));
    
    std::vector<GpuSearchResult_v3> h_topk_results(m * k);
    CUDA_CHECK(cudaEventRecord(start));
    CUDA_CHECK(cudaMemcpy(h_topk_results.data(), d_topk_results, m * k * sizeof(GpuSearchResult_v3), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(stop)); CUDA_CHECK(cudaEventSynchronize(stop)); CUDA_CHECK(cudaEventElapsedTime(&time_D2H, start, stop));
    
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "Performance Breakdown (v3 - Efficient Top-K):" << std::endl;
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "1. Host to Device (H2D) Transfer: " << time_H2D << " ms" << std::endl;
    std::cout << "2. MatMul Kernel (Shared Mem):    " << time_MatMul << " ms" << std::endl;
    std::cout << "3. Top-K Kernel (Efficient):      " << time_TopK << " ms" << std::endl;
    std::cout << "4. Device to Host (D2H) Transfer: " << time_D2H << " ms" << std::endl;
    std::cout << "----------------------------------------" << std::endl;

    CUDA_CHECK(cudaFree(d_base)); CUDA_CHECK(cudaFree(d_query)); CUDA_CHECK(cudaFree(d_results)); CUDA_CHECK(cudaFree(d_topk_results));
    CUDA_CHECK(cudaEventDestroy(start)); CUDA_CHECK(cudaEventDestroy(stop));
    return h_topk_results;
}