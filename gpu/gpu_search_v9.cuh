#pragma once

#include <vector>
#include <iostream>
#include <iomanip>
#include <sys/time.h>
#include <cfloat>
#include "cuda_runtime.h"

#define CUDA_CHECK(err) { if (err != cudaSuccess) { std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " in file " << __FILE__ << " at line " << __LINE__ << std::endl; exit(EXIT_FAILURE); } }

struct GpuSearchResult_final {
    float dist;
    int id;
};

// --- Kernel 1: 矩阵乘法 (回归到V8的、稳定可靠的版本) ---
const int TILE_WIDTH_FINAL = 16;
__global__ void matrix_mul_kernel_stable(const float* base, const float* query, float* results, int n, int m, int d) {
    __shared__ float tile_base[TILE_WIDTH_FINAL][TILE_WIDTH_FINAL];
    __shared__ float tile_query[TILE_WIDTH_FINAL][TILE_WIDTH_FINAL];
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * blockDim.y + ty;
    int col = blockIdx.x * blockDim.x + tx;
    float sum = 0.0f;
    for (int t = 0; t < (d + TILE_WIDTH_FINAL - 1) / TILE_WIDTH_FINAL; ++t) {
        if (row < n && (t * TILE_WIDTH_FINAL + tx) < d) {
            tile_base[ty][tx] = base[row * d + t * TILE_WIDTH_FINAL + tx];
        } else { tile_base[ty][tx] = 0.0f; }
        if (col < m && (t * TILE_WIDTH_FINAL + ty) < d) {
            tile_query[ty][tx] = query[col * d + t * TILE_WIDTH_FINAL + ty];
        } else { tile_query[ty][tx] = 0.0f; }
        __syncthreads();
        for (int k_tile = 0; k_tile < TILE_WIDTH_FINAL; ++k_tile) {
            sum += tile_base[ty][k_tile] * tile_query[k_tile][tx];
        }
        __syncthreads();
    }
    if (row < n && col < m) { results[row * m + col] = sum; }
}


// --- Kernel 2: 近似Top-K (V9的、高效的、经过验证的版本) ---
template <int K_LOCAL, int SAMPLE_NUMERATOR, int SAMPLE_DENOMINATOR>
__global__ void find_topk_kernel_final_approx(const float* results_matrix, GpuSearchResult_final* topk_results, int n, int m, int k) {
    int query_idx = blockIdx.x;
    if (query_idx >= m) return;

    float local_dists[K_LOCAL];
    int local_ids[K_LOCAL];
    for (int i = 0; i < K_LOCAL; ++i) { local_dists[i] = -FLT_MAX; local_ids[i] = -1; }

    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        if ((i / blockDim.x) % SAMPLE_DENOMINATOR < SAMPLE_NUMERATOR) {
            float dist = results_matrix[i * m + query_idx];
            float min_dist = local_dists[0];
            int min_idx = 0;
            for (int j = 1; j < K_LOCAL; ++j) {
                if (local_dists[j] < min_dist) { min_dist = local_dists[j]; min_idx = j; }
            }
            if (dist > min_dist) { local_dists[min_idx] = dist; local_ids[min_idx] = i; }
        }
    }

    extern __shared__ float shared_mem[];
    float* shared_dists = shared_mem;
    int* shared_ids = (int*)&shared_dists[k * blockDim.x];

    for (int i = 0; i < k; ++i) {
        shared_dists[i * blockDim.x + threadIdx.x] = -FLT_MAX;
        shared_ids[i * blockDim.x + threadIdx.x] = -1;
    }
    for (int i = 0; i < K_LOCAL; ++i) {
        shared_dists[i * blockDim.x + threadIdx.x] = local_dists[i];
        shared_ids[i * blockDim.x + threadIdx.x] = local_ids[i];
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        float final_dists[10];
        int final_ids[10];
        for (int i = 0; i < k; ++i) { final_dists[i] = -FLT_MAX; final_ids[i] = -1; }
        for (int i = 0; i < blockDim.x * k; ++i) {
            int local_k_idx = i / blockDim.x;
            int thread_idx = i % blockDim.x;
            if(local_k_idx >= K_LOCAL) continue; // Skip padding
            float dist = shared_dists[local_k_idx * blockDim.x + thread_idx];
            int id = shared_ids[local_k_idx * blockDim.x + thread_idx];
            float min_dist_in_final = final_dists[0];
            int min_idx_in_final = 0;
            for (int j = 1; j < k; ++j) {
                if (final_dists[j] < min_dist_in_final) { min_dist_in_final = final_dists[j]; min_idx_in_final = j; }
            }
            if (dist > min_dist_in_final) { final_dists[min_idx_in_final] = dist; final_ids[min_idx_in_final] = id; }
        }
        for (int i = 0; i < k; ++i) {
            topk_results[query_idx * k + i].dist = final_dists[i];
            topk_results[query_idx * k + i].id = final_ids[i];
        }
    }
}


// --- 主调用函数 ---
template <int K_LOCAL, int SAMPLE_NUMERATOR, int SAMPLE_DENOMINATOR>
std::vector<GpuSearchResult_final>
gpu_flat_search_v9(const float* base, const float* query, size_t n, size_t m, size_t d, size_t k) {
    cudaEvent_t start, stop; CUDA_CHECK(cudaEventCreate(&start)); CUDA_CHECK(cudaEventCreate(&stop));
    float time_H2D = 0, time_MatMul = 0, time_TopK = 0, time_D2H = 0;
    
    float *d_base, *d_query, *d_results; GpuSearchResult_final *d_topk_results;
    CUDA_CHECK(cudaMalloc((void**)&d_base, n * d * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_query, m * d * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_results, n * m * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_topk_results, m * k * sizeof(GpuSearchResult_final)));

    CUDA_CHECK(cudaEventRecord(start));
    CUDA_CHECK(cudaMemcpy(d_base, base, n * d * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_query, query, m * d * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(stop)); CUDA_CHECK(cudaEventSynchronize(stop)); CUDA_CHECK(cudaEventElapsedTime(&time_H2D, start, stop));

    dim3 blockDimMatMul(TILE_WIDTH_FINAL, TILE_WIDTH_FINAL);
    dim3 gridDimMatMul((unsigned int)(m + blockDimMatMul.x - 1) / blockDimMatMul.x, (unsigned int)(n + blockDimMatMul.y - 1) / blockDimMatMul.y);
    CUDA_CHECK(cudaEventRecord(start));
    matrix_mul_kernel_stable<<<gridDimMatMul, blockDimMatMul>>>(d_base, d_query, d_results, n, m, d);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaEventRecord(stop)); CUDA_CHECK(cudaEventSynchronize(stop)); CUDA_CHECK(cudaEventElapsedTime(&time_MatMul, start, stop));
    
    dim3 gridDimTopK((unsigned int)m);
    const int topk_block_size = 256;
    dim3 blockDimTopK(topk_block_size);
    size_t shared_mem_size = topk_block_size * k * (sizeof(float) + sizeof(int));
    CUDA_CHECK(cudaEventRecord(start));
    find_topk_kernel_final_approx<K_LOCAL, SAMPLE_NUMERATOR, SAMPLE_DENOMINATOR><<<gridDimTopK, blockDimTopK, shared_mem_size>>>(d_results, d_topk_results, n, m, k);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaEventRecord(stop)); CUDA_CHECK(cudaEventSynchronize(stop)); CUDA_CHECK(cudaEventElapsedTime(&time_TopK, start, stop));
    
    std::vector<GpuSearchResult_final> h_topk_results(m * k);
    CUDA_CHECK(cudaMemcpy(h_topk_results.data(), d_topk_results, m * k * sizeof(GpuSearchResult_final), cudaMemcpyDeviceToHost));
    
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "Performance Breakdown (Final Version, K_LOCAL=" << K_LOCAL << ", RATE=" << SAMPLE_NUMERATOR << "/" << SAMPLE_DENOMINATOR << "):" << std::endl;
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "1. Host to Device (H2D) Transfer: " << time_H2D << " ms" << std::endl;
    std::cout << "2. MatMul Kernel (Stable):        " << time_MatMul << " ms" << std::endl;
    std::cout << "3. Top-K Kernel (Approx):         " << time_TopK << " ms" << std::endl;
    std::cout << "4. Device to Host (D2H) Transfer: " << time_D2H << " ms" << std::endl;
    std::cout << "----------------------------------------" << std::endl;

    CUDA_CHECK(cudaFree(d_base)); CUDA_CHECK(cudaFree(d_query)); CUDA_CHECK(cudaFree(d_results));
    CUDA_CHECK(cudaFree(d_topk_results));
    CUDA_CHECK(cudaEventDestroy(start)); CUDA_CHECK(cudaEventDestroy(stop));
    return h_topk_results;
}