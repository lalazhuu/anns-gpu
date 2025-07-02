#pragma once

#include <vector>
#include <iostream>
#include <iomanip>
#include <sys/time.h>
#include <cfloat>
#include "cuda_runtime.h"

#define CUDA_CHECK(err) { if (err != cudaSuccess) { std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " in file " << __FILE__ << " at line " << __LINE__ << std::endl; exit(EXIT_FAILURE); } }

struct GpuSearchResult_v8 {
    float dist;
    int id;
};

const int TILE_WIDTH_V8 = 16;
const int WARP_SIZE_V8 = 32;

// --- Kernel 1: 共享内存优化的矩阵乘法  ---
__global__ void matrix_mul_shared_mem_kernel(const float* base, const float* query, float* results, int n, int m, int d) {
    __shared__ float tile_base[TILE_WIDTH_V8][TILE_WIDTH_V8];
    __shared__ float tile_query[TILE_WIDTH_V8][TILE_WIDTH_V8];
    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * blockDim.y + ty, col = blockIdx.x * blockDim.x + tx;
    float sum = 0.0f;
    for (int t = 0; t < (d + TILE_WIDTH_V8 - 1) / TILE_WIDTH_V8; ++t) {
        if (row < n && (t * TILE_WIDTH_V8 + tx) < d) { tile_base[ty][tx] = base[row * d + t * TILE_WIDTH_V8 + tx]; } else { tile_base[ty][tx] = 0.0f; }
        if (col < m && (t * TILE_WIDTH_V8 + ty) < d) { tile_query[ty][tx] = query[col * d + t * TILE_WIDTH_V8 + ty]; } else { tile_query[ty][tx] = 0.0f; }
        __syncthreads();
        for (int k_tile = 0; k_tile < TILE_WIDTH_V8; ++k_tile) { sum += tile_base[ty][k_tile] * tile_query[k_tile][tx]; }
        __syncthreads();
    }
    if (row < n && col < m) { results[row * m + col] = sum; }
}


// --- Kernel 2: GPU 近似Top-K (V8 - 灵活采样 + Warp级合并) ---
template <int K_LOCAL, int SAMPLE_NUMERATOR, int SAMPLE_DENOMINATOR>
__global__ void find_topk_kernel_final_approx(const float* results_matrix, GpuSearchResult_v8* topk_results, int n, int m, int k) {
    int query_idx = blockIdx.x;
    if (query_idx >= m) return;

    // --- 阶段一：线程局部Top-K (大小为K_LOCAL) ---
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

    // --- 阶段二：数据汇集到共享内存 ---
    extern __shared__ float shared_mem[];
    float* shared_dists = shared_mem;
    int* shared_ids = (int*)&shared_dists[k * blockDim.x];

    // 先用负无穷填充所有k个位置
    for (int i = 0; i < k; ++i) {
        shared_dists[i * blockDim.x + threadIdx.x] = -FLT_MAX;
        shared_ids[i * blockDim.x + threadIdx.x] = -1;
    }
    // 只写入K_LOCAL个有效值
    for (int i = 0; i < K_LOCAL; ++i) {
        shared_dists[i * blockDim.x + threadIdx.x] = local_dists[i];
        shared_ids[i * blockDim.x + threadIdx.x] = local_ids[i];
    }
    __syncthreads();

    // --- 阶段三：Warp级并行合并 ---
    if (threadIdx.x < WARP_SIZE_V8) {
        for (int i = threadIdx.x + WARP_SIZE_V8; i < blockDim.x; i += WARP_SIZE_V8) {
            for (int j = 0; j < k; ++j) { // 合并k个元素
                float dist = shared_dists[j * blockDim.x + i];
                int id = shared_ids[j * blockDim.x + i];
                
                float min_dist = shared_dists[0 * blockDim.x + threadIdx.x]; int min_idx = 0;
                for (int l = 1; l < k; ++l) {
                    if (shared_dists[l * blockDim.x + threadIdx.x] < min_dist) { min_dist = shared_dists[l * blockDim.x + threadIdx.x]; min_idx = l; }
                }
                if (dist > min_dist) {
                    shared_dists[min_idx * blockDim.x + threadIdx.x] = dist;
                    shared_ids[min_idx * blockDim.x + threadIdx.x] = id;
                }
            }
        }
    }
    __syncthreads();
    
    // --- 阶段四：Warp内最终归约 ---
    if (threadIdx.x == 0) {
        for (int i = 1; i < WARP_SIZE_V8; ++i) {
             for (int j = 0; j < k; ++j) {
                float dist = shared_dists[j * blockDim.x + i];
                int id = shared_ids[j * blockDim.x + i];
                float min_dist = shared_dists[0]; int min_idx = 0;
                for (int l = 1; l < k; ++l) {
                    if (shared_dists[l] < min_dist) { min_dist = shared_dists[l]; min_idx = l; }
                }
                if (dist > min_dist) { shared_dists[min_idx] = dist; shared_ids[min_idx] = id; }
            }
        }
    }
    __syncthreads();
    
    // --- 阶段五：将最终结果写回全局内存 ---
    if (threadIdx.x < k) {
        topk_results[query_idx * k + threadIdx.x].dist = shared_dists[threadIdx.x];
        topk_results[query_idx * k + threadIdx.x].id = shared_ids[threadIdx.x];
    }
}


// --- 主调用函数 ---
template <int K_LOCAL, int SAMPLE_NUMERATOR, int SAMPLE_DENOMINATOR>
std::vector<GpuSearchResult_v8>
gpu_flat_search_v8(const float* base, const float* query, size_t n, size_t m, size_t d, size_t k) {
    cudaEvent_t start, stop; CUDA_CHECK(cudaEventCreate(&start)); CUDA_CHECK(cudaEventCreate(&stop));
    float time_H2D = 0, time_MatMul = 0, time_TopK = 0, time_D2H = 0;
    
    float *d_base, *d_query, *d_results; GpuSearchResult_v8 *d_topk_results;
    CUDA_CHECK(cudaMalloc((void**)&d_base, n * d * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_query, m * d * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_results, n * m * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_topk_results, m * k * sizeof(GpuSearchResult_v8)));

    CUDA_CHECK(cudaEventRecord(start));
    CUDA_CHECK(cudaMemcpy(d_base, base, n * d * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_query, query, m * d * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(stop)); CUDA_CHECK(cudaEventSynchronize(stop)); CUDA_CHECK(cudaEventElapsedTime(&time_H2D, start, stop));

    dim3 blockDimMatMul(TILE_WIDTH_V8, TILE_WIDTH_V8);
    dim3 gridDimMatMul((unsigned int)(m + blockDimMatMul.x - 1) / blockDimMatMul.x, (unsigned int)(n + blockDimMatMul.y - 1) / blockDimMatMul.y);
    CUDA_CHECK(cudaEventRecord(start));
    matrix_mul_shared_mem_kernel<<<gridDimMatMul, blockDimMatMul>>>(d_base, d_query, d_results, n, m, d);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaEventRecord(stop)); CUDA_CHECK(cudaEventSynchronize(stop)); CUDA_CHECK(cudaEventElapsedTime(&time_MatMul, start, stop));
    
    dim3 gridDimTopK((unsigned int)m);
    const int topk_block_size = 256;
    dim3 blockDimTopK(topk_block_size);
    size_t shared_mem_size = topk_block_size * k * (sizeof(float) + sizeof(int));
    CUDA_CHECK(cudaEventRecord(start));
    find_topk_kernel_final_approx<K_LOCAL, SAMPLE_NUMERATOR, SAMPLE_DENOMINATOR><<<gridDimTopK, blockDimTopK, shared_mem_size>>>(d_results, d_topk_results, n, m, k);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaEventRecord(stop)); CUDA_CHECK(cudaEventSynchronize(stop)); CUDA_CHECK(cudaEventElapsedTime(&time_TopK, start, stop));
    
    std::vector<GpuSearchResult_v8> h_topk_results(m * k);
    CUDA_CHECK(cudaEventRecord(start));
    CUDA_CHECK(cudaMemcpy(h_topk_results.data(), d_topk_results, m * k * sizeof(GpuSearchResult_v8), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(stop)); CUDA_CHECK(cudaEventSynchronize(stop)); CUDA_CHECK(cudaEventElapsedTime(&time_D2H, start, stop));
    
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "Performance Breakdown (v8 - Final Approx, K_LOCAL=" << K_LOCAL << ", RATE=" << SAMPLE_NUMERATOR << "/" << SAMPLE_DENOMINATOR << "):" << std::endl;
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "1. Host to Device (H2D) Transfer: " << time_H2D << " ms" << std::endl;
    std::cout << "2. MatMul Kernel (Shared Mem):    " << time_MatMul << " ms" << std::endl;
    std::cout << "3. Top-K Kernel (Final Approx):   " << time_TopK << " ms" << std::endl;
    std::cout << "4. Device to Host (D2H) Transfer: " << time_D2H << " ms" << std::endl;
    std::cout << "----------------------------------------" << std::endl;

    CUDA_CHECK(cudaFree(d_base)); CUDA_CHECK(cudaFree(d_query)); CUDA_CHECK(cudaFree(d_results)); CUDA_CHECK(cudaFree(d_topk_results));
    CUDA_CHECK(cudaEventDestroy(start)); CUDA_CHECK(cudaEventDestroy(stop));
    return h_topk_results;
}