#pragma once

#include <vector>
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

// --- Kernel 1: 共享内存优化的矩阵乘法 (保持不变) ---
const int TILE_WIDTH_FINAL = 16;
__global__ void matrix_mul_shared_mem_kernel(const float* base, const float* query, float* results, int n, int m, int d);


// --- Kernel 2: GPU Top-K (最终优化版: Warp级并行合并) ---
__global__ void find_topk_kernel_warp_reduce(const float* results_matrix, GpuSearchResult* topk_results, int n, int m, int k,
                                             long long* timing_results) {
    int query_idx = blockIdx.x;
    if (query_idx >= m) return;

    long long start_time, time1, time2, time3, time4;
    if (threadIdx.x == 0) start_time = clock64();

    // --- 阶段一：线程局部Top-K ---
    float local_dists[10]; int local_ids[10];
    for (int i = 0; i < k; ++i) { local_dists[i] = -FLT_MAX; local_ids[i] = -1; }
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float dist = results_matrix[i * m + query_idx];
        float min_dist = local_dists[0]; int min_idx = 0;
        for (int j = 1; j < k; ++j) {
            if (local_dists[j] < min_dist) { min_dist = local_dists[j]; min_idx = j; }
        }
        if (dist > min_dist) { local_dists[min_idx] = dist; local_ids[min_idx] = i; }
    }

    if (threadIdx.x == 0) time1 = clock64();

    // --- 阶段二：数据汇集 (优化布局避免Bank冲突) ---
    extern __shared__ float shared_mem[];
    float* shared_dists = shared_mem;
    int* shared_ids = (int*)&shared_dists[k * blockDim.x];
    for (int i = 0; i < k; ++i) {
        shared_dists[i * blockDim.x + threadIdx.x] = local_dists[i];
        shared_ids[i * blockDim.x + threadIdx.x] = local_ids[i];
    }
    __syncthreads();

    if (threadIdx.x == 0) time2 = clock64();

    // --- 阶段三：Warp级并行合并 ---
    const int WARP_SIZE = 32;
    if (threadIdx.x < WARP_SIZE) {
        // 每个Warp线程负责合并 blockDim.x / WARP_SIZE 份数据
        for (int i = threadIdx.x + WARP_SIZE; i < blockDim.x; i += WARP_SIZE) {
            for (int j = 0; j < k; ++j) {
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

    if (threadIdx.x == 0) time3 = clock64();

    // --- 阶段四：Warp内最终归约 ---
    if (threadIdx.x == 0) {
        // 合并Warp内32个线程的结果
        for (int i = 1; i < WARP_SIZE; ++i) {
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
        // 将最终结果写回全局内存
        for (int i = 0; i < k; ++i) {
            topk_results[query_idx * k + i].dist = shared_dists[i];
            topk_results[query_idx * k + i].id = shared_ids[i];
        }
    }
    
    if (threadIdx.x == 0) {
        time4 = clock64();
        timing_results[query_idx * 4 + 0] = time1 - start_time;
        timing_results[query_idx * 4 + 1] = time2 - time1;
        timing_results[query_idx * 4 + 2] = time3 - time2;
        timing_results[query_idx * 4 + 3] = time4 - time3;
    }
}


// --- 主调用函数 ---
std::vector<GpuSearchResult>
gpu_flat_search_final(const float* base, const float* query, size_t n, size_t m, size_t d, size_t k) {
    cudaEvent_t start, stop; CUDA_CHECK(cudaEventCreate(&start)); CUDA_CHECK(cudaEventCreate(&stop));
    float time_H2D = 0, time_MatMul = 0, time_TopK = 0, time_D2H = 0;
    
    float *d_base, *d_query, *d_results; GpuSearchResult *d_topk_results;
    long long* d_timing_results;
    CUDA_CHECK(cudaMalloc((void**)&d_base, n * d * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_query, m * d * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_results, n * m * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_topk_results, m * k * sizeof(GpuSearchResult)));
    CUDA_CHECK(cudaMalloc((void**)&d_timing_results, m * 4 * sizeof(long long)));

    CUDA_CHECK(cudaEventRecord(start));
    CUDA_CHECK(cudaMemcpy(d_base, base, n * d * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_query, query, m * d * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(stop)); CUDA_CHECK(cudaEventSynchronize(stop)); CUDA_CHECK(cudaEventElapsedTime(&time_H2D, start, stop));

    dim3 blockDimMatMul(TILE_WIDTH_FINAL, TILE_WIDTH_FINAL);
    dim3 gridDimMatMul((unsigned int)(m + blockDimMatMul.x - 1) / blockDimMatMul.x, (unsigned int)(n + blockDimMatMul.y - 1) / blockDimMatMul.y);
    CUDA_CHECK(cudaEventRecord(start));
    matrix_mul_shared_mem_kernel<<<gridDimMatMul, blockDimMatMul>>>(d_base, d_query, d_results, n, m, d);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaEventRecord(stop)); CUDA_CHECK(cudaEventSynchronize(stop)); CUDA_CHECK(cudaEventElapsedTime(&time_MatMul, start, stop));
    
    dim3 gridDimTopK((unsigned int)m);
    const int topk_block_size = 256;
    dim3 blockDimTopK(topk_block_size);
    size_t shared_mem_size = topk_block_size * k * (sizeof(float) + sizeof(int));
    CUDA_CHECK(cudaEventRecord(start));
    find_topk_kernel_warp_reduce<<<gridDimTopK, blockDimTopK, shared_mem_size>>>(d_results, d_topk_results, n, m, k, d_timing_results);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaEventRecord(stop)); CUDA_CHECK(cudaEventSynchronize(stop)); CUDA_CHECK(cudaEventElapsedTime(&time_TopK, start, stop));
    
    std::vector<GpuSearchResult> h_topk_results(m * k);
    std::vector<long long> h_timing_results(m * 4);
    CUDA_CHECK(cudaEventRecord(start));
    CUDA_CHECK(cudaMemcpy(h_topk_results.data(), d_topk_results, m * k * sizeof(GpuSearchResult), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_timing_results.data(), d_timing_results, m * 4 * sizeof(long long), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(stop)); CUDA_CHECK(cudaEventSynchronize(stop)); CUDA_CHECK(cudaEventElapsedTime(&time_D2H, start, stop));
    
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "Performance Breakdown (Final - Warp Reduce):" << std::endl;
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "1. H2D Transfer:                  " << time_H2D << " ms" << std::endl;
    std::cout << "2. MatMul Kernel (Shared Mem):    " << time_MatMul << " ms" << std::endl;
    std::cout << "3. Top-K Kernel (Warp-Reduce):    " << time_TopK << " ms" << std::endl;
    std::cout << "4. D2H Transfer:                  " << time_D2H << " ms" << std::endl;
    
    double avg_time1 = 0, avg_time2 = 0, avg_time3 = 0, avg_time4 = 0;
    for(size_t i=0; i<m; ++i) {
        avg_time1 += h_timing_results[i * 4 + 0];
        avg_time2 += h_timing_results[i * 4 + 1];
        avg_time3 += h_timing_results[i * 4 + 2];
        avg_time4 += h_timing_results[i * 4 + 3];
    }
    avg_time1 /= m; avg_time2 /= m; avg_time3 /= m; avg_time4 /= m;
    std::cout << "--- Top-K Kernel Internal Breakdown (Cycles) ---" << std::endl;
    std::cout << "Phase 1 (Local Top-K):            " << std::fixed << std::setprecision(0) << avg_time1 << " cycles" << std::endl;
    std::cout << "Phase 2 (Data Gathering):         " << avg_time2 << " cycles" << std::endl;
    std::cout << "Phase 3 (Warp-Parallel Merge):    " << avg_time3 << " cycles" << std::endl;
    std::cout << "Phase 4 (Final Merge):            " << avg_time4 << " cycles" << std::endl;
    std::cout << "----------------------------------------" << std::endl;

    CUDA_CHECK(cudaFree(d_base)); CUDA_CHECK(cudaFree(d_query)); CUDA_CHECK(cudaFree(d_results));
    CUDA_CHECK(cudaFree(d_topk_results)); CUDA_CHECK(cudaFree(d_timing_results));
    CUDA_CHECK(cudaEventDestroy(start)); CUDA_CHECK(cudaEventDestroy(stop));
    return h_topk_results;
}


// Kernel 1的完整定义
__global__ void matrix_mul_shared_mem_kernel(const float* base, const float* query, float* results, int n, int m, int d) {
    __shared__ float tile_base[TILE_WIDTH_FINAL][TILE_WIDTH_FINAL];
    __shared__ float tile_query[TILE_WIDTH_FINAL][TILE_WIDTH_FINAL];
    int tx = threadIdx.x; int ty = threadIdx.y;
    int row = blockIdx.y * blockDim.y + ty; int col = blockIdx.x * blockDim.x + tx;
    float sum = 0.0f;
    for (int t = 0; t < (d + TILE_WIDTH_FINAL - 1) / TILE_WIDTH_FINAL; ++t) {
        if (row < n && (t * TILE_WIDTH_FINAL + tx) < d) { tile_base[ty][tx] = base[row * d + t * TILE_WIDTH_FINAL + tx]; } else { tile_base[ty][tx] = 0.0f; }
        if (col < m && (t * TILE_WIDTH_FINAL + ty) < d) { tile_query[ty][tx] = query[col * d + t * TILE_WIDTH_FINAL + ty]; } else { tile_query[ty][tx] = 0.0f; }
        __syncthreads();
        for (int k_tile = 0; k_tile < TILE_WIDTH_FINAL; ++k_tile) { sum += tile_base[ty][k_tile] * tile_query[k_tile][tx]; }
        __syncthreads();
    }
    if (row < n && col < m) { results[row * m + col] = sum; }
}