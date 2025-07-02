#pragma once

#include <vector>
#include <queue>
#include <iostream>
#include <iomanip> // 为了打印格式化输出
#include <sys/time.h> // 为了在头文件中使用 gettimeofday
#include "cuda_runtime.h"

// 将类型别名定义在顶层，这样任何包含此头文件的文件都可以使用它
using MinHeap = std::priority_queue<std::pair<float, int>, std::vector<std::pair<float, int>>, std::greater<std::pair<float, int>>>;

// 宏定义：用于检查CUDA API调用是否成功
#define CUDA_CHECK(err) { \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " in file " << __FILE__ << " at line " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

/**
 * @brief GPU矩阵乘法核函数 (基础版)
 */
__global__ void matrix_mul_kernel(const float* base_vectors, const float* query_vectors, float* results_matrix,
                                  int n, int m, int d)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < m) {
        float sum = 0.0f;
        for (int k = 0; k < d; ++k) {
            sum += base_vectors[row * d + k] * query_vectors[col * d + k];
        }
        results_matrix[row * m + col] = sum;
    }
}


/**
 * @brief GPU暴力搜索(批处理) - 方案一
 */
std::vector<MinHeap>
gpu_flat_search_batch(const float* base, const float* query, size_t base_number, size_t query_number, size_t vecdim, size_t k)
{
    // --- 计时器初始化 ---
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    float time_H2D = 0, time_Kernel = 0, time_D2H = 0;
    
    struct timeval cpu_start, cpu_end;
    double time_CPU_post = 0;

    // --- 1. 在GPU上分配内存 ---
    float *d_base, *d_query, *d_results;
    size_t base_size = base_number * vecdim * sizeof(float);
    size_t query_size = query_number * vecdim * sizeof(float);
    size_t results_size = base_number * query_number * sizeof(float);
    CUDA_CHECK(cudaMalloc((void**)&d_base, base_size));
    CUDA_CHECK(cudaMalloc((void**)&d_query, query_size));
    CUDA_CHECK(cudaMalloc((void**)&d_results, results_size));

    // --- 2. 测量数据上传 (H2D) 时间 ---
    CUDA_CHECK(cudaEventRecord(start));
    CUDA_CHECK(cudaMemcpy(d_base, base, base_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_query, query, query_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&time_H2D, start, stop));

    // --- 3. 设置Kernel启动配置 ---
    const int BLOCK_DIM_X = 16;
    const int BLOCK_DIM_Y = 16;
    dim3 blockDim(BLOCK_DIM_X, BLOCK_DIM_Y);
    dim3 gridDim((unsigned int)(query_number + BLOCK_DIM_X - 1) / BLOCK_DIM_X,
                 (unsigned int)(base_number + BLOCK_DIM_Y - 1) / BLOCK_DIM_Y);

    // --- 4. 测量Kernel执行时间 ---
    CUDA_CHECK(cudaEventRecord(start));
    matrix_mul_kernel<<<gridDim, blockDim>>>(d_base, d_query, d_results, base_number, query_number, vecdim);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&time_Kernel, start, stop));

    // --- 5. 测量数据下载 (D2H) 时间 ---
    float* h_results = new float[base_number * query_number];
    CUDA_CHECK(cudaEventRecord(start));
    CUDA_CHECK(cudaMemcpy(h_results, d_results, results_size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&time_D2H, start, stop));
    
    // --- 6. 在CPU上处理结果，并计时 ---
    gettimeofday(&cpu_start, NULL);
    
    std::vector<MinHeap> final_results(query_number);
    #pragma omp parallel for num_threads(4)
    for (int j = 0; j < query_number; ++j) {
        for (int i = 0; i < base_number; ++i) {
            float dist = h_results[i * query_number + j];
            if (final_results[j].size() < k) {
                final_results[j].push({dist, i});
            } else if (dist > final_results[j].top().first) {
                final_results[j].pop();
                final_results[j].push({dist, i});
            }
        }
    }
    
    gettimeofday(&cpu_end, NULL);
    time_CPU_post = (cpu_end.tv_sec - cpu_start.tv_sec) * 1000.0 + (cpu_end.tv_usec - cpu_start.tv_usec) / 1000.0;

    // --- 打印完整的四阶段耗时报告 ---
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "Performance Breakdown:" << std::endl;
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "1. Host to Device (H2D) Transfer: " << time_H2D << " ms" << std::endl;
    std::cout << "2. Kernel Execution:              " << time_Kernel << " ms" << std::endl;
    std::cout << "3. Device to Host (D2H) Transfer: " << time_D2H << " ms" << std::endl;
    std::cout << "4. CPU Post-processing (Top-K):   " << time_CPU_post << " ms" << std::endl;
    std::cout << "----------------------------------------" << std::endl;

    // --- 7. 释放内存 ---
    delete[] h_results;
    CUDA_CHECK(cudaFree(d_base));
    CUDA_CHECK(cudaFree(d_query));
    CUDA_CHECK(cudaFree(d_results));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return final_results;
}