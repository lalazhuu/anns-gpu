#ifndef IVF_PQ_GPU_V2_H
#define IVF_PQ_GPU_V2_H

#include <vector>
#include <queue>
#include <iostream>
#include <fstream>
#include <string>
#include <random>
#include <algorithm>
#include <cmath>
#include <omp.h>
#include <cstring>
#include <map>
#include <set>
#include <iomanip>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cublas_v2.h"
#include "ivf_pq_common.cuh"

// --- 辅助宏 (V2版本) ---
#define CHECK_CUDA_PQ_V2(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "PQ_V2 CUDA Error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

#define CHECK_CUBLAS_PQ_V2(call) do { \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "PQ_V2 cuBLAS Error at %s:%d (code %d)\n", __FILE__, __LINE__, status); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

// --- Kernel 函数声明 ---
// 我们将在这个文件中明确声明所有用到的Kernel，以保证编译顺序正确
#include "ivf_pq_kernels.cuh" 
#include "ivf_pq_kernels_v2.cuh" 
// in ivf_pq_gpu_v2.cuh, after #includes
__global__ void gemm_kernel(const float* A, const float* B, float* C, int M, int N, int K, float alpha);
// --- 主机端封装类 V2 ---
class IvfPqGpuV2 {
public:
    IvfPqGpuV2(size_t dim, int num_clusters);
    ~IvfPqGpuV2();

    void Build(const float* base, size_t N);
    bool LoadIndex(const std::string& path_prefix, size_t N);
    void SaveIndex(const std::string& path_prefix);
    void TransferToDevice();
    
    void search_batch(
        const float* queries, size_t M, int k, int nprobe,
        int rerank_top_n,
        std::vector<std::priority_queue<std::pair<float, int>>>& final_results,
        std::vector<float>& timings_ms,
        bool debug_mode = false // 新增参数
    );

private:
    // 主机端训练/编码辅助函数
    void run_ivf_kmeans(const float* base, std::vector<int>& assignments);
    void train_pq_codebooks_on_residuals(const float* base, const std::vector<int>& assignments);
    void encode_database_residuals(const float* base, const std::vector<int>& assignments);
    
    // GPU计算辅助函数
    void compute_l2_dist_matrix(const float* d_queries, size_t M, float* d_dist_matrix);

    size_t D_, N_;
    int num_clusters_;
    cublasHandle_t cublas_handle_;
    
    std::vector<float> h_centroids_, h_reordered_base_, h_pq_codebooks_;
    std::vector<int> h_list_offsets_, h_list_sizes_, h_original_ids_;
    std::vector<unsigned char> h_pq_codes_;

    float* d_centroids_ = nullptr;
    float* d_reordered_base_ = nullptr;
    int* d_original_ids_ = nullptr;
    int* d_list_offsets_ = nullptr;
    int* d_list_sizes_ = nullptr;
    float* d_pq_codebooks_ = nullptr;
    unsigned char* d_pq_codes_ = nullptr;
};

// =================================================================
//                      构造/析构 函数实现
// =================================================================
IvfPqGpuV2::IvfPqGpuV2(size_t dim, int num_clusters) : D_(dim), N_(0), num_clusters_(num_clusters) {
    if (D_ % PQ_M != 0) {
        std::cerr << "V2: Dimension " << D_ << " is not divisible by PQ_M " << PQ_M << std::endl;
        exit(EXIT_FAILURE);
    }
    CHECK_CUBLAS_PQ_V2(cublasCreate(&cublas_handle_));
}

IvfPqGpuV2::~IvfPqGpuV2() {
    CHECK_CUBLAS_PQ_V2(cublasDestroy(cublas_handle_));
    if (d_centroids_) CHECK_CUDA_PQ_V2(cudaFree(d_centroids_));
    if (d_reordered_base_) CHECK_CUDA_PQ_V2(cudaFree(d_reordered_base_));
    if (d_original_ids_) CHECK_CUDA_PQ_V2(cudaFree(d_original_ids_));
    if (d_list_offsets_) CHECK_CUDA_PQ_V2(cudaFree(d_list_offsets_));
    if (d_list_sizes_) CHECK_CUDA_PQ_V2(cudaFree(d_list_sizes_));
    if (d_pq_codebooks_) CHECK_CUDA_PQ_V2(cudaFree(d_pq_codebooks_));
    if (d_pq_codes_) CHECK_CUDA_PQ_V2(cudaFree(d_pq_codes_));
}


// =================================================================
//          核心函数 search_batch V2 (重写版)
// =================================================================
// in ivf_pq_gpu_v2.cuh
void IvfPqGpuV2::search_batch(
    const float* queries, size_t M, int k, int nprobe,
    int rerank_top_n,
    std::vector<std::priority_queue<std::pair<float, int>>>& final_results,
    std::vector<float>& timings_ms,
    bool debug_mode) // 新增调试标志参数
{
    // --- 0. 初始化 ---
    cudaEvent_t start, stop;
    CHECK_CUDA_PQ_V2(cudaEventCreate(&start));
    CHECK_CUDA_PQ_V2(cudaEventCreate(&stop));
    timings_ms.clear(); timings_ms.reserve(5);

    // --- 1. 分配GPU内存 ---
    float* d_queries;         CHECK_CUDA_PQ_V2(cudaMalloc(&d_queries, (long long)M * D_ * sizeof(float)));
    float* d_dist_matrix;     CHECK_CUDA_PQ_V2(cudaMalloc(&d_dist_matrix, (long long)M * num_clusters_ * sizeof(float)));
    int*   d_top_probes;      CHECK_CUDA_PQ_V2(cudaMalloc(&d_top_probes, (long long)M * nprobe * sizeof(int)));
    ResultPair* d_rerank_cand;CHECK_CUDA_PQ_V2(cudaMalloc(&d_rerank_cand, (long long)M * rerank_top_n * sizeof(ResultPair)));
    int* d_rerank_offsets;    CHECK_CUDA_PQ_V2(cudaMalloc(&d_rerank_offsets, (long long)M * rerank_top_n * sizeof(int)));
    ResultPair* d_final;      CHECK_CUDA_PQ_V2(cudaMalloc(&d_final, (long long)M * k * sizeof(ResultPair)));
    
    // --- 计时：H2D Transfer ---
    CHECK_CUDA_PQ_V2(cudaEventRecord(start));
    CHECK_CUDA_PQ_V2(cudaMemcpy(d_queries, queries, (long long)M * D_ * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_PQ_V2(cudaEventRecord(stop)); CHECK_CUDA_PQ_V2(cudaEventSynchronize(stop));
    float time_h2d; CHECK_CUDA_PQ_V2(cudaEventElapsedTime(&time_h2d, start, stop));
    timings_ms.push_back(time_h2d);

    // --- 计时：S1 - Probe Selection ---
    CHECK_CUDA_PQ_V2(cudaEventRecord(start));
    // 使用手写GEMM的、逻辑清晰的版本
    compute_l2_dist_matrix(d_queries, M, d_dist_matrix);
    dim3 gridDim_s1(M); dim3 blockDim_s1(256);
    size_t smem_s1 = blockDim_s1.x * nprobe * sizeof(ResultPair);
    find_top_n_probes_dist_kernel<<<gridDim_s1, blockDim_s1, smem_s1>>>(d_dist_matrix, M, num_clusters_, nprobe, d_top_probes);
    CHECK_CUDA_PQ_V2(cudaGetLastError());
    CHECK_CUDA_PQ_V2(cudaEventRecord(stop)); CHECK_CUDA_PQ_V2(cudaEventSynchronize(stop));
    float time_s1; CHECK_CUDA_PQ_V2(cudaEventElapsedTime(&time_s1, start, stop));
    timings_ms.push_back(time_s1);
    
    // --- [V2 核心] 预处理和分配临时存储 ---
    std::vector<int> h_top_probes(M * nprobe);
    CHECK_CUDA_PQ_V2(cudaMemcpy(h_top_probes.data(), d_top_probes, M * nprobe * sizeof(int), cudaMemcpyDeviceToHost));
    
    int max_points_in_probes = 0;
    for(size_t i = 0; i < M; ++i) {
        int current_total = 0;
        for (int j = 0; j < nprobe; ++j) {
            int cluster_id = h_top_probes[i * nprobe + j];
            if (cluster_id >= 0) {
                current_total += h_list_sizes_[cluster_id];
            }
        }
        if (current_total > max_points_in_probes) {
            max_points_in_probes = current_total;
        }
    }

    float* d_temp_storage;
    CHECK_CUDA_PQ_V2(cudaMalloc(&d_temp_storage, (long long)M * max_points_in_probes * sizeof(float)));
    CHECK_CUDA_PQ_V2(cudaMemset(d_temp_storage, 0, (long long)M * max_points_in_probes * sizeof(float)));

    // --- 计时：S2 - Fused ADC Search ---
    CHECK_CUDA_PQ_V2(cudaEventRecord(start));

    // V2 FINAL KERNEL: One block per query
    dim3 gridDim_s2(M); 
    dim3 blockDim_s2(256); // 256 or 512 are good starting points

    // Shared memory for LUT + block-level candidate reduction
    // The capacity of s_bests is rerank_top_n * 2
    size_t smem_s2 = (PQ_M * PQ_KS * sizeof(float)) + (blockDim_s2.x * sizeof(ResultPair));

    final_simulated_and_verified_kernel<<<gridDim_s2, blockDim_s2, smem_s2>>>(
        d_queries, d_centroids_, d_pq_codebooks_, d_pq_codes_, d_list_offsets_, d_list_sizes_, d_top_probes,
        M, D_, nprobe, rerank_top_n, 
        d_rerank_cand // Output is directly the final rerank candidates
    );
    CHECK_CUDA_PQ_V2(cudaGetLastError());

    CHECK_CUDA_PQ_V2(cudaEventRecord(stop)); CHECK_CUDA_PQ_V2(cudaEventSynchronize(stop));
    float time_s2; CHECK_CUDA_PQ_V2(cudaEventElapsedTime(&time_s2, start, stop));
    timings_ms.push_back(time_s2);



    // --- 计时：S3 - Rerank ---
    CHECK_CUDA_PQ_V2(cudaEventRecord(start));
    extract_offsets_kernel<<<(M * rerank_top_n + 255) / 256, 256>>>(d_rerank_cand, d_rerank_offsets, M, rerank_top_n);
    dim3 gridDim_s3(M); dim3 blockDim_s3(256);
    size_t smem_s3 = rerank_top_n * sizeof(ResultPair);
    rerank_kernel<<<gridDim_s3, blockDim_s3, smem_s3>>>(d_queries, d_reordered_base_, d_original_ids_, d_rerank_offsets, M, D_, k, rerank_top_n, d_final);
    CHECK_CUDA_PQ_V2(cudaGetLastError());
    CHECK_CUDA_PQ_V2(cudaEventRecord(stop)); CHECK_CUDA_PQ_V2(cudaEventSynchronize(stop));
    float time_s3; CHECK_CUDA_PQ_V2(cudaEventElapsedTime(&time_s3, start, stop));
    timings_ms.push_back(time_s3);

    // --- 计时：D2H Transfer ---
    std::vector<ResultPair> h_results( (long long)M * k );
    CHECK_CUDA_PQ_V2(cudaEventRecord(start));
    CHECK_CUDA_PQ_V2(cudaMemcpy(h_results.data(), d_final, (long long)M * k * sizeof(ResultPair), cudaMemcpyDeviceToHost));
    CHECK_CUDA_PQ_V2(cudaEventRecord(stop)); CHECK_CUDA_PQ_V2(cudaEventSynchronize(stop));
    float time_d2h; CHECK_CUDA_PQ_V2(cudaEventElapsedTime(&time_d2h, start, stop));
    timings_ms.push_back(time_d2h);
    
    // --- CPU端后处理 ---
    final_results.assign(M, std::priority_queue<std::pair<float, int>>());
    for (size_t i = 0; i < M; ++i) {
        for (int j = 0; j < k; ++j) {
            ResultPair p = h_results[(long long)i * k + j];
            if(p.id != -1) {
                final_results[i].push({ -p.distance, p.id });
            }
        }
    }

    // --- 清理GPU内存 ---
    CHECK_CUDA_PQ_V2(cudaFree(d_queries));
    CHECK_CUDA_PQ_V2(cudaFree(d_dist_matrix));
    CHECK_CUDA_PQ_V2(cudaFree(d_top_probes));
    CHECK_CUDA_PQ_V2(cudaFree(d_rerank_cand));
    CHECK_CUDA_PQ_V2(cudaFree(d_rerank_offsets));
    CHECK_CUDA_PQ_V2(cudaFree(d_final));
    CHECK_CUDA_PQ_V2(cudaEventDestroy(start));
    CHECK_CUDA_PQ_V2(cudaEventDestroy(stop));
}

// =================================================================
//        GPU 计算辅助函数 (重写版，确保维度正确)
// =================================================================
// void IvfPqGpuV2::compute_l2_dist_matrix(const float* d_queries, size_t M, float* d_dist_matrix) {
//     // 目标: 计算 d_dist_matrix (M x num_clusters_)
    
//     // 1. 计算 -2 * Q * C^T
//     const float alpha = -2.0f;
//     dim3 grid_gemm( (num_clusters_ + 15) / 16, (M + 15) / 16 );
//     dim3 block_gemm(16, 16);
//     gemm_kernel<<<grid_gemm, block_gemm>>>(d_queries, d_centroids_, d_dist_matrix, M, num_clusters_, D_, alpha);
//     CHECK_CUDA_PQ_V2(cudaGetLastError());

//     // 2. 计算范数
//     float* d_queries_norm_sq;
//     CHECK_CUDA_PQ_V2(cudaMalloc(&d_queries_norm_sq, M * sizeof(float)));
//     l2_norm_squared_kernel<<< (M + 255) / 256, 256 >>>(d_queries, d_queries_norm_sq, M, D_);
    
//     float* d_centroids_norm_sq;
//     CHECK_CUDA_PQ_V2(cudaMalloc(&d_centroids_norm_sq, num_clusters_ * sizeof(float)));
//     l2_norm_squared_kernel<<< (num_clusters_ + 255) / 256, 256 >>>(d_centroids_, d_centroids_norm_sq, num_clusters_, D_);
//     CHECK_CUDA_PQ_V2(cudaGetLastError());

//     // 3. 对 M x num_clusters_ 矩阵进行广播加法
//     dim3 grid_broadcast( (num_clusters_ + 15) / 16, (M + 15) / 16 );
//     dim3 block_broadcast(16, 16);
    
//     // 把 ||Q||^2 (M x 1) 按列广播
//     add_broadcast_col_kernel<<<grid_broadcast, block_broadcast>>>(d_dist_matrix, d_queries_norm_sq, M, num_clusters_);
//     CHECK_CUDA_PQ_V2(cudaGetLastError());

//     // [!!! 根本性修正 !!!]
//     // 把 ||C||^2 (num_clusters x 1) 按行广播
//     add_broadcast_row_kernel<<<grid_broadcast, block_broadcast>>>(d_dist_matrix, d_centroids_norm_sq, M, num_clusters_);
//     CHECK_CUDA_PQ_V2(cudaGetLastError());

//     // 4. 清理
//     CHECK_CUDA_PQ_V2(cudaFree(d_queries_norm_sq));
//     CHECK_CUDA_PQ_V2(cudaFree(d_centroids_norm_sq));
// }

// 替换 ivf_pq_gpu_v2.cuh 中的 compute_l2_dist_matrix 函数

void IvfPqGpuV2::compute_l2_dist_matrix(const float* d_queries, size_t M, float* d_dist_matrix) {
    // 目标: 计算 d_dist_matrix = ||Q||^2 - 2*Q*C^T + ||C||^2
    // --- 1. 使用 cuBLAS 计算交叉项 -2 * Q * C^T ---
    const float alpha = -2.0f;
    const float beta = 0.0f;

    // 这个调用是经过V1验证的、正确的、利用了C/Fortran内存布局等价性的版本。
    // 它计算的是 D_c^T = -2 * C_c * Q_c^T，但因为内存布局，结果可以直接当作 D_c 使用。
    CHECK_CUBLAS_PQ_V2(cublasSgemm(cublas_handle_,
                              CUBLAS_OP_T,       // Transpose Centroids
                              CUBLAS_OP_N,       // No Transpose Queries
                              num_clusters_, M, D_, // m, n, k
                              &alpha,
                              d_centroids_, D_,   // A, lda
                              d_queries, D_,      // B, ldb
                              &beta,
                              d_dist_matrix, num_clusters_)); // C, ldc
    CHECK_CUDA_PQ_V2(cudaGetLastError());

    // --- 2. 计算范数 (这部分逻辑一直是正确的) ---
    float* d_queries_norm_sq;
    CHECK_CUDA_PQ_V2(cudaMalloc(&d_queries_norm_sq, M * sizeof(float)));
    l2_norm_squared_kernel<<< (M + 255) / 256, 256 >>>(d_queries, d_queries_norm_sq, M, D_);
    
    float* d_centroids_norm_sq;
    CHECK_CUDA_PQ_V2(cudaMalloc(&d_centroids_norm_sq, num_clusters_ * sizeof(float)));
    l2_norm_squared_kernel<<< (num_clusters_ + 255) / 256, 256 >>>(d_centroids_, d_centroids_norm_sq, num_clusters_, D_);
    CHECK_CUDA_PQ_V2(cudaGetLastError());

    // --- 3. 广播加法 (这部分逻辑一直是正确的) ---
    dim3 grid_broadcast( (num_clusters_ + 15) / 16, (M + 15) / 16 );
    dim3 block_broadcast(16, 16);
    
    add_broadcast_col_kernel<<<grid_broadcast, block_broadcast>>>(d_dist_matrix, d_queries_norm_sq, M, num_clusters_);
    CHECK_CUDA_PQ_V2(cudaGetLastError());

    add_broadcast_row_kernel<<<grid_broadcast, block_broadcast>>>(d_dist_matrix, d_centroids_norm_sq, M, num_clusters_);
    CHECK_CUDA_PQ_V2(cudaGetLastError());

    // --- 4. 清理 ---
    CHECK_CUDA_PQ_V2(cudaFree(d_queries_norm_sq));
    CHECK_CUDA_PQ_V2(cudaFree(d_centroids_norm_sq));
}
// =================================================================
//        主机端训练/编码/IO函数实现 (与V1一致)
// =================================================================
float euclidean_dist_sq_v2(const float* v1, const float* v2, size_t D) {
    float dist_sq = 0.0f;
    for (size_t i = 0; i < D; ++i) {
        float diff = v1[i] - v2[i];
        dist_sq += diff * diff;
    }
    return dist_sq;
}

void IvfPqGpuV2::run_ivf_kmeans(const float* base, std::vector<int>& assignments) {
    std::cout << "PQ_Build: Starting IVF K-Means (L2 distance)...\n";
    h_centroids_.resize((long long)num_clusters_ * D_);
    std::vector<int> random_indices(N_);
    for(size_t i=0; i<N_; ++i) random_indices[i] = i;
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(random_indices.begin(), random_indices.end(), g);
    for(int i=0; i<num_clusters_; ++i) {
        memcpy(h_centroids_.data() + (long long)i * D_, base + (long long)random_indices[i] * D_, D_ * sizeof(float));
    }
    int max_iters = 50;
    for (int iter = 0; iter < max_iters; ++iter) {
        #pragma omp parallel for
        for (long long i = 0; i < N_; ++i) {
            float min_dist = 1e9f;
            int best_cluster = -1;
            for (int c = 0; c < num_clusters_; ++c) {
                float dist = euclidean_dist_sq_v2(base + i * D_, h_centroids_.data() + c * D_, D_);
                if (dist < min_dist) {
                    min_dist = dist;
                    best_cluster = c;
                }
            }
            assignments[i] = best_cluster;
        }
        h_centroids_.assign((long long)num_clusters_ * D_, 0.0f);
        std::vector<int> cluster_counts(num_clusters_, 0);
        for (size_t i = 0; i < N_; ++i) {
            int cluster_id = assignments[i];
            for (size_t d = 0; d < D_; ++d) h_centroids_[(long long)cluster_id * D_ + d] += base[(long long)i * D_ + d];
            cluster_counts[cluster_id]++;
        }
        #pragma omp parallel for
        for (int c = 0; c < num_clusters_; ++c) {
            if (cluster_counts[c] > 0) {
                for (size_t d = 0; d < D_; ++d) h_centroids_[(long long)c * D_ + d] /= cluster_counts[c];
            }
        }
    }
    std::cout << "PQ_Build: IVF K-Means finished.\n";
}

void IvfPqGpuV2::train_pq_codebooks_on_residuals(const float* base, const std::vector<int>& assignments) {
    std::cout << "PQ_Build: Starting PQ codebook training on residuals...\n";
    h_pq_codebooks_.resize((long long)num_clusters_ * PQ_M * PQ_KS * PQ_DS);
    h_pq_codebooks_.assign(h_pq_codebooks_.size(), 0.0f);
    #pragma omp parallel for
    for (int c = 0; c < num_clusters_; ++c) {
        std::vector<const float*> vectors_in_cluster;
        for (size_t i = 0; i < N_; ++i) {
            if (assignments[i] == c) vectors_in_cluster.push_back(base + i * D_);
        }
        size_t num_vectors_in_cluster = vectors_in_cluster.size();
        if (num_vectors_in_cluster == 0) continue;
        const float* centroid_vec = h_centroids_.data() + (long long)c * D_;
        for (int m = 0; m < PQ_M; ++m) {
            std::vector<float> sub_residuals(num_vectors_in_cluster * PQ_DS);
            for (size_t i = 0; i < num_vectors_in_cluster; ++i) {
                const float* original_vec = vectors_in_cluster[i];
                for (int d = 0; d < PQ_DS; ++d) {
                    sub_residuals[i * PQ_DS + d] = original_vec[m * PQ_DS + d] - centroid_vec[m * PQ_DS + d];
                }
            }
            float* sub_codebook = h_pq_codebooks_.data() + (((long long)c * PQ_M + m) * PQ_KS * PQ_DS);
            std::vector<int> random_indices(num_vectors_in_cluster);
            for(size_t i=0; i<num_vectors_in_cluster; ++i) random_indices[i] = i;
            std::random_device rd;
            std::mt19937 g(rd());
            std::shuffle(random_indices.begin(), random_indices.end(), g);
            for(int ks=0; ks<PQ_KS; ++ks) {
                 memcpy(sub_codebook + ks * PQ_DS, 
                        sub_residuals.data() + random_indices[ks % num_vectors_in_cluster] * PQ_DS,
                        PQ_DS * sizeof(float));
            }
            std::vector<int> pq_assignments(num_vectors_in_cluster);
            int pq_iters = 10;
            for(int iter=0; iter < pq_iters; ++iter) {
                for(size_t i=0; i<num_vectors_in_cluster; ++i) {
                    float min_dist_sq = 1e9f; int best_ks = -1;
                    for(int ks=0; ks<PQ_KS; ++ks) {
                        float dist_sq = euclidean_dist_sq_v2(sub_residuals.data() + i * PQ_DS, sub_codebook + ks * PQ_DS, PQ_DS);
                        if(dist_sq < min_dist_sq) { min_dist_sq = dist_sq; best_ks = ks; }
                    }
                    pq_assignments[i] = best_ks;
                }
                std::vector<float> new_sub_codebook(PQ_KS * PQ_DS, 0.0f);
                std::vector<int> counts(PQ_KS, 0);
                for(size_t i=0; i<num_vectors_in_cluster; ++i) {
                    int ks_id = pq_assignments[i];
                    counts[ks_id]++;
                    for(int d=0; d<PQ_DS; ++d) new_sub_codebook[ks_id * PQ_DS + d] += sub_residuals[i * PQ_DS + d];
                }
                for(int ks=0; ks<PQ_KS; ++ks) {
                    if(counts[ks] > 0) {
                        for(int d=0; d<PQ_DS; ++d) new_sub_codebook[ks * PQ_DS + d] /= counts[ks];
                    }
                }
                memcpy(sub_codebook, new_sub_codebook.data(), PQ_KS * PQ_DS * sizeof(float));
            }
        }
    }
    std::cout << "PQ_Build: PQ codebook training finished.\n";
}

void IvfPqGpuV2::encode_database_residuals(const float* base, const std::vector<int>& assignments) {
    std::cout << "PQ_Build: Starting database encoding...\n";
    std::vector<std::vector<int>> inverted_lists(num_clusters_);
    for(size_t i=0; i<N_; ++i) inverted_lists[assignments[i]].push_back(i);
    h_reordered_base_.resize((long long)N_ * D_);
    h_original_ids_.resize(N_);
    h_pq_codes_.resize((long long)N_ * PQ_M);
    h_list_offsets_.resize(num_clusters_);
    h_list_sizes_.resize(num_clusters_);
    int current_offset = 0;
    for(int c=0; c<num_clusters_; ++c) {
        h_list_offsets_[c] = current_offset;
        h_list_sizes_[c] = inverted_lists[c].size();
        const float* centroid_vec = h_centroids_.data() + (long long)c * D_;
        const float* pq_codebook_for_cluster = h_pq_codebooks_.data() + (long long)c * PQ_M * PQ_KS * PQ_DS;
        for(int original_id : inverted_lists[c]) {
            const float* base_vec = base + (long long)original_id * D_;
            memcpy(h_reordered_base_.data() + (long long)current_offset * D_, base_vec, D_ * sizeof(float));
            h_original_ids_[current_offset] = original_id;
            std::vector<float> residual(D_);
            for (int d = 0; d < D_; ++d) residual[d] = base_vec[d] - centroid_vec[d];
            for (int m = 0; m < PQ_M; ++m) {
                const float* sub_residual = residual.data() + m * PQ_DS;
                const float* sub_codebook = pq_codebook_for_cluster + (long long)m * PQ_KS * PQ_DS;
                float min_dist_sq = 1e9f; int best_ks = -1;
                for(int ks=0; ks<PQ_KS; ++ks) {
                    float dist_sq = euclidean_dist_sq_v2(sub_residual, sub_codebook + ks * PQ_DS, PQ_DS);
                    if (dist_sq < min_dist_sq) { min_dist_sq = dist_sq; best_ks = ks; }
                }
                h_pq_codes_[(long long)current_offset * PQ_M + m] = (unsigned char)best_ks;
            }
            current_offset++;
        }
    }
    std::cout << "PQ_Build: Database encoding finished.\n";
}
void IvfPqGpuV2::Build(const float* base, size_t N) {
    N_ = N;
    std::vector<int> assignments(N);
    run_ivf_kmeans(base, assignments);
    train_pq_codebooks_on_residuals(base, assignments);
    encode_database_residuals(base, assignments);
}

void IvfPqGpuV2::SaveIndex(const std::string& path_prefix) {
    std::cout << "IVF-PQ: Saving index to files with prefix: " << path_prefix << std::endl;
    std::ofstream centroids_out(path_prefix + ".centroids.bin", std::ios::binary);
    centroids_out.write((char*)h_centroids_.data(), h_centroids_.size() * sizeof(float));
    std::ofstream reordered_out(path_prefix + ".reordered_base.bin", std::ios::binary);
    reordered_out.write((char*)h_reordered_base_.data(), h_reordered_base_.size() * sizeof(float));
    std::ofstream orig_ids_out(path_prefix + ".original_ids.bin", std::ios::binary);
    orig_ids_out.write((char*)h_original_ids_.data(), h_original_ids_.size() * sizeof(int));
    std::ofstream meta_out(path_prefix + ".meta.bin", std::ios::binary);
    meta_out.write((char*)h_list_offsets_.data(), h_list_offsets_.size() * sizeof(int));
    meta_out.write((char*)h_list_sizes_.data(), h_list_sizes_.size() * sizeof(int));
    std::ofstream pq_codes_out(path_prefix + ".pq_codes.bin", std::ios::binary);
    pq_codes_out.write((char*)h_pq_codes_.data(), h_pq_codes_.size() * sizeof(unsigned char));
    std::ofstream pq_codebooks_out(path_prefix + ".pq_codebooks.bin", std::ios::binary);
    pq_codebooks_out.write((char*)h_pq_codebooks_.data(), h_pq_codebooks_.size() * sizeof(float));
    std::cout << "IVF-PQ: Index saved successfully." << std::endl;
}

bool IvfPqGpuV2::LoadIndex(const std::string& path_prefix, size_t N) {
    N_ = N;
    std::cout << "IVF-PQ: Attempting to load index from files with prefix: " << path_prefix << std::endl;
    
    std::ifstream centroids_in(path_prefix + ".centroids.bin", std::ios::binary);
    if (!centroids_in) { std::cerr << "IVF-PQ: Cannot open centroids file." << std::endl; return false; }
    std::ifstream reordered_in(path_prefix + ".reordered_base.bin", std::ios::binary);
    if (!reordered_in) { std::cerr << "IVF-PQ: Cannot open reordered_base file." << std::endl; return false; }
    std::ifstream orig_ids_in(path_prefix + ".original_ids.bin", std::ios::binary);
    if (!orig_ids_in) { std::cerr << "IVF-PQ: Cannot open original_ids file." << std::endl; return false; }
    std::ifstream meta_in(path_prefix + ".meta.bin", std::ios::binary);
    if (!meta_in) { std::cerr << "IVF-PQ: Cannot open meta file." << std::endl; return false; }
    std::ifstream pq_codes_in(path_prefix + ".pq_codes.bin", std::ios::binary);
    if (!pq_codes_in) { std::cerr << "IVF-PQ: Cannot open pq_codes file." << std::endl; return false; }
    std::ifstream pq_codebooks_in(path_prefix + ".pq_codebooks.bin", std::ios::binary);
    if (!pq_codebooks_in) { std::cerr << "IVF-PQ: Cannot open pq_codebooks file." << std::endl; return false; }

    h_centroids_.resize((long long)num_clusters_ * D_);
    centroids_in.read((char*)h_centroids_.data(), h_centroids_.size() * sizeof(float));
    h_reordered_base_.resize((long long)N_ * D_);
    reordered_in.read((char*)h_reordered_base_.data(), h_reordered_base_.size() * sizeof(float));
    h_original_ids_.resize(N_);
    orig_ids_in.read((char*)h_original_ids_.data(), h_original_ids_.size() * sizeof(int));
    h_list_offsets_.resize(num_clusters_);
    meta_in.read((char*)h_list_offsets_.data(), h_list_offsets_.size() * sizeof(int));
    h_list_sizes_.resize(num_clusters_);
    meta_in.read((char*)h_list_sizes_.data(), h_list_sizes_.size() * sizeof(int));
    h_pq_codes_.resize((long long)N_ * PQ_M);
    pq_codes_in.read((char*)h_pq_codes_.data(), h_pq_codes_.size() * sizeof(unsigned char));
    h_pq_codebooks_.resize((long long)num_clusters_ * PQ_M * PQ_KS * PQ_DS);
    pq_codebooks_in.read((char*)h_pq_codebooks_.data(), h_pq_codebooks_.size() * sizeof(float));
    
    std::cout << "IVF-PQ: Index loaded from files successfully." << std::endl;
    return true;
}

void IvfPqGpuV2::TransferToDevice() {
    std::cout << "IVF-PQ: Transferring index to device..." << std::endl;
    // [修正3] 使用正确的宏 CHECK_CUDA_PQ
    CHECK_CUDA_PQ_V2(cudaMalloc(&d_centroids_, h_centroids_.size() * sizeof(float)));
    CHECK_CUDA_PQ_V2(cudaMemcpy(d_centroids_, h_centroids_.data(), h_centroids_.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_PQ_V2(cudaMalloc(&d_reordered_base_, h_reordered_base_.size() * sizeof(float)));
    CHECK_CUDA_PQ_V2(cudaMemcpy(d_reordered_base_, h_reordered_base_.data(), h_reordered_base_.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_PQ_V2(cudaMalloc(&d_original_ids_, h_original_ids_.size() * sizeof(int)));
    CHECK_CUDA_PQ_V2(cudaMemcpy(d_original_ids_, h_original_ids_.data(), h_original_ids_.size() * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA_PQ_V2(cudaMalloc(&d_list_offsets_, h_list_offsets_.size() * sizeof(int)));
    CHECK_CUDA_PQ_V2(cudaMemcpy(d_list_offsets_, h_list_offsets_.data(), h_list_offsets_.size() * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA_PQ_V2(cudaMalloc(&d_list_sizes_, h_list_sizes_.size() * sizeof(int)));
    CHECK_CUDA_PQ_V2(cudaMemcpy(d_list_sizes_, h_list_sizes_.data(), h_list_sizes_.size() * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA_PQ_V2(cudaMalloc(&d_pq_codes_, h_pq_codes_.size() * sizeof(unsigned char)));
    CHECK_CUDA_PQ_V2(cudaMemcpy(d_pq_codes_, h_pq_codes_.data(), h_pq_codes_.size() * sizeof(unsigned char), cudaMemcpyHostToDevice));
    CHECK_CUDA_PQ_V2(cudaMalloc(&d_pq_codebooks_, h_pq_codebooks_.size() * sizeof(float)));
    CHECK_CUDA_PQ_V2(cudaMemcpy(d_pq_codebooks_, h_pq_codebooks_.data(), h_pq_codebooks_.size() * sizeof(float), cudaMemcpyHostToDevice));
    std::cout << "IVF-PQ: Index data transferred to GPU.\n";
}

#endif // IVF_PQ_GPU_V2_H