#ifndef IVF_GPU_V2_HYBRID_FIXED_H
#define IVF_GPU_V2_HYBRID_FIXED_H

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
#include <iomanip>
#include <numeric>
#include <tuple>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cublas_v2.h"
#include "ivf_pq_common.cuh"
#include "ivf_pq_kernels.cuh"

#define CHECK_CUDA_IVF_V2(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "IVF_V2 CUDA Error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

#define CHECK_CUBLAS_IVF_V2(call) do { \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "IVF_V2 cuBLAS Error at %s:%d (code %d)\n", __FILE__, __LINE__, status); \
        exit(EXIT_FAILURE); \
    } \
} while (0)
struct SearchTask {
    __host__ __device__
    SearchTask() : query_idx(0), cluster_offset(0), cluster_size(0) {}

    int query_idx;
    int cluster_offset;
    int cluster_size;
};

// =================================================================
//                 IVF V2 - 优化后的内核套件
// =================================================================

// 【通用内核】: 初始化结果池
__global__ void initialize_results_kernel(ResultPair* d_results, size_t total_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_size) return;
    d_results[idx] = {1e9f, -1};
}

// 【通用设备函数】: 原子地更新一个全局内存中的结果池
__device__ void update_result_pool_atomic(ResultPair* d_pool, int pool_size, int point_id, float dist) {
    // 在循环中寻找和替换，保证原子性和正确性
    while (true) {
        // 1. (在循环内) 找到当前池中最差的距离和位置
        float worst_dist_float = dist; // 初始基准设为我们自己的距离
        int worst_idx = -1;

        for (int i = 0; i < pool_size; i++) {
            if (d_pool[i].distance > worst_dist_float) {
                worst_dist_float = d_pool[i].distance;
                worst_idx = i;
            }
        }

        // 2. 如果没有找到比我们更差的，说明我们的点不够好，直接退出
        if (worst_idx == -1) {
            return;
        }

        // 3. 尝试原子地替换找到的最差元素
        unsigned int worst_dist_as_int = __float_as_int(worst_dist_float);
        unsigned int dist_as_int = __float_as_int(dist);

        unsigned int old_val_as_int = atomicCAS(
            (unsigned int*)&d_pool[worst_idx].distance,
            worst_dist_as_int,
            dist_as_int
        );

        // 4. 如果替换成功，更新ID并退出
        if (old_val_as_int == worst_dist_as_int) {
            d_pool[worst_idx].id = point_id;
            return;
        }
        
    }
}

// 【S3辅助设备函数】: 在共享内存中更新Top-K池 (非原子，因为在块内同步)
__device__ void update_smem_pool(ResultPair* smem_pool, int pool_size, int point_id, float dist) {
    // 1. 找到当前池中最差的距离和位置
    float worst_dist_float = -1.0f;
    int worst_idx = -1;
    for (int i = 0; i < pool_size; i++) {
        // 直接读取即可，后续的CAS会保证原子性
        if (smem_pool[i].distance > worst_dist_float) {
            worst_dist_float = smem_pool[i].distance;
            worst_idx = i;
        }
    }

    // 2. 如果我们的新距离比池中最差的还要差，或者池子还没满（worst_dist是1e9f），就尝试更新
    while (dist < worst_dist_float) {
        unsigned int worst_dist_as_int = __float_as_int(worst_dist_float);
        unsigned int dist_as_int = __float_as_int(dist);

        unsigned int old_val_as_int = atomicCAS(
            (unsigned int*)&smem_pool[worst_idx].distance,
            worst_dist_as_int,
            dist_as_int
        );

        if (old_val_as_int == worst_dist_as_int) {
            smem_pool[worst_idx].id = point_id;
            return;
        }
        
        // 如果失败，用返回的新值更新我们的基准，再次尝试
        worst_dist_float = __int_as_float(old_val_as_int);
    }
}

__global__ void final_search_and_update_kernel(
    const float* d_queries,
    const int* d_sub_query_indices,
    const float* d_cluster_points,
    const int* d_cluster_point_ids,
    int cluster_size,
    int D,
    int k,
    ResultPair* d_final_results)
{
    // --- 1 & 2: Setup 和加载查询向量 ---
    int sub_query_idx = blockIdx.x;
    int real_query_idx = d_sub_query_indices[sub_query_idx];
    
    const int lane_id = threadIdx.x % 32;
    const int warp_id = threadIdx.x / 32;
    const int num_warps = blockDim.x / 32;

    extern __shared__ float smem[];
    float* smem_query = smem;

    const float* query_vec_global = d_queries + (long long)real_query_idx * D;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        smem_query[i] = query_vec_global[i];
    }
    __syncthreads();


    // --- 3: Warp-per-Point 距离计算 & Warp-level 找 Top-C 最优 ---
    
    // 每个 warp 在寄存器中维护一个大小为 C 的 Top-C 池
    const int C = 4; // 每个 warp 找 4 个最好的。这个值可以调整, 比如 2 或 4。
    ResultPair warp_top_c[C];
    // 初始化这个小池子
    for (int i = 0; i < C; ++i) {
        warp_top_c[i] = {1e9f, -1};
    }

    for (int point_offset = warp_id; point_offset < cluster_size; point_offset += num_warps) {
        const float* point_vec_global = d_cluster_points + (long long)point_offset * D;
        float partial_dist = 0.0f;
        for (int d = lane_id; d < D; d += 32) {
            float diff = smem_query[d] - point_vec_global[d];
            partial_dist += diff * diff;
        }
        for (int offset = 16; offset > 0; offset /= 2) {
            partial_dist += __shfl_down_sync(0xFFFFFFFF, partial_dist, offset);
        }

        // Warp的第一个线程 (lane 0) 负责将结果插入到本Warp的Top-C池
        if (lane_id == 0) {
            float dist = partial_dist;
            int id = d_cluster_point_ids[point_offset];

            // 检查新点是否能进入Top-C池 (比池中最差的要好)
            if (dist < warp_top_c[C-1].distance) {
                // 替换最差的点
                warp_top_c[C-1] = {dist, id};
                // 使用简单的冒泡排序，维持池的有序状态 (对于C=4非常快)
                for (int j = C - 1; j > 0; --j) {
                    if (warp_top_c[j].distance < warp_top_c[j-1].distance) {
                        // 交换
                        ResultPair temp = warp_top_c[j];
                        warp_top_c[j] = warp_top_c[j-1];
                        warp_top_c[j-1] = temp;
                    } else {
                        // 如果已经有序，则提前退出
                        break;
                    }
                }
            }
        }
    }

    // --- 4. 将每个Warp的Top-C候选者原子地更新到全局最终结果池 ---
    // 每个 Warp 的 lane 0 线程负责写入本 warp 找到的 C 个候选者
    if (lane_id == 0) {
        ResultPair* global_pool_for_query = d_final_results + (long long)real_query_idx * k;
        for (int i = 0; i < C; ++i) {
            if (warp_top_c[i].id != -1) {
                update_result_pool_atomic(global_pool_for_query, k, warp_top_c[i].id, warp_top_c[i].distance);
            }
        }
    }
}

__global__ void cluster_search_kernel_final(
    const float* d_queries,
    const int* d_sub_query_indices,
    const float* d_cluster_points,
    const int* d_cluster_point_ids,
    int cluster_size,
    int D,
    int rerank_size,
    ResultPair* d_intermediate_results)
{
    // --- 1. Setup ---
    int sub_query_idx = blockIdx.x;
    int real_query_idx = d_sub_query_indices[sub_query_idx];
    
    // Warp-level and Block-level IDs
    const int lane_id = threadIdx.x % 32;
    const int warp_id = threadIdx.x / 32;
    const int num_warps = blockDim.x / 32;

    // --- 2. 共享内存布局和加载查询向量 ---
    // 布局: [ smem_query(D) | smem_candidates(num_warps) ]
    extern __shared__ float smem[];
    float* smem_query = smem;
    ResultPair* smem_candidates = (ResultPair*)(smem_query + D);

    // 协作加载查询向量到共享内存
    const float* query_vec_global = d_queries + (long long)real_query_idx * D;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        smem_query[i] = query_vec_global[i];
    }
    __syncthreads();

    // --- 3. Warp-per-Point 距离计算 & Warp-level 找最优 ---
    ResultPair warp_best = {1e9f, -1};

    // 每个 Warp 负责处理一部分点
    for (int point_offset = warp_id; point_offset < cluster_size; point_offset += num_warps) {
        const float* point_vec_global = d_cluster_points + (long long)point_offset * D;
        
        // (Warp内协作) 并行计算一个距离
        float partial_dist = 0.0f;
        for (int d = lane_id; d < D; d += 32) { // 32 is warpSize
            float diff = smem_query[d] - point_vec_global[d];
            partial_dist += diff * diff;
        }

        // (Warp内规约) 使用 shuffle 指令高效求和
        for (int offset = 16; offset > 0; offset /= 2) {
            partial_dist += __shfl_down_sync(0xFFFFFFFF, partial_dist, offset);
        }

        // Warp的第一个线程 (lane 0) 得到最终距离，并更新本Warp的最优值
        if (lane_id == 0) {
            if (partial_dist < warp_best.distance) {
                warp_best.distance = partial_dist;
                warp_best.id = d_cluster_point_ids[point_offset];
            }
        }
    }

    // --- 4. Block-Level Reduction & Sort ---
    // 每个Warp的lane 0线程，将本Warp的最优结果写入共享内存
    if (lane_id == 0) {
        smem_candidates[warp_id] = warp_best;
    }
    __syncthreads(); // 确保所有Warp都已写入

    // 只让线程 0 进行排序
    if (threadIdx.x == 0) {
        int num_candidates = num_warps;

        int sort_limit = min(rerank_size, num_candidates);

        for (int i = 0; i < sort_limit; ++i) {
            int min_idx = i;
            for (int j = i + 1; j < num_candidates; ++j) {
                if(smem_candidates[j].distance < smem_candidates[min_idx].distance) {
                    min_idx = j;
                }
            }
            ResultPair temp = smem_candidates[i];
            smem_candidates[i] = smem_candidates[min_idx];
            smem_candidates[min_idx] = temp;
        }
    }
    __syncthreads();

    // --- 5. Write Back to Global Intermediate Pool ---
    ResultPair* global_pool_for_query = d_intermediate_results + (long long)real_query_idx * rerank_size;
    int num_to_write = min(rerank_size, num_warps);
    
    // 所有线程协作写回
    for (int i = threadIdx.x; i < num_to_write; i += blockDim.x) {
        ResultPair res = smem_candidates[i];
        if (res.id != -1) {
            update_result_pool_atomic(global_pool_for_query, rerank_size, res.id, res.distance);
        }
    }
}


// 【S4内核】: 最终池排序，与原版基本一致
__global__ void final_pool_sort_kernel(
    const ResultPair* d_intermediate_results,
    int M, int k, int rerank_size,
    ResultPair* d_final_results) 
{
    int query_idx = blockIdx.x;
    if (query_idx >= M) return;

    extern __shared__ ResultPair smem_pool[];
    const ResultPair* my_pool_start = d_intermediate_results + (long long)query_idx * rerank_size;
    
    // 1. 协作加载到共享内存
    for (int i = threadIdx.x; i < rerank_size; i += blockDim.x) {
        smem_pool[i] = my_pool_start[i];
    }
    __syncthreads();

    // 2. 线程0进行排序
    if (threadIdx.x == 0) {
        // 移除无效条目
        int num_valid = 0;
        for(int i = 0; i < rerank_size; ++i) {
            if(smem_pool[i].id != -1) {
                smem_pool[num_valid++] = smem_pool[i];
            }
        }
        
        // 简单选择排序
        int sort_limit = min(k, num_valid);
        for (int i = 0; i < sort_limit; ++i) {
            int min_idx = i;
            for (int j = i + 1; j < num_valid; ++j) {
                if(smem_pool[j].distance < smem_pool[min_idx].distance) min_idx = j;
            }
            ResultPair temp = smem_pool[i]; 
            smem_pool[i] = smem_pool[min_idx]; 
            smem_pool[min_idx] = temp;
        }

        // 写回最终结果
        for (int i = 0; i < k; ++i) {
            d_final_results[query_idx * k + i] = (i < sort_limit) ? smem_pool[i] : ResultPair{1e9f, -1};
        }
    }
}
// =================================================================
//                   IVF V2 - 主机端封装类
// =================================================================
class IvfGpuV2 {
public:
    IvfGpuV2(size_t dim, int num_clusters);
    ~IvfGpuV2();
    void Build(const float* base, size_t N);
    bool LoadIndex(const std::string& path_prefix, size_t N);
    void SaveIndex(const std::string& path_prefix);
    void TransferToDevice();
    void search_batch(
        const float* queries, size_t M, int k, int nprobe,
        std::vector<std::priority_queue<std::pair<float, int>>>& final_results,
        std::vector<float>& timings_ms);

private:
    void run_ivf_kmeans(const float* base, std::vector<int>& assignments);
    void reorder_base_vectors(const float* base, const std::vector<int>& assignments);
    void compute_dist_matrix_for_probes(const float* d_queries, size_t M, float* d_dist_matrix);
    
    size_t D_, N_;
    int num_clusters_;
    cublasHandle_t cublas_handle_;
    std::vector<float> h_centroids_, h_reordered_base_;
    std::vector<int> h_list_offsets_, h_list_sizes_, h_original_ids_;
    float* d_centroids_ = nullptr;
    float* d_reordered_base_ = nullptr;
    int* d_original_ids_ = nullptr;
};


IvfGpuV2::IvfGpuV2(size_t dim, int num_clusters) : D_(dim), N_(0), num_clusters_(num_clusters) {
    CHECK_CUBLAS_IVF_V2(cublasCreate(&cublas_handle_));
}

IvfGpuV2::~IvfGpuV2() {
    CHECK_CUBLAS_IVF_V2(cublasDestroy(cublas_handle_));
    if (d_centroids_) CHECK_CUDA_IVF_V2(cudaFree(d_centroids_));
    if (d_reordered_base_) CHECK_CUDA_IVF_V2(cudaFree(d_reordered_base_));
    if (d_original_ids_) CHECK_CUDA_IVF_V2(cudaFree(d_original_ids_));
}

void IvfGpuV2::Build(const float* base, size_t N) {
    N_ = N;
    std::vector<int> assignments(N);
    run_ivf_kmeans(base, assignments);
    reorder_base_vectors(base, assignments);
}

void IvfGpuV2::SaveIndex(const std::string& path_prefix) {
    std::cout << "IVF_V2: Saving index to files with prefix: " << path_prefix << std::endl;
    std::ofstream centroids_out(path_prefix + ".centroids.bin", std::ios::binary);
    centroids_out.write((char*)h_centroids_.data(), h_centroids_.size() * sizeof(float));
    std::ofstream reordered_out(path_prefix + ".reordered_base.bin", std::ios::binary);
    reordered_out.write((char*)h_reordered_base_.data(), h_reordered_base_.size() * sizeof(float));
    std::ofstream orig_ids_out(path_prefix + ".original_ids.bin", std::ios::binary);
    orig_ids_out.write((char*)h_original_ids_.data(), h_original_ids_.size() * sizeof(int));
    std::ofstream meta_out(path_prefix + ".meta.bin", std::ios::binary);
    meta_out.write((char*)h_list_offsets_.data(), h_list_offsets_.size() * sizeof(int));
    meta_out.write((char*)h_list_sizes_.data(), h_list_sizes_.size() * sizeof(int));
    std::cout << "IVF_V2: Index saved successfully." << std::endl;
}

bool IvfGpuV2::LoadIndex(const std::string& path_prefix, size_t N) {
    N_ = N;
    std::cout << "IVF_V2: Attempting to load index from files with prefix: " << path_prefix << std::endl;
    std::ifstream centroids_in(path_prefix + ".centroids.bin", std::ios::binary);
    if (!centroids_in) { std::cerr << "Cannot open centroids file." << std::endl; return false; }
    std::ifstream reordered_in(path_prefix + ".reordered_base.bin", std::ios::binary);
    if (!reordered_in) { std::cerr << "Cannot open reordered_base file." << std::endl; return false; }
    std::ifstream orig_ids_in(path_prefix + ".original_ids.bin", std::ios::binary);
    if (!orig_ids_in) { std::cerr << "Cannot open original_ids file." << std::endl; return false; }
    std::ifstream meta_in(path_prefix + ".meta.bin", std::ios::binary);
    if (!meta_in) { std::cerr << "Cannot open meta file." << std::endl; return false; }

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
    std::cout << "IVF_V2: Index loaded from files successfully." << std::endl;
    return true;
}

void IvfGpuV2::TransferToDevice() {
    std::cout << "IVF_V2: Transferring index to device..." << std::endl;
    CHECK_CUDA_IVF_V2(cudaMalloc(&d_centroids_, h_centroids_.size() * sizeof(float)));
    CHECK_CUDA_IVF_V2(cudaMemcpy(d_centroids_, h_centroids_.data(), h_centroids_.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_IVF_V2(cudaMalloc(&d_reordered_base_, h_reordered_base_.size() * sizeof(float)));
    CHECK_CUDA_IVF_V2(cudaMemcpy(d_reordered_base_, h_reordered_base_.data(), h_reordered_base_.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_IVF_V2(cudaMalloc(&d_original_ids_, h_original_ids_.size() * sizeof(int)));
    CHECK_CUDA_IVF_V2(cudaMemcpy(d_original_ids_, h_original_ids_.data(), h_original_ids_.size() * sizeof(int), cudaMemcpyHostToDevice));
    std::cout << "IVF_V2: Index data transferred to GPU.\n";
}

float euclidean_dist_sq_v2(const float* v1, const float* v2, size_t D) {
    float dist_sq = 0.0f;
    for (size_t i = 0; i < D; ++i) {
        float diff = v1[i] - v2[i];
        dist_sq += diff * diff;
    }
    return dist_sq;
}

void IvfGpuV2::run_ivf_kmeans(const float* base, std::vector<int>& assignments) {
    std::cout << "IVF_V2_Build: Starting IVF K-Means (L2 distance)...\n";
    h_centroids_.resize((long long)num_clusters_ * D_);
    std::vector<int> random_indices(N_);
    std::iota(random_indices.begin(), random_indices.end(), 0);
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(random_indices.begin(), random_indices.end(), g);
    for(int i=0; i<num_clusters_; ++i) {
        memcpy(h_centroids_.data() + (long long)i * D_, base + (long long)random_indices[i] * D_, D_ * sizeof(float));
    }
    int max_iters = 55;
    for (int iter = 0; iter < max_iters; ++iter) {
        #pragma omp parallel for
        for (long long i = 0; i < N_; ++i) {
            float min_dist = 1e9f; int best_cluster = -1;
            for (int c = 0; c < num_clusters_; ++c) {
                float dist = euclidean_dist_sq_v2(base + i * D_, h_centroids_.data() + c * D_, D_);
                if (dist < min_dist) { min_dist = dist; best_cluster = c; }
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
        std::cout << "K-Means iter " << iter << " done." << std::endl;
    }
}

void IvfGpuV2::reorder_base_vectors(const float* base, const std::vector<int>& assignments) {
    std::vector<std::vector<int>> inverted_lists(num_clusters_);
    for(size_t i=0; i<N_; ++i) inverted_lists[assignments[i]].push_back(i);
    h_reordered_base_.resize((long long)N_ * D_);
    h_original_ids_.resize(N_);
    h_list_offsets_.resize(num_clusters_);
    h_list_sizes_.resize(num_clusters_);
    int current_offset = 0;
    for(int c=0; c<num_clusters_; ++c) {
        h_list_offsets_[c] = current_offset;
        h_list_sizes_[c] = inverted_lists[c].size();
        for(int original_id : inverted_lists[c]) {
            memcpy(h_reordered_base_.data() + (long long)current_offset * D_, base + (long long)original_id * D_, D_ * sizeof(float));
            h_original_ids_[current_offset] = original_id;
            current_offset++;
        }
    }
}


void IvfGpuV2::compute_dist_matrix_for_probes(const float* d_queries, size_t M, float* d_dist_matrix) {
    const float alpha = -2.0f;
    const float beta = 0.0f;


    CHECK_CUBLAS_IVF_V2(cublasSgemm(cublas_handle_, 
                                  CUBLAS_OP_T,       
                                  CUBLAS_OP_N,       
                                  num_clusters_, M, D_, 
                                  &alpha, 
                                  d_centroids_, D_,  // B matrix
                                  d_queries, D_,     // A matrix
                                  &beta, 
                                  d_dist_matrix, num_clusters_)); // C matrix


    float* d_q_norm; CHECK_CUDA_IVF_V2(cudaMalloc(&d_q_norm, M * sizeof(float)));
    float* d_c_norm; CHECK_CUDA_IVF_V2(cudaMalloc(&d_c_norm, num_clusters_ * sizeof(float)));
    
    l2_norm_squared_kernel<<< (M + 255) / 256, 256 >>>(d_queries, d_q_norm, M, D_);
    l2_norm_squared_kernel<<< (num_clusters_ + 255) / 256, 256 >>>(d_centroids_, d_c_norm, num_clusters_, D_);
    
    dim3 grid_b( (num_clusters_ + 15) / 16, (M + 15) / 16 );
    dim3 block_b(16, 16);

    add_broadcast_col_kernel<<<grid_b, block_b>>>(d_dist_matrix, d_q_norm, M, num_clusters_);
    
    add_broadcast_row_kernel<<<grid_b, block_b>>>(d_dist_matrix, d_c_norm, M, num_clusters_);

    CHECK_CUDA_IVF_V2(cudaFree(d_q_norm));
    CHECK_CUDA_IVF_V2(cudaFree(d_c_norm));
}

// =================================================================
//        search_batch 的最终优化版 (Streams + 新内核)
// =================================================================
void IvfGpuV2::search_batch(
    const float* queries, size_t M, int k, int nprobe,
    // rerank_size 参数已被移除
    std::vector<std::priority_queue<std::pair<float, int>>>& final_results,
    std::vector<float>& timings_ms)
{
    // --- 0. 事件和计时器设置 ---
    cudaEvent_t start_total, stop_total, start_stage, stop_stage;
    CHECK_CUDA_IVF_V2(cudaEventCreate(&start_total));
    CHECK_CUDA_IVF_V2(cudaEventCreate(&stop_total));
    CHECK_CUDA_IVF_V2(cudaEventCreate(&start_stage));
    CHECK_CUDA_IVF_V2(cudaEventCreate(&stop_stage));
    timings_ms.clear();
    float stage_time;
    
    CHECK_CUDA_IVF_V2(cudaEventRecord(start_total));

    // --- H2D Transfer ---
    CHECK_CUDA_IVF_V2(cudaEventRecord(start_stage));
    float* d_queries;
    CHECK_CUDA_IVF_V2(cudaMalloc(&d_queries, (long long)M * D_ * sizeof(float)));
    CHECK_CUDA_IVF_V2(cudaMemcpy(d_queries, queries, (long long)M * D_ * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_IVF_V2(cudaEventRecord(stop_stage));
    CHECK_CUDA_IVF_V2(cudaEventSynchronize(stop_stage));
    CHECK_CUDA_IVF_V2(cudaEventElapsedTime(&stage_time, start_stage, stop_stage));
    timings_ms.push_back(stage_time); // [0] H2D

    // --- S1: Probe Selection (GPU) ---
    CHECK_CUDA_IVF_V2(cudaEventRecord(start_stage));
    float* d_dist_matrix;
    CHECK_CUDA_IVF_V2(cudaMalloc(&d_dist_matrix, (long long)M * num_clusters_ * sizeof(float)));
    compute_dist_matrix_for_probes(d_queries, M, d_dist_matrix);
    
    int* d_top_probes;
    CHECK_CUDA_IVF_V2(cudaMalloc(&d_top_probes, (long long)M * nprobe * sizeof(int)));
    find_top_n_probes_dist_kernel<<<M, 256, 256 * nprobe * sizeof(ResultPair)>>>(d_dist_matrix, M, num_clusters_, nprobe, d_top_probes);
    CHECK_CUDA_IVF_V2(cudaGetLastError());
    
    std::vector<int> h_top_probes(M * nprobe);
    CHECK_CUDA_IVF_V2(cudaMemcpy(h_top_probes.data(), d_top_probes, (long long)M * nprobe * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_IVF_V2(cudaEventRecord(stop_stage));
    CHECK_CUDA_IVF_V2(cudaEventSynchronize(stop_stage));
    CHECK_CUDA_IVF_V2(cudaEventElapsedTime(&stage_time, start_stage, stop_stage));
    timings_ms.push_back(stage_time); // [1] S1 Probe Selection

    // --- S2: Work Reorganization (CPU, Cluster-centric) ---
    CHECK_CUDA_IVF_V2(cudaEventRecord(start_stage));
    std::vector<std::vector<int>> cluster_to_queries_map(num_clusters_);
    for (size_t i = 0; i < M; ++i) {
        for (int j = 0; j < nprobe; ++j) {
            int cluster_id = h_top_probes[i * nprobe + j];
            if (cluster_id >= 0 && cluster_id < num_clusters_) {
                cluster_to_queries_map[cluster_id].push_back(i);
            }
        }
    }
    CHECK_CUDA_IVF_V2(cudaEventRecord(stop_stage));
    CHECK_CUDA_IVF_V2(cudaEventSynchronize(stop_stage));
    CHECK_CUDA_IVF_V2(cudaEventElapsedTime(&stage_time, start_stage, stop_stage));
    timings_ms.push_back(stage_time); // [2] S2 CPU Scheduling

    // --- S3: Streamed Search and Direct Update ---
    CHECK_CUDA_IVF_V2(cudaEventRecord(start_stage));
    
    // 不再需要 d_intermediate_results，直接分配 d_final_results
    ResultPair* d_final_results;
    CHECK_CUDA_IVF_V2(cudaMalloc(&d_final_results, (long long)M * k * sizeof(ResultPair)));
    // 初始化最终结果池
    initialize_results_kernel<<<( (long long)M * k + 255) / 256, 256>>>(d_final_results, (long long)M * k);
    
    const int num_streams = 8;
    std::vector<cudaStream_t> streams(num_streams);
    for (int i = 0; i < num_streams; ++i) {
        CHECK_CUDA_IVF_V2(cudaStreamCreate(&streams[i]));
    }
    std::vector<int*> temp_d_sub_query_indices_list;

    for (int c = 0; c < num_clusters_; ++c) {
        const auto& queries_for_this_cluster = cluster_to_queries_map[c];
        if (queries_for_this_cluster.empty() || h_list_sizes_[c] == 0) continue;
        
        cudaStream_t current_stream = streams[c % num_streams];
        
        int num_queries_for_cluster = queries_for_this_cluster.size();
        int* d_sub_query_indices;
        CHECK_CUDA_IVF_V2(cudaMalloc(&d_sub_query_indices, num_queries_for_cluster * sizeof(int)));
        temp_d_sub_query_indices_list.push_back(d_sub_query_indices);

        CHECK_CUDA_IVF_V2(cudaMemcpyAsync(d_sub_query_indices, queries_for_this_cluster.data(), num_queries_for_cluster * sizeof(int), cudaMemcpyHostToDevice, current_stream));

        dim3 grid(num_queries_for_cluster);
        dim3 block(256);
        
        // 共享内存大小与之前 v8/final 版本相同
        size_t smem_size = (D_ * sizeof(float)) + ((block.x / 32) * sizeof(ResultPair));

        final_search_and_update_kernel<<<grid, block, smem_size, current_stream>>>(
            d_queries,
            d_sub_query_indices,
            d_reordered_base_ + (long long)h_list_offsets_[c] * D_,
            d_original_ids_ + h_list_offsets_[c],
            h_list_sizes_[c],
            D_,
            k, 
            d_final_results 
        );
        CHECK_CUDA_IVF_V2(cudaGetLastError());
    }

    for (int i = 0; i < num_streams; ++i) {
        CHECK_CUDA_IVF_V2(cudaStreamSynchronize(streams[i]));
    }
    
    CHECK_CUDA_IVF_V2(cudaEventRecord(stop_stage));
    CHECK_CUDA_IVF_V2(cudaEventSynchronize(stop_stage));
    CHECK_CUDA_IVF_V2(cudaEventElapsedTime(&stage_time, start_stage, stop_stage));
    timings_ms.push_back(stage_time); // [3] S3 Clustered Search

    for (auto ptr : temp_d_sub_query_indices_list) {
        CHECK_CUDA_IVF_V2(cudaFree(ptr));
    }
    for (int i = 0; i < num_streams; ++i) {
        CHECK_CUDA_IVF_V2(cudaStreamDestroy(streams[i]));
    }

    // --- S4: Stage Removed ---
   
    timings_ms.push_back(0.0f); // [4] S4 (Removed)

    // --- D2H Transfer ---
    CHECK_CUDA_IVF_V2(cudaEventRecord(start_stage));
    std::vector<ResultPair> h_results((long long)M * k);
    // 直接从 d_final_results 拷贝，因为它已经被填充
    CHECK_CUDA_IVF_V2(cudaMemcpy(h_results.data(), d_final_results, (long long)M * k * sizeof(ResultPair), cudaMemcpyDeviceToHost));
    CHECK_CUDA_IVF_V2(cudaEventRecord(stop_stage));
    CHECK_CUDA_IVF_V2(cudaEventSynchronize(stop_stage));
    CHECK_CUDA_IVF_V2(cudaEventElapsedTime(&stage_time, start_stage, stop_stage));
    timings_ms.push_back(stage_time); // [5] D2H

    // --- CPU Post-processing ---
    final_results.assign(M, std::priority_queue<std::pair<float, int>>());
    for (size_t i = 0; i < M; ++i) {
        for (int j = 0; j < k; ++j) {
            ResultPair p = h_results[i * k + j];
            if(p.id != -1) {
                final_results[i].push({-sqrtf(p.distance), p.id});
            }
        }
    }
    
    // --- Cleanup ---
    CHECK_CUDA_IVF_V2(cudaFree(d_queries));
    CHECK_CUDA_IVF_V2(cudaFree(d_dist_matrix));
    CHECK_CUDA_IVF_V2(cudaFree(d_top_probes));
    // 不再需要 d_intermediate_results
    CHECK_CUDA_IVF_V2(cudaFree(d_final_results));
    
    CHECK_CUDA_IVF_V2(cudaEventDestroy(start_total));
    CHECK_CUDA_IVF_V2(cudaEventDestroy(stop_total));
    CHECK_CUDA_IVF_V2(cudaEventDestroy(start_stage));
    CHECK_CUDA_IVF_V2(cudaEventDestroy(stop_stage));
}
#endif // IVF_GPU_V2_HYBRID_FIXED_H