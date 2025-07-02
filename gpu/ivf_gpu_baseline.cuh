#ifndef IVF_GPU_BASELINE_H
#define IVF_GPU_BASELINE_H

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

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cublas_v2.h"

// --- 辅助宏 ---
#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

#define CHECK_CUBLAS(call) do { \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS Error at %s:%d (code %d)\n", __FILE__, __LINE__, status); \
        exit(EXIT_FAILURE); \
    } \
} while (0)


// --- GPU数据结构 ---
struct ResultPair {
    float inner_product;
    int id;
};


// --- GPU 核函数 ---

// S1阶段: 为每个查询找到内积最大的nprobe个簇

__global__ void find_top_n_probes_kernel_ip(
    const float* ip_matrix,
    int M,
    int num_clusters,
    int nprobe,
    int* top_probes_gpu)
{
    //  使用 1个Block处理1个Query的模式 
    int query_idx = blockIdx.x;
    if (query_idx >= M) return;

    extern __shared__ ResultPair smem_probes[];
    const float* query_ips = ip_matrix + (long long)query_idx * num_clusters;
    int* query_probes_out = top_probes_gpu + (long long)query_idx * nprobe;

    // 1. 每个线程负责一段，找到自己范围内的Top-nprobe
    ResultPair* local_top_probes = smem_probes + threadIdx.x * nprobe;
    for (int i = 0; i < nprobe; ++i) local_top_probes[i] = {-2.0f, -1};
    
    for (int i = threadIdx.x; i < num_clusters; i += blockDim.x) {
        float current_ip = query_ips[i];
        if (local_top_probes[nprobe - 1].id == -1 || current_ip > local_top_probes[nprobe - 1].inner_product) {
            local_top_probes[nprobe - 1] = {current_ip, i};
            for(int j = nprobe - 1; j > 0; --j){
                if(local_top_probes[j].inner_product > local_top_probes[j-1].inner_product){
                    ResultPair temp = local_top_probes[j];
                    local_top_probes[j] = local_top_probes[j-1];
                    local_top_probes[j-1] = temp;
                } else break;
            }
        }
    }
    __syncthreads();

    // 2. 块内归约：由线程0合并所有线程的Top-nprobe结果
    if (threadIdx.x == 0) {
        for(int i = 1; i < blockDim.x; ++i){
            ResultPair* other_probes = smem_probes + i * nprobe;
            for(int j = 0; j < nprobe; ++j){
                if (other_probes[j].id == -1) break;
                float current_ip = other_probes[j].inner_product;
                if(local_top_probes[nprobe-1].id == -1 || current_ip > local_top_probes[nprobe-1].inner_product){
                    local_top_probes[nprobe-1] = other_probes[j];
                    for(int l = nprobe - 1; l > 0; --l){
                        if(local_top_probes[l-1].id == -1 || local_top_probes[l].inner_product > local_top_probes[l-1].inner_product){
                            ResultPair temp = local_top_probes[l];
                            local_top_probes[l] = local_top_probes[l-1];
                            local_top_probes[l-1] = temp;
                        } else break;
                    }
                }
            }
        }
        // 将最终结果写回全局内存
        for (int i = 0; i < nprobe; ++i) {
            query_probes_out[i] = local_top_probes[i].id;
        }
    }
}
// S2阶段
__global__ void intra_cluster_search_kernel_ip(
    const float* queries,
    const float* reordered_base,
    const int* list_offsets,
    const int* list_sizes,
    const int* original_ids,
    const int* top_probes,
    int M, int D, int K, int nprobe,
    int num_clusters,
    size_t N,
    ResultPair* final_results_gpu)
{
    int query_idx = blockIdx.x;
    if (query_idx >= M) return;

    extern __shared__ ResultPair smem_candidates[];

    // 1. 每个线程找到自己的局部最优
    ResultPair local_best = { -2.0f, -1 };
    const float* query_vec = queries + (long long)query_idx * D;
    const int* probes_for_query = top_probes + (long long)query_idx * nprobe;

    for (int p_idx = 0; p_idx < nprobe; ++p_idx) {
        int cluster_id = probes_for_query[p_idx];
        if (cluster_id < 0 || cluster_id >= num_clusters) continue;
        
        int list_offset = list_offsets[cluster_id];
        int list_size = list_sizes[cluster_id];
        if (list_size <= 0) continue;

        for (int i = threadIdx.x; i < list_size; i += blockDim.x) {
            long long full_offset = (long long)list_offset + i;
            if (full_offset >= N) continue;

            const float* base_vec = reordered_base + full_offset * D;
            float ip = 0.0f;
            for (int d = 0; d < D; ++d) {
                ip += base_vec[d] * query_vec[d];
            }
            if (ip > local_best.inner_product) {
                local_best.inner_product = ip;
                local_best.id = original_ids[full_offset];
            }
        }
    }
    
    // 2. 将所有线程的局部最优结果写入共享内存
    smem_candidates[threadIdx.x] = local_best;
    __syncthreads();

    // 3. 块内串行选择Top-K (由线程0完成)
    if (threadIdx.x == 0) {
        // 使用一个简单的选择排序来找到最大的K个元素
        for (int i = 0; i < K; ++i) {
            int max_idx = i;
            for (int j = i + 1; j < blockDim.x; ++j) {
                if (smem_candidates[j].inner_product > smem_candidates[max_idx].inner_product) {
                    max_idx = j;
                }
            }
            ResultPair temp = smem_candidates[i];
            smem_candidates[i] = smem_candidates[max_idx];
            smem_candidates[max_idx] = temp;
        }
        
        // 将最终结果写入全局内存
        for (int i = 0; i < K; ++i) {
            final_results_gpu[(long long)query_idx * K + i] = smem_candidates[i];
        }
    }
}


// --- 主机端 K-Means 和数据重排 ---
void run_cpu_kmeans_and_reorder(
    const float* base, size_t N, size_t D, int num_clusters,
    std::vector<float>& centroids, 
    std::vector<float>& reordered_base,
    std::vector<int>& list_offsets,
    std::vector<int>& list_sizes,
    std::vector<int>& original_ids)
{
    std::cout << "Starting CPU K-Means...\n";
    
    centroids.resize((long long)num_clusters * D);
    std::vector<int> random_indices(N);
    for(size_t i=0; i<N; ++i) random_indices[i] = i;
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(random_indices.begin(), random_indices.end(), g);
    for(int i=0; i<num_clusters; ++i) {
        memcpy(centroids.data() + (long long)i * D, base + (long long)random_indices[i] * D, (long long)D * sizeof(float));
    }

    std::vector<int> assignments(N);
    int max_iters = 55;
    for (int iter = 0; iter < max_iters; ++iter) {
        std::cout << "K-Means iteration " << iter + 1 << "/" << max_iters << "\n";
        
        #pragma omp parallel for
        for (long long i = 0; i < N; ++i) {
            float max_ip = -1e9;
            int best_cluster = -1;
            for (int c = 0; c < num_clusters; ++c) {
                float current_ip = 0.0f;
                for (size_t d = 0; d < D; ++d) {
                    current_ip += base[i * D + d] * centroids[c * D + d];
                }
                if (current_ip > max_ip) {
                    max_ip = current_ip;
                    best_cluster = c;
                }
            }
            assignments[i] = best_cluster;
        }

        centroids.assign((long long)num_clusters * D, 0.0f);
        std::vector<int> cluster_counts(num_clusters, 0);
        for (size_t i = 0; i < N; ++i) {
            int cluster_id = assignments[i];
            for (size_t d = 0; d < D; ++d) {
                centroids[(long long)cluster_id * D + d] += base[(long long)i * D + d];
            }
            cluster_counts[cluster_id]++;
        }

        #pragma omp parallel for
        for (int c = 0; c < num_clusters; ++c) {
            if (cluster_counts[c] > 0) {
                float norm = 0.0f;
                for (size_t d = 0; d < D; ++d) {
                    centroids[(long long)c * D + d] /= cluster_counts[c];
                    norm += centroids[(long long)c * D + d] * centroids[(long long)c * D + d];
                }
                norm = std::sqrt(norm);
                if (norm > 0) {
                    for(size_t d=0; d<D; ++d) centroids[(long long)c * D + d] /= norm;
                }
            }
        }
    }
    std::cout << "K-Means finished.\n";

    std::cout << "Starting data reordering...\n";
    std::vector<std::vector<int>> inverted_lists(num_clusters);
    for(size_t i=0; i<N; ++i) {
        inverted_lists[assignments[i]].push_back(i);
    }
    
    reordered_base.resize((long long)N*D);
    original_ids.resize(N);
    list_offsets.resize(num_clusters);
    list_sizes.resize(num_clusters);

    int current_offset = 0;
    for(int c=0; c<num_clusters; ++c) {
        list_offsets[c] = current_offset;
        list_sizes[c] = inverted_lists[c].size();
        for(int original_id : inverted_lists[c]) {
            memcpy(reordered_base.data() + (long long)current_offset * D, base + (long long)original_id * D, (long long)D * sizeof(float));
            original_ids[current_offset] = original_id;
            current_offset++;
        }
    }
    std::cout << "Data reordering finished.\n";
}


// --- 主机端封装类 ---
class IvfGpuBaseline {
public:
    IvfGpuBaseline(size_t dim, int num_clusters) : D_(dim), N_(0), num_clusters_(num_clusters) {
        CHECK_CUBLAS(cublasCreate(&cublas_handle_));
    }

    ~IvfGpuBaseline() {
        CHECK_CUBLAS(cublasDestroy(cublas_handle_));
        if (d_centroids_) CHECK_CUDA(cudaFree(d_centroids_));
        if (d_reordered_base_) CHECK_CUDA(cudaFree(d_reordered_base_));
        if (d_list_offsets_) CHECK_CUDA(cudaFree(d_list_offsets_));
        if (d_list_sizes_) CHECK_CUDA(cudaFree(d_list_sizes_));
        if (d_original_ids_) CHECK_CUDA(cudaFree(d_original_ids_));
    }

    void SaveIndex(const std::string& path_prefix) {
        std::cout << "Saving index to files with prefix: " << path_prefix << std::endl;
        std::ofstream centroids_out(path_prefix + ".centroids.bin", std::ios::binary);
        centroids_out.write((char*)h_centroids_.data(), h_centroids_.size() * sizeof(float));
        std::ofstream reordered_out(path_prefix + ".reordered_base.bin", std::ios::binary);
        reordered_out.write((char*)h_reordered_base_.data(), h_reordered_base_.size() * sizeof(float));
        std::ofstream orig_ids_out(path_prefix + ".original_ids.bin", std::ios::binary);
        orig_ids_out.write((char*)h_original_ids_.data(), h_original_ids_.size() * sizeof(int));
        std::ofstream meta_out(path_prefix + ".meta.bin", std::ios::binary);
        meta_out.write((char*)h_list_offsets_.data(), h_list_offsets_.size() * sizeof(int));
        meta_out.write((char*)h_list_sizes_.data(), h_list_sizes_.size() * sizeof(int));
        std::cout << "Index saved successfully." << std::endl;
    }

    bool LoadIndex(const std::string& path_prefix, size_t N) {
        N_ = N;
        std::cout << "Attempting to load index from files with prefix: " << path_prefix << std::endl;
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
        
        std::cout << "Index loaded from files successfully." << std::endl;
        return true;
    }

    void Build(const float* base, size_t N) {
        N_ = N;
        run_cpu_kmeans_and_reorder(base, N, D_, num_clusters_, h_centroids_, 
                                   h_reordered_base_, h_list_offsets_, h_list_sizes_, h_original_ids_);
    }

    void TransferToDevice() {
        std::cout << "Transferring index to device..." << std::endl;
        CHECK_CUDA(cudaMalloc(&d_centroids_, h_centroids_.size() * sizeof(float)));
        CHECK_CUDA(cudaMemcpy(d_centroids_, h_centroids_.data(), h_centroids_.size() * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMalloc(&d_reordered_base_, h_reordered_base_.size() * sizeof(float)));
        CHECK_CUDA(cudaMemcpy(d_reordered_base_, h_reordered_base_.data(), h_reordered_base_.size() * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMalloc(&d_list_offsets_, h_list_offsets_.size() * sizeof(int)));
        CHECK_CUDA(cudaMemcpy(d_list_offsets_, h_list_offsets_.data(), h_list_offsets_.size() * sizeof(int), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMalloc(&d_list_sizes_, h_list_sizes_.size() * sizeof(int)));
        CHECK_CUDA(cudaMemcpy(d_list_sizes_, h_list_sizes_.data(), h_list_sizes_.size() * sizeof(int), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMalloc(&d_original_ids_, h_original_ids_.size() * sizeof(int)));
        CHECK_CUDA(cudaMemcpy(d_original_ids_, h_original_ids_.data(), h_original_ids_.size() * sizeof(int), cudaMemcpyHostToDevice));
        std::cout << "Index data transferred to GPU.\n";
    }


    void search_batch(
        const float* queries, size_t M, int k, int nprobe,
        std::vector<std::priority_queue<std::pair<float, int>>>& final_results,
        std::vector<float>& timings_ms) // 新增：用于返回各阶段耗时的vector
    {
        // --- 1. 创建CUDA Events ---
        cudaEvent_t start, stop;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));

        timings_ms.clear();
        timings_ms.reserve(4);

        // --- 2. 分配GPU内存 ---
        float* d_queries; CHECK_CUDA(cudaMalloc(&d_queries, (long long)M * D_ * sizeof(float)));
        float* d_ip_matrix; CHECK_CUDA(cudaMalloc(&d_ip_matrix, (long long)M * num_clusters_ * sizeof(float)));
        int* d_top_probes; CHECK_CUDA(cudaMalloc(&d_top_probes, (long long)M * nprobe * sizeof(int)));
        ResultPair* d_final_results; CHECK_CUDA(cudaMalloc(&d_final_results, (long long)M * k * sizeof(ResultPair)));
        
        // --- 3. 计时 H2D Transfer ---
        CHECK_CUDA(cudaEventRecord(start));
        CHECK_CUDA(cudaMemcpy(d_queries, queries, (long long)M * D_ * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        float time_h2d;
        CHECK_CUDA(cudaEventElapsedTime(&time_h2d, start, stop));
        timings_ms.push_back(time_h2d);
        
        // --- 4. 计时 S1: Probe Selection ---
        CHECK_CUDA(cudaEventRecord(start));
        
        const float alpha = 1.0f; const float beta = 0.0f;
        CHECK_CUBLAS(cublasSgemm(cublas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
            num_clusters_, M, D_, &alpha,
            d_centroids_, D_, d_queries, D_, &beta,
            d_ip_matrix, num_clusters_));

        dim3 gridDim_s1(M); 
        dim3 blockDim_s1(256);
        size_t shared_mem_s1 = blockDim_s1.x * nprobe * sizeof(ResultPair);
        find_top_n_probes_kernel_ip<<<gridDim_s1, blockDim_s1, shared_mem_s1>>>(d_ip_matrix, M, num_clusters_, nprobe, d_top_probes);
        CHECK_CUDA(cudaGetLastError());
        
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        float time_s1;
        CHECK_CUDA(cudaEventElapsedTime(&time_s1, start, stop));
        timings_ms.push_back(time_s1);

        // --- 5. 计时 S2: Intra-Cluster Search ---
        CHECK_CUDA(cudaEventRecord(start));

        dim3 gridDim_s2(M);
        dim3 blockDim_s2(256); 
        size_t shared_mem_s2 = blockDim_s2.x * sizeof(ResultPair);
        intra_cluster_search_kernel_ip<<<gridDim_s2, blockDim_s2, shared_mem_s2>>>(
            d_queries, d_reordered_base_, d_list_offsets_, d_list_sizes_, d_original_ids_,
            d_top_probes, M, D_, k, nprobe, num_clusters_, N_, d_final_results);
        CHECK_CUDA(cudaGetLastError());
        
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        float time_s2;
        CHECK_CUDA(cudaEventElapsedTime(&time_s2, start, stop));
        timings_ms.push_back(time_s2);

        // --- 6. 计时 D2H Transfer ---
        std::vector<ResultPair> h_results((long long)M * k);
        CHECK_CUDA(cudaEventRecord(start));
        CHECK_CUDA(cudaMemcpy(h_results.data(), d_final_results, (long long)M * k * sizeof(ResultPair), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        float time_d2h;
        CHECK_CUDA(cudaEventElapsedTime(&time_d2h, start, stop));
        timings_ms.push_back(time_d2h);

        // --- 7. CPU端后处理 (不计时) ---
        final_results.resize(M);
        for (size_t i = 0; i < M; ++i) {
            for (int j = 0; j < k; ++j) {
                ResultPair p = h_results[(long long)i * k + j];
                if(p.id != -1) {
                    final_results[i].push({ 1.0f - p.inner_product, p.id });
                }
            }
        }

        // --- 8. 清理资源 ---
        CHECK_CUDA(cudaFree(d_queries));
        CHECK_CUDA(cudaFree(d_ip_matrix));
        CHECK_CUDA(cudaFree(d_top_probes));
        CHECK_CUDA(cudaFree(d_final_results));
        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(stop));
    }
private:
    size_t D_, N_;
    int num_clusters_;
    cublasHandle_t cublas_handle_;
    
    std::vector<float> h_centroids_, h_reordered_base_;
    std::vector<int> h_list_offsets_, h_list_sizes_, h_original_ids_;

    float* d_centroids_ = nullptr;
    float* d_reordered_base_ = nullptr;
    int* d_list_offsets_ = nullptr;
    int* d_list_sizes_ = nullptr;
    int* d_original_ids_ = nullptr;
};

#endif // IVF_GPU_BASELINE_H