#ifndef IVF_PQ_KERNELS_V2_H
#define IVF_PQ_KERNELS_V2_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "ivf_pq_common.cuh"

__global__ void final_simulated_and_verified_kernel(
    const float* queries, const float* centroids, const float* pq_codebooks, 
    const unsigned char* pq_codes, const int* list_offsets, const int* list_sizes, 
    const int* top_probes,
    int M, int D, int nprobe, int rerank_top_n,
    ResultPair* rerank_candidates_gpu)
{
    // --- 1. Setup ---
    int query_idx = blockIdx.x;
    if (query_idx >= M) return;

    int tid = threadIdx.x;
    const float* query_vec = queries + (long long)query_idx * D;
    const int* probes_for_query = top_probes + (long long)query_idx * nprobe;

    // --- 2. Shared Memory Allocation ---
    extern __shared__ char s_mem[];
    float* s_lut = (float*)s_mem;
    ResultPair* s_bests = (ResultPair*)(s_mem + PQ_M * PQ_KS * sizeof(float));
    
    // --- 3. Find Thread-Local Best Candidate ---
    // 每个线程只负责找到它自己能看到的所有向量里最好的那一个
    ResultPair thread_best = {1e9f, -1};
    
    // 遍历所有分配给这个查询的探针
    for (int p_idx = 0; p_idx < nprobe; ++p_idx) {
        int cluster_id = probes_for_query[p_idx];
        if (cluster_id < 0) continue;

        // 3.1. 计算当前探针的LUT到共享内存 (此逻辑已验证正确)
        const float* centroid_vec = centroids + (long long)cluster_id * D;
        const float* pq_codebook_cluster = pq_codebooks + ((long long)cluster_id * PQ_M * PQ_KS * PQ_DS);
        for (int lut_idx = tid; lut_idx < PQ_M * PQ_KS; lut_idx += blockDim.x) {
            int m = lut_idx / PQ_KS;
            int ks = lut_idx % PQ_KS;
            const float* codeword = pq_codebook_cluster + (m * PQ_KS + ks) * PQ_DS;
            float dist_sq = 0.0f;
            for (int d = 0; d < PQ_DS; ++d) {
                float residual_d = query_vec[m * PQ_DS + d] - centroid_vec[m * PQ_DS + d];
                float diff = residual_d - codeword[d];
                dist_sq += diff * diff;
            }
            s_lut[lut_idx] = dist_sq;
        }
        __syncthreads(); 

        // 3.2. 扫描倒排列表 (此逻辑已验证正确)
        int list_offset = list_offsets[cluster_id];
        int list_size   = list_sizes[cluster_id];
        for (int i = tid; i < list_size; i += blockDim.x) {
            long long point_offset = list_offset + i;
            const unsigned char* code = pq_codes + point_offset * PQ_M;
            float approx_dist_sq = 0.0f;
            #pragma unroll
            for (int m = 0; m < PQ_M; ++m) {
                approx_dist_sq += s_lut[m * PQ_KS + code[m]];
            }
            if (approx_dist_sq < thread_best.distance) {
                thread_best = {approx_dist_sq, (int)point_offset};
            }
        }
        __syncthreads(); // 确保所有线程都处理完当前列表，再进入下一个探针的LUT计算
    }

    // --- 4. Block-Level Reduction (修正后的无信息损失版本) ---
    // 每个线程把自己找到的唯一最优候选写入共享内存
    s_bests[tid] = thread_best;
    __syncthreads();

    // --- 5. Final Sort by thread 0 ---
    // 现在 s_bests 中有 blockDim.x (例如256) 个候选者
    // 我们需要在这些候选中选出 rerank_top_n (例如32) 个
    if (tid == 0) {
        int num_candidates = blockDim.x;
        // 我们要排序的数量，不能超过 rerank_top_n，也不能超过实际拥有的候选数量
        int sort_limit = min(rerank_top_n, num_candidates);

        // 使用一个简单的选择排序，对共享内存中的数据进行排序
        for (int i = 0; i < sort_limit; ++i) {
            int min_idx = i;
            for (int j = i + 1; j < num_candidates; ++j) {
                if (s_bests[j].distance < s_bests[min_idx].distance) {
                    min_idx = j;
                }
            }
            // 交换，将最小的元素放到前面
            ResultPair temp = s_bests[i];
            s_bests[i] = s_bests[min_idx];
            s_bests[min_idx] = temp;
        }

        // 写回最终的 rerank_top_n 个结果到全局内存
        for (int i = 0; i < rerank_top_n; ++i) {
            // 如果我们拥有的候选数量不足 rerank_top_n，用无效值填充
            if (i < num_candidates) {
                 rerank_candidates_gpu[(long long)query_idx * rerank_top_n + i] = s_bests[i];
            } else {
                 rerank_candidates_gpu[(long long)query_idx * rerank_top_n + i] = {1e9f, -1};
            }
        }
    }
}
#endif