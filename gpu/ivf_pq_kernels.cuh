#ifndef IVF_PQ_KERNELS_H
#define IVF_PQ_KERNELS_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "ivf_pq_common.cuh"



// =================================================================
//                      S1: 候选簇选择 Kernel
// =================================================================
__global__ void find_top_n_probes_dist_kernel(
    const float* dist_matrix, int M, int num_clusters, int nprobe, int* top_probes_gpu)
{
    int query_idx = blockIdx.x;
    if (query_idx >= M) return;

    extern __shared__ ResultPair smem_probes[];
    const float* query_dists = dist_matrix + (long long)query_idx * num_clusters;
    int* query_probes_out = top_probes_gpu + (long long)query_idx * nprobe;

    ResultPair* local_top_probes = smem_probes + threadIdx.x * nprobe;
    for (int i = 0; i < nprobe; ++i) local_top_probes[i] = {1e9f, -1};
    
    for (int i = threadIdx.x; i < num_clusters; i += blockDim.x) {
        float current_dist = query_dists[i];
        if (current_dist < local_top_probes[nprobe - 1].distance) {
            local_top_probes[nprobe - 1] = {current_dist, i};
            for(int j = nprobe - 1; j > 0; --j){
                if(local_top_probes[j].distance < local_top_probes[j-1].distance){
                    ResultPair temp = local_top_probes[j];
                    local_top_probes[j] = local_top_probes[j-1];
                    local_top_probes[j-1] = temp;
                } else break;
            }
        }
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        for (int i = 1; i < blockDim.x; ++i) {
            ResultPair* other_probes = smem_probes + i * nprobe;
            for (int j = 0; j < nprobe; ++j) {
                if (other_probes[j].id == -1) break;
                if (other_probes[j].distance < local_top_probes[nprobe - 1].distance) {
                    local_top_probes[nprobe-1] = other_probes[j];
                    for(int l = nprobe - 1; l > 0; --l) {
                        if (local_top_probes[l].distance < local_top_probes[l-1].distance) {
                            ResultPair temp = local_top_probes[l];
                            local_top_probes[l] = local_top_probes[l-1];
                            local_top_probes[l-1] = temp;
                        } else break;
                    }
                }
            }
        }
        for (int i = 0; i < nprobe; ++i) {
            query_probes_out[i] = local_top_probes[i].id;
        }
    }
}

// =================================================================
//                      S2.1: LUT 预计算 Kernel
// =================================================================
// in ivf_pq_kernels.cuh
// __global__ void precompute_lut_kernel(
//     const float* queries, const float* centroids, const float* pq_codebooks, const int* top_probes,
//     int M, int D, int nprobe, int num_clusters,
//     float* luts)
// {
//     int query_idx = blockIdx.x;
//     if (query_idx >= M) return;

//     const float* query = queries + (long long)query_idx * D;
    
//     // 每个线程负责一个LUT条目 m_ks_idx = m * PQ_KS + ks
//     for (int m_ks_idx = threadIdx.x; m_ks_idx < PQ_M * PQ_KS; m_ks_idx += blockDim.x) {
//         int m = m_ks_idx / PQ_KS;
//         int ks = m_ks_idx % PQ_KS;

//         for(int p_idx = 0; p_idx < nprobe; ++p_idx) {
//             int cluster_id = top_probes[(long long)query_idx * nprobe + p_idx];
//             if (cluster_id < 0) continue;

//             const float* centroid_vec = centroids + (long long)cluster_id * D;

//             // [修正] 计算查询残差的子向量
//             float query_sub_residual[PQ_DS]; // 用栈上数组，比访问全局内存快
//             const float* query_sub = query + m * PQ_DS;
//             const float* centroid_sub = centroid_vec + m * PQ_DS;
//             for(int d = 0; d < PQ_DS; ++d) {
//                 query_sub_residual[d] = query_sub[d] - centroid_sub[d];
//             }

//             const float* codeword = pq_codebooks + (((long long)cluster_id * PQ_M + m) * PQ_KS + ks) * PQ_DS;
            
//             float dist_sq = 0.0f;
//             for (int d = 0; d < PQ_DS; ++d) {
//                 float diff = query_sub_residual[d] - codeword[d];
//                 dist_sq += diff * diff;
//             }
            
//             luts[((long long)query_idx * nprobe + p_idx) * PQ_M * PQ_KS + m_ks_idx] = dist_sq;
//         }
//     }
// }

// in ivf_pq_kernels.cuh

// =================================================================
//        S2.1: LUT 预计算 Kernel (REVISED OPTIMIZED VERSION)
// =================================================================
__global__ void precompute_lut_kernel(
    const float* queries, const float* centroids, const float* pq_codebooks, const int* top_probes,
    int M, int D, int nprobe, int num_clusters,
    float* luts)
{
    // 每个Block处理一个查询
    int query_idx = blockIdx.x;
    if (query_idx >= M) return;

    // --- 1. 优化：只将Query向量加载到共享内存 ---
    // 这是访问最频繁的数据，值得放入共享内存。
    extern __shared__ float smem_query[];

    const float* query_glob = queries + (long long)query_idx * D;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        smem_query[i] = query_glob[i];
    }
    __syncthreads(); // 确保Query加载完毕

    // --- 2. 并行计算所有LUT条目 ---
    int total_lut_entries = nprobe * PQ_M * PQ_KS;
    for (int entry_idx = threadIdx.x; entry_idx < total_lut_entries; entry_idx += blockDim.x) {
        
        // 解码索引
        int p_idx = entry_idx / (PQ_M * PQ_KS);
        int remainder = entry_idx % (PQ_M * PQ_KS);
        int m = remainder / PQ_KS;
        int ks = remainder % PQ_KS;

        int cluster_id = top_probes[(long long)query_idx * nprobe + p_idx];
        if (cluster_id < 0) continue;

        // 从全局内存读取质心和码字
        const float* centroid_glob = centroids + (long long)cluster_id * D;
        const float* codeword_glob = pq_codebooks + (((long long)cluster_id * PQ_M + m) * PQ_KS + ks) * PQ_DS;

        // 计算残差时，query部分来自共享内存，centroid部分来自全局内存
        const float* query_sub_smem = smem_query + m * PQ_DS;
        const float* centroid_sub_glob = centroid_glob + m * PQ_DS;

        float dist_sq = 0.0f;
        #pragma unroll
        for (int d = 0; d < PQ_DS; ++d) {
            float residual_d = query_sub_smem[d] - centroid_sub_glob[d];
            float diff = residual_d - codeword_glob[d];
            dist_sq += diff * diff;
        }
        
        luts[((long long)query_idx * nprobe + p_idx) * (PQ_M * PQ_KS) + remainder] = dist_sq;
    }
}

// =================================================================
//                      S2.2: ADC 近似搜索 Kernel
// =================================================================
__global__ void adc_search_kernel(
    const float* luts, const unsigned char* pq_codes, const int* list_offsets, 
    const int* list_sizes, const int* top_probes,
    int M, int nprobe, int rerank_top_n,
    ResultPair* rerank_candidates_gpu)
{
    int query_idx = blockIdx.x;
    if (query_idx >= M) return;

    extern __shared__ ResultPair smem_bests[]; // size: blockDim.x
    ResultPair local_best = {1e9f, -1};

    for (int p_idx = 0; p_idx < nprobe; ++p_idx) {
        int cluster_id = top_probes[(long long)query_idx * nprobe + p_idx];
        if (cluster_id < 0) continue;

        const float* lut = luts + ((long long)query_idx * nprobe + p_idx) * PQ_M * PQ_KS;
        int list_offset = list_offsets[cluster_id];
        int list_size = list_sizes[cluster_id];

        for (int i = threadIdx.x; i < list_size; i += blockDim.x) {
            long long point_offset = list_offset + i;
            const unsigned char* code = pq_codes + point_offset * PQ_M;
            
            float approx_dist_sq = 0;
            #pragma unroll
            for (int m = 0; m < PQ_M; ++m) {
                approx_dist_sq += lut[m * PQ_KS + code[m]];
            }
            
            if (approx_dist_sq < local_best.distance) {
                local_best = {approx_dist_sq, (int)point_offset};
            }
        }
    }
    
    smem_bests[threadIdx.x] = local_best;
    __syncthreads();

    // 块内归约，选出 rerank_top_n 个候选
    if (threadIdx.x == 0) {
        for (int i = 0; i < rerank_top_n; ++i) {
            int min_idx = i;
            for (int j = i + 1; j < blockDim.x; ++j) {
                if (smem_bests[j].distance < smem_bests[min_idx].distance) {
                    min_idx = j;
                }
            }
            ResultPair temp = smem_bests[i];
            smem_bests[i] = smem_bests[min_idx];
            smem_bests[min_idx] = temp;
        }
        for (int i = 0; i < rerank_top_n; ++i) {
            rerank_candidates_gpu[(long long)query_idx * rerank_top_n + i] = smem_bests[i];
        }
    }
}

// =================================================================
//                      S3: Rerank Kernel 辅助
// =================================================================

// 从 ResultPair 中提取出点的偏移量
__global__ void extract_offsets_kernel(const ResultPair* rerank_cand_pairs, int* rerank_cand_offsets, int M, int rerank_top_n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * rerank_top_n) return;
    rerank_cand_offsets[idx] = rerank_cand_pairs[idx].id; // 这里的id存的是point_offset
}

// =================================================================
//                      S3: Rerank Kernel
// =================================================================
__global__ void rerank_kernel(
    const float* queries, const float* reordered_base, const int* original_ids,
    const int* rerank_point_offsets,
    int M, int D, int K, int rerank_top_n,
    ResultPair* final_results_gpu)
{
    int query_idx = blockIdx.x;
    if (query_idx >= M) return;
    
    extern __shared__ ResultPair smem_reranked[]; // size: rerank_top_n
    
    const float* query = queries + (long long)query_idx * D;
    const int* offsets_for_query = rerank_point_offsets + (long long)query_idx * rerank_top_n;
    
    // 并行计算精确距离
    for(int i = threadIdx.x; i < rerank_top_n; i += blockDim.x) {
        int point_offset = offsets_for_query[i];
        if(point_offset < 0) {
            smem_reranked[i] = {1e9f, -1};
            continue;
        }

        const float* base = reordered_base + (long long)point_offset * D;
        
        float dist_sq = 0.0f;
        for(int d = 0; d < D; ++d) {
            float diff = query[d] - base[d];
            dist_sq += diff * diff;
        }
        smem_reranked[i] = {dist_sq, original_ids[point_offset]};
    }
    __syncthreads();
    
    // 块内排序选出Top-K
    if (threadIdx.x == 0) {
        for (int i = 0; i < K; ++i) {
            int min_idx = i;
            for (int j = i + 1; j < rerank_top_n; ++j) {
                if (smem_reranked[j].distance < smem_reranked[min_idx].distance) {
                    min_idx = j;
                }
            }
            ResultPair temp = smem_reranked[i];
            smem_reranked[i] = smem_reranked[min_idx];
            smem_reranked[min_idx] = temp;
        }
        for (int i = 0; i < K; ++i) {
            final_results_gpu[(long long)query_idx * K + i] = smem_reranked[i];
        }
    }
}
// =================================================================
//                      辅助：范数计算与广播加法 Kernel
// =================================================================
/*
 *  Kernel to compute the L2 norm squared for each row vector in a matrix.
 *  output_norms is a column vector.
 *  matrix is of size num_vectors x dim.
 *  output_norms is of size num_vectors x 1.
 */
 __global__ void l2_norm_squared_kernel(const float* matrix, float* output_norms, int num_vectors, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_vectors) {
        float norm_sq = 0.0f;
        const float* vector = matrix + (long long)idx * dim;
        for (int i = 0; i < dim; ++i) {
            norm_sq += vector[i] * vector[i];
        }
        output_norms[idx] = norm_sq;
    }
}


/*
 *  Kernel to add two matrices with broadcasting.
 *  C = A + B, where B is a column vector broadcasted across A's columns.
 *  A is of size M x N, B is of size M x 1.
 */
__global__ void add_broadcast_col_kernel(float* matrix_A, const float* vector_B_col, int M, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        // A is M x N matrix, its element is at row * N + col
        // B is M x 1 vector, its element is at row
        matrix_A[row * N + col] += vector_B_col[row];
    }
}


/*
 *  Kernel to add two matrices with broadcasting.
 *  C = A + B, where B is a row vector broadcasted across A's rows.
 *  A is of size M x N, B is of size 1 x N.
 */
__global__ void add_broadcast_row_kernel(float* matrix_A, const float* vector_B_row, int M, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        // A is M x N matrix, its element is at row * N + col
        // B is 1 x N vector, its element is at col
        matrix_A[row * N + col] += vector_B_row[col];
    }
}
// in ivf_pq_kernels.cuh
__global__ void transpose_kernel(const float* input, float* output, int N, int M) {
    // N: rows of input, M: cols of input
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * M) return;

    int r = idx / M;
    int c = idx % M;
    output[c * N + r] = input[r * M + c];
}
// in ivf_pq_kernels.cuh
// A: M x K, B: N x K, C: M x N.  Computes C = A * B^T
__global__ void gemm_kernel(const float* A, const float* B, float* C, int M, int N, int K, float alpha) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int i = 0; i < K; ++i) {
            sum += A[row * K + i] * B[col * K + i];
        }
        C[row * N + col] = alpha * sum;
    }
}
#endif // IVF_PQ_KERNELS_H