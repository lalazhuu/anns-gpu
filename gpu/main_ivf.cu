#include <vector>
#include <cstring>
#include <string>
#include <iostream>
#include <fstream>
#include <set>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <sys/time.h>
#include <omp.h>
//#include "ivf_gpu_baseline.cuh"
//#include "ivf_gpu_baseline_v2.cuh"
//#include "ivf_gpu_v1_optimized.cuh"
//#include "ivf_gpu_v1_time.cuh"
// #include "ivf_pq_gpu.cuh"
#include "ivf_pq_gpu_v2.cuh"

// --- LoadData 函数 (保持不变) ---
template<typename T>
T *LoadData(std::string data_path, size_t& n, size_t& d)
{
    std::ifstream fin;
    fin.open(data_path, std::ios::in | std::ios::binary);
    if (!fin) {
        std::cerr << "Error opening file: " << data_path << std::endl;
        exit(EXIT_FAILURE);
    }
    fin.read((char*)&n,4);
    fin.read((char*)&d,4);
    T* data = new T[n*d];
    size_t sz = sizeof(T);
    fin.read((char*)data, n * d * sz);
    fin.close();

    std::cerr<<"load data "<<data_path<<"\n";
    std::cerr<<"dimension: "<<d<<"  number:"<<n<<"  size_per_element:"<<sizeof(T)<<"\n";

    return data;
}

// --- main 函数 (完整修正版) ---
int main(int argc, char *argv[])
{
    size_t test_number = 0, base_number = 0;
    size_t test_gt_d = 0, vecdim = 0;

    // 请确保这个路径对于你的执行环境是正确的
    std::string data_path = "./anndata/"; 
    auto test_query = LoadData<float>(data_path + "DEEP100K.query.fbin", test_number, vecdim);
    auto test_gt = LoadData<int>(data_path + "DEEP100K.gt.query.100k.top100.bin", test_number, test_gt_d);
    auto base = LoadData<float>(data_path + "DEEP100K.base.100k.fbin", base_number, vecdim);
    
    // 只测试前2000条查询
    test_number = 2000;

    // const size_t k = 10;
    // const int num_clusters = 512; // IVF参数
    // const int nprobe = 10;         // IVF参数
    
    // // 定义索引文件路径，必须在 'files/' 目录下
    // const std::string index_path_prefix = "files/my_ivf_index";

    // // 1. 创建IVF-GPU实例
    // //IvfGpuBaseline ivf_gpu(vecdim, num_clusters);
    // //IvfGpuV2 ivf_gpu(vecdim, num_clusters);
    // //IvfGpuV1_Optimized ivf_gpu(vecdim, num_clusters);
    // IvfGpuV1_Timed ivf_gpu(vecdim, num_clusters);


    // // 2. 尝试加载索引，如果失败，则构建并保存
    // if (!ivf_gpu.LoadIndex(index_path_prefix, base_number)) {
    //     std::cout << "Index file not found or failed to load. Building new index..." << std::endl;
    //     ivf_gpu.Build(base, base_number);
    //     ivf_gpu.SaveIndex(index_path_prefix);
    // }
    
    // // 3. 将索引数据传输到GPU
    // ivf_gpu.TransferToDevice();

    // // 4. 准备接收结果
    // std::vector<std::priority_queue<std::pair<float, int>>> batch_results;
    // std::vector<float> timings_ms; // 用于存储各阶段耗时
    // // 新增：用于接收S2内部计时的容器
    // std::vector<long long> s2_internal_timings;

    // // 5. 计时和查询
    // // 总计时器仍然保留，用于验证和对比
    // auto start_time = std::chrono::high_resolution_clock::now();
    
    // //ivf_gpu.search_batch(test_query, test_number, k, nprobe, batch_results, timings_ms);
    // ivf_gpu.search_batch(
    //     test_query, test_number, k, nprobe, 
    //     batch_results, timings_ms, 
    //     s2_internal_timings // 传入新的容器
    // );

    // // 这句可以省略了，因为search_batch内部已经做了同步
    // // CHECK_CUDA(cudaDeviceSynchronize()); 

    // auto end_time = std::chrono::high_resolution_clock::now();
    // auto total_latency_us = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
    // // 6. 评估整个批次的结果
    // float avg_recall = 0;
    // for (size_t i = 0; i < test_number; ++i) {
    //     std::set<uint32_t> gtset;
    //     for (size_t j = 0; j < k; ++j) {
    //         gtset.insert(test_gt[j + i * test_gt_d]);
    //     }

    //     auto& res_pq = batch_results[i];
    //     size_t acc = 0;
    //     while (!res_pq.empty()) {
    //         // priority_queue是最大堆，我们存的是距离，所以top()是最大的距离（最不相似）
    //         // 但是我们的评估需要ID，所以取 .second
    //         int x = res_pq.top().second;
    //         if (gtset.count(x)) {
    //             ++acc;
    //         }
    //         res_pq.pop();
    //     }
    //     avg_recall += (float)acc / k;
    // }
    // // --- 7. 打印性能分析结果 ---
    // std::cout << "-------------------------------------------\n";
    // std::cout << "Performance Breakdown (ms):\n";
    // std::cout << std::fixed << std::setprecision(4);
    // if (timings_ms.size() == 4) {
    //     float total_gpu_time = timings_ms[0] + timings_ms[1] + timings_ms[2] + timings_ms[3];
    //     std::cout << "  H2D Transfer:      " << timings_ms[0] << " ms (" << (timings_ms[0]/total_gpu_time)*100 << "%)\n";
    //     std::cout << "  S1 Probe Selection:  " << timings_ms[1] << " ms (" << (timings_ms[1]/total_gpu_time)*100 << "%)\n";
    //     std::cout << "  S2 Search:           " << timings_ms[2] << " ms (" << (timings_ms[2]/total_gpu_time)*100 << "%)\n";
    //     std::cout << "  D2H Transfer:      " << timings_ms[3] << " ms (" << (timings_ms[3]/total_gpu_time)*100 << "%)\n";
    //     std::cout << "-------------------------------------------\n";
    //     std::cout << "Total GPU Time (from events): " << total_gpu_time << " ms\n";
    // }
    // std::cout << "Total End-to-End Time (chrono): " << (double)total_latency_us / 1000.0 << " ms\n";

    // std::cout << "-------------------------------------------\n";
    // std::cout << "Average recall: " << avg_recall / test_number << "\n";
    // std::cout << "Average latency per query (us): " << (double)total_latency_us / test_number << "\n";


    // // ==========================================================
    // //        8. 打印S2内部精细计时分析报告 (核心新增部分)
    // // ==========================================================
    // long long total_find_best_clocks = 0;
    // long long total_reduction_clocks = 0;

    // for (size_t i = 0; i < test_number; ++i) {
    //     total_find_best_clocks += s2_internal_timings[i * 2 + 0];
    //     total_reduction_clocks += s2_internal_timings[i * 2 + 1];
    // }

    // double avg_find_best_clocks = (double)total_find_best_clocks / test_number;
    // double avg_reduction_clocks = (double)total_reduction_clocks / test_number;
    // double total_avg_s2_clocks = avg_find_best_clocks + avg_reduction_clocks;

    // std::cout << "\n\n<<<<<<<<<<<<<<< S2 KERNEL INTERNAL TIMING ANALYSIS >>>>>>>>>>>>>>>\n";
    // std::cout << "------------------------------------------------------------------\n";
    // std::cout << "Analysis based on " << test_number << " queries (raw clock cycles):\n";
    // std::cout << "------------------------------------------------------------------\n";
    // std::cout << std::fixed << std::setprecision(2);
    // if(total_avg_s2_clocks > 0) {
    //     std::cout << "  [1] Avg. Find Best (Compute/Mem): " << std::setw(12) << avg_find_best_clocks << " cycles ("
    //             << (avg_find_best_clocks / total_avg_s2_clocks) * 100.0 << "%)\n";
    //     std::cout << "  [2] Avg. Reduction (Sort in SMEM):  " << std::setw(12) << avg_reduction_clocks << " cycles ("
    //             << (avg_reduction_clocks / total_avg_s2_clocks) * 100.0 << "%)\n";
    // }
    // std::cout << "------------------------------------------------------------------\n";


    // --- 2. 设置 IVF-PQ 参数 ---
    const size_t k = 10;
    const int num_clusters = 512;
    const int nprobe = 11;           // IVF-PQ通常可以用更小的nprobe
    const int rerank_top_n = 32;    // ADC搜索后，选出64个候选进行精确重排
    
    // // 定义新的IVF-PQ索引文件路径
    const std::string index_path_prefix = "files/my_ivf_pq_index";

    // // --- 3. 创建IVF-PQ实例 ---
    // IvfPqGpu ivf_pq_gpu(vecdim, num_clusters);

    // // --- 4. 构建或加载索引 ---
    // // 首次运行时，索引不存在，会进入Build流程
    // if (!ivf_pq_gpu.LoadIndex(index_path_prefix, base_number)) {
    //     std::cout << "IVF-PQ index not found. Building new index (this may take a while)..." << std::endl;
    //     auto build_start = std::chrono::high_resolution_clock::now();
    //     ivf_pq_gpu.Build(base, base_number);
    //     auto build_end = std::chrono::high_resolution_clock::now();
    //     auto build_duration = std::chrono::duration_cast<std::chrono::seconds>(build_end - build_start).count();
    //     std::cout << "Index built in " << build_duration << " seconds.\n";
        
    //     std::cout << "Saving index for future use...\n";
    //     ivf_pq_gpu.SaveIndex(index_path_prefix);
    // }
    
    // // --- 5. 将索引数据传输到GPU ---
    // ivf_pq_gpu.TransferToDevice();

    // // --- 6. 准备接收结果 ---
    // std::vector<std::priority_queue<std::pair<float, int>>> batch_results;
    // std::vector<float> timings_ms;

    // // --- 7. 计时和查询 ---
    // std::cout << "\nStarting IVF-PQ batch search..." << std::endl;
    // ivf_pq_gpu.search_batch(
    //     test_query, test_number, k, nprobe, rerank_top_n, 
    //     batch_results, timings_ms
    // );

    // // --- 8. 评估总召回率 ---
    // float avg_recall = 0;
    // for (size_t i = 0; i < test_number; ++i) {
    //     std::set<int> gtset;
    //     for (size_t j = 0; j < k; ++j) {
    //         gtset.insert(test_gt[j + i * test_gt_d]);
    //     }
    //     auto& res_pq = batch_results[i];
    //     size_t acc = 0;
    //     while (!res_pq.empty()) {
    //         // priority_queue是最大堆，我们存的是 -distance，所以top()是最小的-distance（即最大距离）
    //         // .second 是 ID
    //         if (gtset.count(res_pq.top().second)) { ++acc; }
    //         res_pq.pop();
    //     }
    //     avg_recall += (float)acc / k;
    // }

    // // --- 9. 打印详细的性能分析结果 ---
    // std::cout << "\n\n<<<<<<<<<<<<<<< PERFORMANCE REPORT (IVF-PQ) >>>>>>>>>>>>>>>\n";
    // std::cout << "------------------------------------------------------------\n";
    // std::cout << "Parameters: nprobe=" << nprobe << ", k=" << k << ", rerank_n=" << rerank_top_n << "\n";
    // std::cout << "------------------------------------------------------------\n";
    // if (timings_ms.size() == 6) { // 确保计时数据完整
    //     float total_gpu_time = timings_ms[0] + timings_ms[1] + timings_ms[2] + timings_ms[3] + timings_ms[4] + timings_ms[5];
    //     std::cout << std::fixed << std::setprecision(4);
    //     std::cout << "  [H2D]    Host to Device Transfer: " << timings_ms[0] << " ms (" << (timings_ms[0]/total_gpu_time)*100 << "%)\n";
    //     std::cout << "  [S1]     Probe Selection:         " << timings_ms[1] << " ms (" << (timings_ms[1]/total_gpu_time)*100 << "%)\n";
    //     std::cout << "  [S2.1]   LUT Precomputation:      " << timings_ms[2] << " ms (" << (timings_ms[2]/total_gpu_time)*100 << "%)\n";
    //     std::cout << "  [S2.2]   ADC Search:              " << timings_ms[3] << " ms (" << (timings_ms[3]/total_gpu_time)*100 << "%)\n";
    //     std::cout << "  [S3]     Rerank:                  " << timings_ms[4] << " ms (" << (timings_ms[4]/total_gpu_time)*100 << "%)\n";
    //     std::cout << "  [D2H]    Device to Host Transfer: " << timings_ms[5] << " ms (" << (timings_ms[5]/total_gpu_time)*100 << "%)\n";
    //     std::cout << "------------------------------------------------------------\n";
    //     std::cout << "  Total GPU Time (from events): " << total_gpu_time << " ms\n";
    //     std::cout << "  Avg Latency per Query:        " << (total_gpu_time * 1000.0 / test_number) << " us\n";
    // }
    // std::cout << "  Overall Avg Recall:           " << avg_recall / test_number << "\n";
    // std::cout << "------------------------------------------------------------\n";


    // --- [修改2] 创建IvfPqGpuV2实例 ---
    IvfPqGpuV2 ivf_pq_gpu_v2(vecdim, num_clusters);

    // --- [修改3] 使用新的v2对象调用方法 ---
    if (!ivf_pq_gpu_v2.LoadIndex(index_path_prefix, base_number)) {
        std::cout << "IVF-PQ index not found. Building new index (this may take a while)..." << std::endl;
        auto build_start = std::chrono::high_resolution_clock::now();
        ivf_pq_gpu_v2.Build(base, base_number);
        auto build_end = std::chrono::high_resolution_clock::now();
        auto build_duration = std::chrono::duration_cast<std::chrono::seconds>(build_end - build_start).count();
        std::cout << "Index built in " << build_duration << " seconds.\n";
        
        std::cout << "Saving index for future use...\n";
        ivf_pq_gpu_v2.SaveIndex(index_path_prefix);
    }
    
    ivf_pq_gpu_v2.TransferToDevice();

    // --- 准备接收结果 ---
    std::vector<std::priority_queue<std::pair<float, int>>> batch_results;
    std::vector<float> timings_ms;

    // --- [修改3] 调用V2的search_batch ---
    bool debug_mode = false; 

    std::cout << "\nStarting IVF-PQ (V2 Fused Kernel) batch search..." << std::endl;
    ivf_pq_gpu_v2.search_batch(
        test_query, test_number, k, nprobe, rerank_top_n, 
        batch_results, timings_ms,
        debug_mode // 传入调试标志
    );

    // --- 评估总召回率 (逻辑不变) ---
    float avg_recall = 0;
    for (size_t i = 0; i < test_number; ++i) {
        std::set<int> gtset;
        for (size_t j = 0; j < k; ++j) {
            gtset.insert(test_gt[j + i * test_gt_d]);
        }
        auto& res_pq = batch_results[i];
        size_t acc = 0;
        while (!res_pq.empty()) {
            if (gtset.count(res_pq.top().second)) { ++acc; }
            res_pq.pop();
        }
        avg_recall += (float)acc / k;
    }

    // --- 打印详细的性能分析结果 (V2版本) ---
    std::cout << "\n\n<<<<<<<<<<<<<<< PERFORMANCE REPORT (IVF-PQ V2) >>>>>>>>>>>>>>>\n";
    std::cout << "------------------------------------------------------------\n";
    std::cout << "Parameters: nprobe=" << nprobe << ", k=" << k << ", rerank_n=" << rerank_top_n << "\n";
    std::cout << "------------------------------------------------------------\n";
    if (timings_ms.size() == 5) { // 确保计时数据完整 (V2版本是5个阶段)
        float total_gpu_time = timings_ms[0] + timings_ms[1] + timings_ms[2] + timings_ms[3] + timings_ms[4];
        std::cout << std::fixed << std::setprecision(4);
        std::cout << "  [H2D]    Host to Device Transfer: " << timings_ms[0] << " ms (" << (timings_ms[0]/total_gpu_time)*100 << "%)\n";
        std::cout << "  [S1]     Probe Selection:         " << timings_ms[1] << " ms (" << (timings_ms[1]/total_gpu_time)*100 << "%)\n";
        std::cout << "  [S2]     Fused ADC Search:        " << timings_ms[2] << " ms (" << (timings_ms[2]/total_gpu_time)*100 << "%)\n";
        std::cout << "  [S3]     Rerank:                  " << timings_ms[3] << " ms (" << (timings_ms[3]/total_gpu_time)*100 << "%)\n";
        std::cout << "  [D2H]    Device to Host Transfer: " << timings_ms[4] << " ms (" << (timings_ms[4]/total_gpu_time)*100 << "%)\n";
        std::cout << "------------------------------------------------------------\n";
        std::cout << "  Total GPU Time (from events): " << total_gpu_time << " ms\n";
        std::cout << "  Avg Latency per Query:        " << (total_gpu_time * 1000.0 / test_number) << " us\n";
    }
    std::cout << "  Overall Avg Recall:           " << avg_recall / test_number << "\n";
    std::cout << "------------------------------------------------------------\n";
    // 释放CPU内存
    delete[] test_query;
    delete[] test_gt;
    delete[] base;

    return 0;
}