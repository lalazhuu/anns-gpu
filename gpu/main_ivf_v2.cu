// main_ivf_v2.cu
// 用于测试和运行 Cluster-at-a-time 高性能精确IVF (ivf_gpu_v2.cuh)

#include <vector>
#include <cstring>
#include <string>
#include <iostream>
#include <fstream>
#include <set>
#include <chrono>
#include <iomanip>
#include <sstream>

#include "ivf_gpu_v2.cuh" 


template<typename T>
T *LoadData(std::string data_path, size_t& n, size_t& d) {
    std::ifstream fin;
    fin.open(data_path, std::ios::in | std::ios::binary);
    if (!fin) {
        std::cerr << "Error opening file: " << data_path << std::endl;
        exit(EXIT_FAILURE);
    }
    
    uint32_t n_32, d_32;
    fin.read((char*)&n_32, sizeof(uint32_t));
    fin.read((char*)&d_32, sizeof(uint32_t));
    n = n_32;
    d = d_32;
    
    T* data = new T[n*d];
    size_t sz = sizeof(T);
    fin.read((char*)data, n * d * sz);
    fin.close();

    std::cerr << "load data " << data_path << "\n";
    std::cerr << "dimension: " << d << "  number:" << n << "  size_per_element:" << sizeof(T) << "\n";

    return data;
}

float calculate_recall(
    std::vector<std::priority_queue<std::pair<float, int>>>& results,
    const int* gt_data, size_t M, int k, size_t gt_dim)
{
    float total_recall = 0;
    #pragma omp parallel for reduction(+:total_recall)
    for (size_t i = 0; i < M; ++i) {
        std::set<int> gtset;
        for (int j = 0; j < k; ++j) {
            gtset.insert(gt_data[j + i * gt_dim]);
        }
        
        auto res_pq = results[i]; 
        size_t acc = 0;
        while (!res_pq.empty()) {
            if (gtset.count(res_pq.top().second)) { ++acc; }
            res_pq.pop();
        }
        total_recall += (float)acc / k;
    }
    return total_recall / M;
}



void print_performance(const std::string& version_name, const std::vector<float>& timings,
                       float recall, size_t M, int nprobe, int k)
{
    std::cout << "\n\n<<<<<<<<<<<<<<< PERFORMANCE REPORT (" << version_name << ") >>>>>>>>>>>>>>>\n";
    std::cout << "------------------------------------------------------------\n";
    // 参数行不再打印 rerank_size
    std::cout << "Parameters: nprobe=" << nprobe << ", k=" << k << "\n";
    std::cout << "------------------------------------------------------------\n";

    float total_gpu_time = 0;
    // 最终方案的 timings 应该有6个元素，但第5个(S4)是0
    if(timings.size() == 6) { 
        for(const auto& t : timings) total_gpu_time += t;
        
        std::cout << std::fixed << std::setprecision(4);
        std::cout << "  [H2D]    Host to Device Transfer: " << timings[0] << " ms (" << (timings[0]/total_gpu_time)*100 << "%)\n";
        std::cout << "  [S1]     Probe Selection:         " << timings[1] << " ms (" << (timings[1]/total_gpu_time)*100 << "%)\n";
        std::cout << "  [S2]     CPU Scheduling:          " << timings[2] << " ms (" << (timings[2]/total_gpu_time)*100 << "%)\n";
        // 标签修改为更准确的 "Search & Sort"
        std::cout << "  [S3]     Search & Sort:           " << timings[3] << " ms (" << (timings[3]/total_gpu_time)*100 << "%)\n";
        // S4阶段被移除，时间为0，可以不打印或打印为0
        std::cout << "  [S4]     (Removed Stage):         " << timings[4] << " ms (" << (timings[4]/total_gpu_time)*100 << "%)\n";
        std::cout << "  [D2H]    Device to Host Transfer: " << timings[5] << " ms (" << (timings[5]/total_gpu_time)*100 << "%)\n";
        
        std::cout << "------------------------------------------------------------\n";
        std::cout << "  Total GPU Time (from events): " << total_gpu_time << " ms\n";
        std::cout << "  Avg Latency per Query:        " << (total_gpu_time * 1000.0 / M) << " us\n";
    } else {
        std::cout << "  Error: Timing data is incomplete. Expected 6 timing values.\n";
    }
    
    std::cout << "  Overall Avg Recall:           " << recall << "\n";
    std::cout << "------------------------------------------------------------\n";
}

int main(int argc, char *argv[])
{
    size_t test_number = 0, base_number = 0;
    size_t test_gt_d = 0, vecdim = 0;

    std::string data_path = "./anndata/"; 
    auto test_query = LoadData<float>(data_path + "DEEP100K.query.fbin", test_number, vecdim);
    auto test_gt = LoadData<int>(data_path + "DEEP100K.gt.query.100k.top100.bin", test_number, test_gt_d);
    auto base = LoadData<float>(data_path + "DEEP100K.base.100k.fbin", base_number, vecdim);
    
    const size_t M = 2000;

    // --- 设置参数 ---
    const int k = 10;
    const int num_clusters = 512;
    const int nprobe = 10;

    // const int rerank_size = 10; 
    
    // --- 运行 Cluster-at-a-time 精确IVF (V2 - 混合方案) ---
    {
        IvfGpuV2 ivf_gpu_v2(vecdim, num_clusters);
        const std::string index_path_prefix = "files/my_ivf_v2_index";

        if (!ivf_gpu_v2.LoadIndex(index_path_prefix, base_number)) {
            std::cout << "IVF_V2 index not found. Building new index..." << std::endl;
            ivf_gpu_v2.Build(base, base_number);
            ivf_gpu_v2.SaveIndex(index_path_prefix);
        }
        
        ivf_gpu_v2.TransferToDevice();

        std::vector<std::priority_queue<std::pair<float, int>>> batch_results;
        std::vector<float> timings_ms;

        ivf_gpu_v2.search_batch(test_query, M, k, nprobe, batch_results, timings_ms);
        
        float recall = calculate_recall(batch_results, test_gt, M, k, test_gt_d);

        print_performance("IVF_V2 (Final)", timings_ms, recall, M, nprobe, k);
    }

    // 释放CPU内存
    delete[] test_query;
    delete[] test_gt;
    delete[] base;

    return 0;
}