#define __builtin_ia32_serialize() do {} while (0)
#define _serialize_h

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
//#include "gpu_search.cuh"
//#include "gpu_search_v2.cuh"
// #include "gpu_search_v3.cuh"
// #include "gpu_search_v4.cuh"
// #include "gpu_search_v5.cuh"
// #include "gpu_search_v6.cuh"
// #include "gpu_search_v7.cuh"
// #include "gpu_search_v8.cuh"
 #include "gpu_search_v9.cuh"

template<typename T>
T *LoadData(std::string data_path, size_t& n, size_t& d)
{
    std::ifstream fin;
    fin.open(data_path, std::ios::in | std::ios::binary);
    if (!fin.is_open()) {
        std::cerr << "Error opening file: " << data_path << std::endl;
        exit(EXIT_FAILURE);
    }
    fin.read((char*)&n,4);
    fin.read((char*)&d,4);
    T* data = new T[n*d];
    int sz = sizeof(T);
    for(size_t i = 0; i < n; ++i){
        fin.read(((char*)data + i*d*sz), d*sz);
    }
    fin.close();

    std::cerr<<"load data "<<data_path<<"\n";
    std::cerr<<"dimension: "<<d<<"  number:"<<n<<"  size_per_element:"<<sizeof(T)<<"\n";

    return data;
}

struct SearchResult
{
    float recall;
    int64_t latency;
};

int main(int argc, char *argv[])
{
    size_t test_number = 0, base_number = 0;
    size_t test_gt_d = 0, vecdim = 0;

    std::string data_path = "./anndata/"; 
    auto test_query = LoadData<float>(data_path + "DEEP100K.query.fbin", test_number, vecdim);
    auto test_gt = LoadData<int>(data_path + "DEEP100K.gt.query.100k.top100.bin", test_number, test_gt_d);
    auto base = LoadData<float>(data_path + "DEEP100K.base.100k.fbin", base_number, vecdim);
    
    test_number = 2000;
    const size_t k = 10;

    // std::cout << "Starting GPU batch search for " << test_number << " queries..." << std::endl;
    
    // const unsigned long Converter = 1000 * 1000;
    // struct timeval total_start, total_end;
    // gettimeofday(&total_start, NULL);

    // // 调用GPU函数，它内部会打印详细耗时
    // auto all_gpu_results = gpu_flat_search_batch(base, test_query, base_number, test_number, vecdim, k);

    // gettimeofday(&total_end, NULL);
    // int64_t total_latency_us = (total_end.tv_sec * Converter + total_end.tv_usec) - (total_start.tv_sec * Converter + total_start.tv_usec);
    
    // std::cout << "GPU batch search finished. Total wall-clock time: " << total_latency_us << " us." << std::endl;

    // // --- 计算召回率 ---
    // float total_recall = 0.0f;
    // for(size_t i = 0; i < test_number; ++i) {
    //     auto res = all_gpu_results[i];
    //     std::set<uint32_t> gtset;
    //     for(size_t j = 0; j < k; ++j){
    //         int t = test_gt[j + i*test_gt_d];
    //         gtset.insert(t);
    //     }
    //     size_t acc = 0;
    //     while (!res.empty()) {   
    //         int x = res.top().second;
    //         if(gtset.count(x)){
    //             ++acc;
    //         }
    //         res.pop();
    //     }
    //     total_recall += (float)acc/k;
    // }


    // //--- 调用 V2 GPU 批处理查询 ---
    // std::cout << "Starting GPU batch search (v2) for " << test_number << " queries..." << std::endl;
    // const unsigned long Converter = 1000 * 1000;
    // struct timeval total_start, total_end;
    // gettimeofday(&total_start, NULL);

    // // 调用我们新的 V2 版本函数
    // auto final_results_flat = gpu_flat_search_v2(base, test_query, base_number, test_number, vecdim, k);

    // gettimeofday(&total_end, NULL);
    // int64_t total_latency_us = (total_end.tv_sec * Converter + total_end.tv_usec) - (total_start.tv_sec * Converter + total_start.tv_usec);
    
    // std::cout << "GPU batch search (v2) finished. Total wall-clock time: " << total_latency_us << " us." << std::endl;

    // // --- 计算召回率 (逻辑需要调整) ---
    // float total_recall = 0.0f;
    // for(size_t i = 0; i < test_number; ++i) {
    //     std::set<uint32_t> gtset;
    //     for(size_t j = 0; j < k; ++j){
    //         int t = test_gt[j + i * test_gt_d];
    //         gtset.insert(t);
    //     }

    //     size_t acc = 0;
    //     // 遍历返回的扁平化结果数组中对应于当前查询的部分
    //     for (size_t j = 0; j < k; ++j) {
    //         int found_id = final_results_flat[i * k + j].id;
    //         if (gtset.count(found_id)) {
    //             ++acc;
    //         }
    //     }
    //     total_recall += (float)acc / k;
    // }


    // --- 调用 V3 GPU 批处理查询 ---
    std::cout << "Starting GPU batch search (v3) for " << test_number << " queries..." << std::endl;
    const unsigned long Converter = 1000 * 1000;
    struct timeval total_start, total_end;
    gettimeofday(&total_start, NULL);

    //调用我们新的 V3 版本函数
    //auto final_results_flat = gpu_flat_search_v3(base, test_query, base_number, test_number, vecdim, k);

    // const int K_LOCAL_PARAM = 1; // 在这里设置你的参数
    // auto final_results_flat = gpu_flat_search_v4<K_LOCAL_PARAM>(base, test_query, base_number, test_number, vecdim, k);

    // const int K_LOCAL_PARAM = 1;    // 每个线程贡献几个候选？
    // const int SAMPLE_RATIO_PARAM = 2; // 每几个点采样一次？值为2代表只看一半数据

    // auto final_results_flat = gpu_flat_search_v5<K_LOCAL_PARAM, SAMPLE_RATIO_PARAM>(base, test_query, base_number, test_number, vecdim, k);

    // const int K_LOCAL_PARAM = 1;
    // const int SAMPLE_NUMERATOR_PARAM = 11;   // 分子
    // const int SAMPLE_DENOMINATOR_PARAM = 12; // 分母

    // //auto final_results_flat = gpu_flat_search_v6<K_LOCAL_PARAM, SAMPLE_NUMERATOR_PARAM, SAMPLE_DENOMINATOR_PARAM>(base, test_query, base_number, test_number, vecdim, k);
    // // 调用我们新的 V7 版本函数
    // auto final_results_flat = gpu_flat_search_final(base, test_query, base_number, test_number, vecdim, k);

    // const int K_LOCAL_PARAM = 1;
    // const int SAMPLE_NUMERATOR_PARAM = 6;   // 分子 (e.g., 11)
    // const int SAMPLE_DENOMINATOR_PARAM = 7; // 分母 (e.g., 12)

    // // 调用终极版函数
    // auto final_results_flat = gpu_flat_search_v8<K_LOCAL_PARAM, SAMPLE_NUMERATOR_PARAM, SAMPLE_DENOMINATOR_PARAM>(base, test_query, base_number, test_number, vecdim, k);
    const int K_LOCAL_PARAM = 1;
    const int SAMPLE_NUMERATOR_PARAM = 11;   // 分子 (e.g., 11)
    const int SAMPLE_DENOMINATOR_PARAM = 12; // 分母 (e.g., 12)

    // 调用终极版函数
    auto final_results_flat = gpu_flat_search_v9<K_LOCAL_PARAM, SAMPLE_NUMERATOR_PARAM, SAMPLE_DENOMINATOR_PARAM>(base, test_query, base_number, test_number, vecdim, k);
    
    gettimeofday(&total_end, NULL);
    int64_t total_latency_us = (total_end.tv_sec * Converter + total_end.tv_usec) - (total_start.tv_sec * Converter + total_start.tv_usec);

    std::cout << "GPU batch search (v3) finished. Total wall-clock time: " << total_latency_us << " us." << std::endl;

    // --- 计算召回率 (逻辑与V2版完全相同) ---
    float total_recall = 0.0f;
    for(size_t i = 0; i < test_number; ++i) {
        std::set<uint32_t> gtset;
        for(size_t j = 0; j < k; ++j){
            int t = test_gt[j + i * test_gt_d];
            gtset.insert(t);
        }

        size_t acc = 0;
        for (size_t j = 0; j < k; ++j) {
            int found_id = final_results_flat[i * k + j].id;
            if (gtset.count(found_id)) {
                ++acc;
            }
        }
        total_recall += (float)acc / k;
    }

    // --- 最终结果输出 ---
    std::cout << "average recall: "<< total_recall / test_number <<"\n";
    std::cout << "average latency (us): "<< (double)total_latency_us / test_number <<"\n";

    delete[] test_query;
    delete[] test_gt;
    delete[] base;
    
    return 0;
}