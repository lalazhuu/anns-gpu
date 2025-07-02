#ifndef IVF_PQ_COMMON_H
#define IVF_PQ_COMMON_H

#include <vector> //
#include <queue>  // 这些也属于公共定义
#include <string> //
#include <map>    //
#include <set>    //

// --- 共享的数据结构 ---
struct ResultPair {
    float distance;
    int id;
};

// --- 共享的PQ参数 ---
const int PQ_M = 16;
const int PQ_KS = 256;
const int PQ_DS = 96 / PQ_M;

#endif // IVF_PQ_COMMON_H