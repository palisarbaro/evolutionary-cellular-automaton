#ifndef __MODEL_H__
#define __MODEL_H__
#include<vector>
#include<CL/sycl.hpp>
#include"network.h"
typedef int elType;
class Model{
   public:
    int h, w;
    int typeNum;
    elType* curr;
    elType* next;
    Network network;
    sycl::queue queue;
    Model(int h, int w, int typeNum);
    ~Model();
    void step(int iterations);
    void reset();
};
#endif // __MODEL_H__