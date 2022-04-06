#ifndef __MODEL_H__
#define __MODEL_H__
#include<vector>
#include<CL/sycl.hpp>
typedef int elType;
class Model{
   public:
    int h, w;
    elType* curr;
    elType* next;
    sycl::queue queue;
    Model(int h, int w);
    ~Model();
    void step(int iterations);
};
#endif // __MODEL_H__