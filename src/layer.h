#ifndef __LAYER_H__
#define __LAYER_H__
#include<CL/sycl.hpp>
enum ActivationFunction{
    SoftMax
};
class Layer{
public:
    int in_size, out_size;
    int elements_count;
    float* weights;
    sycl::queue* queue;
    ActivationFunction activation;
    bool initialized=false;


    Layer(int elements_count, int in_size, int out_size, ActivationFunction activation, sycl::queue* queue, bool equality=false);
    Layer();
    SYCL_EXTERNAL float& at(int element,int i, int j) const;
    SYCL_EXTERNAL void calc(int element, float* input, float* output) const;
    void deinit();
};

#endif // __LAYER_H__