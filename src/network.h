#ifndef __NETWORK_H__
#define __NETWORK_H__
#include"layer.h"
#include<CL/sycl.hpp>
#include <cstdarg>

struct LayerDefinition{
  int neurons_count;
  ActivationFunction activation;
};
class Network{
  public:
    float* weights;
    int elements_count;
    int layers_count=0;
    sycl::queue* queue;
    unsigned long int* seeds;
    static const int MaxLayers=10;
    Layer layers[MaxLayers];
    Network(sycl::queue* queue,int elements_count, int input_size, std::vector<LayerDefinition> layers_definitions, bool equality = false);
    Network();
    void deinit();
    SYCL_EXTERNAL void mutate(int element,float coef) const;
    SYCL_EXTERNAL void randomize(int element) const;
    SYCL_EXTERNAL void copy(int from, int to) const;
    void mutate(float coef) const;
    SYCL_EXTERNAL void calc(int element, float** layer) const;
};
#endif // __NETWORK_H__