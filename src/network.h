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
    static const int MaxLayers=10;
    Layer layers[MaxLayers];
    Network(sycl::queue* queue,int elements_count, int input_size, std::vector<LayerDefinition> layers_definitions);
    Network();
    void deinit();
    SYCL_EXTERNAL void calc(int element, float** layer) const;
};
#endif // __NETWORK_H__