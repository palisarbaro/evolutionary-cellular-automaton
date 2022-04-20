
#include "network.h"


Network::Network()
{
    
}

Network::Network(sycl::queue* queue,int elements_count, int input_size, std::vector<LayerDefinition> layers_definitions):elements_count(elements_count)
{
    layers_count = 0;
    for(auto ld: layers_definitions){
        layers[layers_count] = Layer(elements_count,input_size,ld.neurons_count,ld.activation,queue);
        layers_count++;
        input_size = ld.neurons_count;
    }
}

void Network::deinit()
{
    for(int i=0;i<MaxLayers;i++){
        layers[i].deinit();
    }
}

void Network::calc(int element, float** layers_data) const
{
    for(int i=0;i<layers_count;i++){
        layers[i].calc(element,layers_data[i],layers_data[i+1]);
    }
}
