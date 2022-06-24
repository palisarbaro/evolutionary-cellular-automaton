
#include "h/network.h"
Network::Network()
{
    
}

Network::Network(sycl::queue* queue,int elements_count, int input_size, std::vector<LayerDefinition> layers_definitions, bool equality):queue(queue),elements_count(elements_count)
{
    layers_count = 0;
    seeds = sycl::malloc_shared<int>(elements_count, *queue);
    auto v = rand();
    for(int i=0;i<elements_count;i++){
        if(!equality || true){
            v = rand();
        }
        seeds[i] = v;
    }
    for(auto ld: layers_definitions){
        layers[layers_count] = Layer(elements_count,input_size,ld.neurons_count,ld.activation,queue,equality);
        layers_count++;
        input_size = ld.neurons_count;
    }
}

void Network::deinit()
{
    sycl::free(seeds,*queue);
    for(int i=0;i<MaxLayers;i++){
        layers[i].deinit();
    }
}

void Network::mutate(int element,float coef) const
{
    for(int layer=0;layer<layers_count;layer++){
        for(int i=0;i<layers[layer].in_size+1;i++){
            for(int j=0;j<layers[layer].out_size;j++){
                float rnd1 = getRandFloat(element);
                rnd1*=2;
                rnd1-=1;
                float rnd2 = getRandFloat(element);
                rnd2*=2;
                rnd2-=1;
                float mult = 1+coef*rnd1;
                float add = coef*rnd2;
                layers[layer].at(element,i,j)*=mult;
                layers[layer].at(element,i,j)+=add;
            }
        }
    }
}

void Network::mutate(float coef) const
{
    int elements_count = this->elements_count;
    Network net = *this;
    sycl::event e = queue->parallel_for(sycl::range<1>(elements_count), [=](sycl::item<1> item) {
        int element = item.get_linear_id();
        net.mutate(element,coef);
    });
    e.wait_and_throw();
}

void Network::randomize(int element) const
{
    for(int layer=0;layer<layers_count;layer++){
        for(int i=0;i<layers[layer].in_size+1;i++){
            for(int j=0;j<layers[layer].out_size;j++){
                float rnd1 = getRandFloat(element);
                rnd1*=4;
                rnd1-=2;
                layers[layer].at(element,i,j)=rnd1;
            }
        }
    }
}

void Network::randomize() const
{
    int elements_count = this->elements_count;
    Network net = *this;
    sycl::event e = queue->parallel_for(sycl::range<1>(elements_count), [=](sycl::item<1> item) {
        int element = item.get_linear_id();
        net.randomize(element);
    });
    e.wait_and_throw();
}

SYCL_EXTERNAL void Network::copy(int from, int to) const
{
     for(int layer=0;layer<layers_count;layer++){
        for(int i=0;i<layers[layer].in_size+1;i++){
            for(int j=0;j<layers[layer].out_size;j++){
                layers[layer].at(to,i,j) = layers[layer].at(from,i,j);
            }
        }
     }
}

void Network::calc(int element, float** layers_data) const
{
    for(int i=0;i<layers_count;i++){
        layers[i].calc(element,layers_data[i],layers_data[i+1]);
    }
}

SYCL_EXTERNAL int Network::getRand(int element) const
{
    seeds[element] = seeds[element] * 1103515245 + 12345;
    return seeds[element];
}

SYCL_EXTERNAL float Network::getRandFloat(int element) const
{
    return ((unsigned int)(getRand(element) / 65536) % 32768)/32767.;
}

void Network::printStatistics()
{
    for(int i=0;i<layers_count;i++){
        std::cout<<"layer#"<<i<<std::endl;
        layers[i].printStatistic();
    }
}
