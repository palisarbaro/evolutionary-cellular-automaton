#include "layer.h"
Layer::Layer(int elements_count, int in_size, int out_size, ActivationFunction activation, sycl::queue* queue):elements_count(elements_count),in_size(in_size),out_size(out_size),activation(activation),queue(queue)
{
    initialized=true;
    weights = sycl::malloc_shared<float>(elements_count*in_size*out_size, *queue);
    
    for(int i=0;i<in_size;i++){
        for(int j=0;j<out_size;j++){
            
            for(int x=0;x<elements_count;x++){
                float v = rand()/static_cast<float>(RAND_MAX)*4-2;
                at(x,i,j)=v;
            }
        }
    }
}

Layer::Layer()
{
    
}

inline float& Layer::at(int element,int i, int j) const
{
    return weights[element+elements_count*(i+in_size*j)];
}


void softmax(float* input, int size) {
	int i;
	double m, sum, constant;

	m = -INFINITY;
	for (i = 0; i < size; ++i) {
		if (m < input[i]) {
			m = input[i];
		}
	}

	sum = 0.0;
	for (i = 0; i < size; ++i) {
		sum += sycl::exp(input[i] - m);
	}

	constant = m + sycl::log(sum);
	for (i = 0; i < size; ++i) {
		input[i] = sycl::exp(input[i] - constant);
	}
}

void Layer::calc(int element, float* input, float* output) const
{
    for(int j=0;j<out_size;j++){
        float res = 0;
        for(int i=0;i<in_size;i++){
            res += input[i]*at(element,i,j);
        }
        output[j] = res; 
    }
    switch (activation)
    {
    case ActivationFunction::SoftMax:
        softmax(output,out_size);
        break;
    
    default:
        break;
    }
}

void Layer::deinit()
{
    if(initialized){
        sycl::free(weights,*queue);
    }   
}
