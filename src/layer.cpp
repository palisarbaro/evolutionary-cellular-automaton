#include "h/layer.h"
Layer::Layer(int elements_count, int in_size, int out_size, ActivationFunction activation, sycl::queue* queue, bool equality):elements_count(elements_count),in_size(in_size),out_size(out_size),activation(activation),queue(queue)
{
    initialized=true;
    weights = sycl::malloc_shared<float>(elements_count*(in_size+1)*out_size, *queue);
    
    for(int i=0;i<in_size+1;i++){
        for(int j=0;j<out_size;j++){
            float v = rand()/static_cast<float>(RAND_MAX)*4-2;
            for(int x=0;x<elements_count;x++){
                if(!equality){
                    v = rand()/static_cast<float>(RAND_MAX)*4-2;
                }
                at(x,i,j)=v;
            }
        }
    }
}

Layer::Layer()
{
    
}

float& Layer::at(int element,int i, int j) const
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
        output[j] = res+at(element,in_size,j); 
    }
    switch (activation)
    {
    case ActivationFunction::SoftMax:
        softmax(output,out_size);
        break;

    case ActivationFunction::Tanh01:
        for(int i=0;i<out_size;i++){
            output[i] = (sycl::tanh(output[i])+1)/2;
        }
    
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

void Layer::printStatistic()
{
    float mean=0;
    float min=MAXFLOAT;
    float max=-MAXFLOAT;
    float m2=0;
    int len = (in_size+1)*out_size;
    for(int w=0;w<len;w++){
        mean+=weights[w];
        m2+=weights[w]*weights[w];
        if(min>weights[w]) min = weights[w];
        if(max<weights[w]) max = weights[w];
    }
    mean/=len;
    m2/=len;
    float D = m2-mean*mean;
    std::cout<<"mean: "<<mean<<std::endl;
    std::cout<<"min: "<<min<<std::endl;
    std::cout<<"max: "<<max<<std::endl;
    std::cout<<"D: "<<D<<std::endl;
}
