#include "model.h"

const int view_radius = 1;
const int visible_cells_count = (view_radius*2+1)*(view_radius*2+1);
Model::Model(int h, int w, int typeNum):h(h),w(w), typeNum(typeNum)
{
    auto device = sycl::platform::get_platforms()[0].get_devices()[0];
    queue = sycl::queue(device);
    curr = sycl::malloc_shared<elType>(h*w, queue);
    next = sycl::malloc_shared<elType>(h*w, queue);
    network = Network(&queue,w*h,visible_cells_count,std::vector<LayerDefinition>({{49,SoftMax},{18,SoftMax},{2,SoftMax}}));
    reset();
}

Model::~Model()
{
    sycl::free(curr,queue);
    sycl::free(next,queue);
    network.deinit();
}

void Model::step(int iterations)
{
    auto curr = this->curr;
    auto next = this->next;
    int w = this->w;
    int h = this->h;
    Network network = this->network;
    for (int iteration = 0; iteration < iterations; iteration++) {
		sycl::event e = queue.parallel_for(sycl::range<2>(h,w), [=](sycl::item<2> item) {
            int x = item.get_id(1);
            int y = item.get_id(0);
            int element = item.get_linear_id();
            float input[visible_cells_count];
            float middle1[49];
            float middle2[18];
            float output[2];
            for(int _x=-view_radius;_x<view_radius+1;_x++){
                for(int _y=-view_radius;_y<view_radius+1;_y++){
                    int X = (x+_x+w)%w;
                    int Y = (y+_y+h)%h;
                    input[_x+view_radius+(view_radius*2+1)*(_y+view_radius)] = curr[X+w*Y];
                }
            }
            float* layers_data[4] = {input,middle1, middle2, output};
            network.calc(element,layers_data);
			next[x+w*y]=output[0]>0.5?1:0;
		});
		e.wait_and_throw();

		elType* tmp = curr;
		curr = next;
		next = tmp;

	}
    this->curr = curr;
    this->next = next;
    
}

void Model::reset()
{
    for(int x=0;x<w;x++){
        for(int y=0;y<h;y++){
            curr[x+w*y] = rand()/static_cast<float>(RAND_MAX)>0.9?1:0;
        }
    }
}
