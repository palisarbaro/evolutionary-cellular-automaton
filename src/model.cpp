#include "h/model.h"
#include <algorithm>
const int view_radius = 1;
const int visible_cells_count = (view_radius*2+1)*(view_radius*2+1);
Model::Model(int h, int w)
{
    rand.seed(std::time(nullptr));

    f.h = h;
    f.w = w;
    auto device = sycl::platform::get_platforms()[0].get_devices()[0];
    std::cout<< device.get_info<sycl::info::device::name>()<<std::endl;
    queue = sycl::queue(device);
    f.curr = sycl::malloc_shared<elType>(h*w, queue);
    f.next = sycl::malloc_shared<elType>(h*w, queue);
    f.automate_network = Network(&queue,2,visible_cells_count*Fields::dim,std::vector<LayerDefinition>({{Fields::dim,Tanh01}}),false);

    reset();
}

Model::~Model()
{
    sycl::free(f.curr,queue);
    sycl::free(f.next,queue);
    f.automate_network.deinit();
}

void Model::step()
{
    automateStep();

    f.time++;
}

void collectData(int x,int y, int view_radius, float* to, elType* from, const Fields& f){
    for(int _x=-view_radius;_x<view_radius+1;_x++){
            for(int _y=-view_radius;_y<view_radius+1;_y++){
                int X = (x+_x+f.w)%f.w;
                int Y = (y+_y+f.h)%f.h;
                for(int i=0;i<Fields::dim;i++){
                    to[i+Fields::dim*(_x+view_radius+(view_radius*2+1)*(_y+view_radius))] = from[X+f.w*Y][i];
                }
            }
        }
}

void Model::automateStep()
{
    Fields& f = this->f;
    sycl::event e = queue.parallel_for(sycl::range<2>(f.h,f.w), [=](sycl::item<2> item) {
        int x = item.get_id(1);
        int y = item.get_id(0);
        int element = item.get_linear_id();
        float input[visible_cells_count*Fields::dim];
        float middle2[50];
        float output[Fields::dim];
        collectData(x,y,view_radius,input,f.curr,f);
        float* layers_data[2] = {input, output};
        f.automate_network.calc(0,layers_data);
        f.next[x+f.w*y]=elType(output);//>0.5?0:1;
    });
    e.wait_and_throw();

    elType* tmp = f.curr;
    f.curr = f.next;
    f.next = tmp;
}



void Model::reset()
{
    for(int x=0;x<f.w;x++){
        for(int y=0;y<f.h;y++){
            float  r[Fields::dim];
            for(int i=0;i<Fields::dim;i++){
                r[i] = rand()/static_cast<float>(RAND_MAX);
            }
            f.curr[x+f.w*y] = elType(r);
        }
    }
}