#include "model.h"
#include <algorithm>
const int view_radius = 3;
const int visible_cells_count = (view_radius*2+1)*(view_radius*2+1);
const int actions_count = 10;
Model::Model(int h, int w, int typeNum)
{
    rand.seed(std::time(nullptr));

    f.h = h;
    f.w = w;
    f.typeNum = typeNum;
    auto device = sycl::platform::get_platforms()[0].get_devices()[0];
    std::cout<< device.get_info<sycl::info::device::name>()<<std::endl;
    queue = sycl::queue(device);
    f.curr = sycl::malloc_shared<elType>(h*w, queue);
    f.next = sycl::malloc_shared<elType>(h*w, queue);
    f.curr_bots = sycl::malloc_shared<Bot>(h*w, queue);
    f.next_bots = sycl::malloc_shared<Bot>(h*w, queue);
    for(int i=0;i<w*h;i++){
        f.curr_bots[i] = {-1,-1};
        f.next_bots[i] = {-2,-2};
    }
    f.history = sycl::malloc_shared<uint64_t>(h*w, queue);
    f.automate_network = Network(&queue,w*h,visible_cells_count,std::vector<LayerDefinition>({{18,SoftMax},{2,SoftMax}}),false);
    f.spawn_network = Network(&queue,w*h,visible_cells_count,std::vector<LayerDefinition>({{18,SoftMax},{2,SoftMax}}),false);
    f.action_network = Network(&queue,w*h,visible_cells_count,std::vector<LayerDefinition>({{18,SoftMax},{actions_count,SoftMax}}),false);
    reset();
}

Model::~Model()
{
    sycl::free(f.curr,queue);
    sycl::free(f.next,queue);
    sycl::free(f.curr_bots,queue);
    sycl::free(f.next_bots,queue);
    sycl::free(f.history,queue);
    f.automate_network.deinit();
    f.spawn_network.deinit();
}

void Model::step()
{
    automateStep();
    spawnBots();
    botStep();
    f.time++;
}

void collectData(int x,int y, int view_radius, float* to, elType* from, const Fields& f){
    for(int _x=-view_radius;_x<view_radius+1;_x++){
            for(int _y=-view_radius;_y<view_radius+1;_y++){
                int X = (x+_x+f.w)%f.w;
                int Y = (y+_y+f.h)%f.h;
                to[_x+view_radius+(view_radius*2+1)*(_y+view_radius)] = from[X+f.w*Y];
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
        float input[visible_cells_count];
        float middle2[18];
        float output[2];
        collectData(x,y,view_radius,input,f.curr,f);
        float* layers_data[3] = {input, middle2, output};
        f.automate_network.calc(element,layers_data);
        int res = output[0]>0.5?1:0;
        f.next[x+f.w*y]=res;
    });
    e.wait_and_throw();

    elType* tmp = f.curr;
    f.curr = f.next;
    f.next = tmp;
}

void Model::botStep()
{
    Fields& f = this->f;
    sycl::event e2 = queue.parallel_for(sycl::range<2>(f.h,f.w), [=](sycl::item<2> item) {
        int x = item.get_id(1);
        int y = item.get_id(0);
        int element = item.get_linear_id();
        f.next_bots[element]={-1,-1};
    });
    e2.wait_and_throw();
    sycl::event e = queue.parallel_for(sycl::range<2>(f.h,f.w), [=](sycl::item<2> item) {
        int x = item.get_id(1);
        int y = item.get_id(0);
        int element = item.get_linear_id();
        if(f.curr_bots[element].x==-1){
            return;
        }
        float input[visible_cells_count];
        collectData(x,y,view_radius,input,f.curr,f);
        float middle2[18];
        float output[actions_count];
        float* layers_data[3] = {input, middle2, output};
        f.action_network.calc(element,layers_data);
        int res = std::max_element(output,output+actions_count)-output;
        if(res<9){ // move
            int _x = res%3 -1;
            int _y = res/3 - 1;
            int X = (x+_x+f.w)%f.w;
            int Y = (y+_y+f.h)%f.h;
            f.next_bots[X+f.w*Y] = f.curr_bots[element];
        }
        if(res==9){ // infect
            //f.next_bots[element] = f.curr_bots[element];
            float mutate_coef = 0.01;
            int elementfrom = f.curr_bots[element].x +f.w * f.curr_bots[element].y;
            f.action_network.copy(elementfrom,element);
            f.action_network.mutate(element,mutate_coef);

            f.spawn_network.copy(elementfrom,element);
            f.spawn_network.mutate(element,mutate_coef);

            f.automate_network.copy(elementfrom,element);
            f.automate_network.mutate(element,mutate_coef);
        }

    });
    e.wait_and_throw();
    
    auto tmp = f.curr_bots;
    f.curr_bots = f.next_bots;
    f.next_bots = tmp;
}

void Model::spawnBots()
{
    for(int i=0;i<5;i++){
        int x = rand()%f.w;
        int y = rand()%f.h;
        f.curr_bots[x+f.w*y] = {x,y};
    }
}

void Model::reset()
{
    for(int x=0;x<f.w;x++){
        for(int y=0;y<f.h;y++){
            f.curr[x+f.w*y] = rand()/static_cast<float>(RAND_MAX)>0.9?1:0;
        }
    }
}

void Model::averageWeights()
{
    Fields& f = this->f;
    sycl::event e = queue.parallel_for(sycl::range<2>(f.h,f.w), [=](sycl::item<2> item) {
        int x = item.get_id(1);
        int y = item.get_id(0);
        int element = item.get_linear_id();
        for(int layer=0;layer<f.automate_network.layers_count;layer++){
            for(int i=0;i<f.automate_network.layers[layer].in_size;i++){
                for(int j=0;j<f.automate_network.layers[layer].out_size;j++){
                    float av = 0;
                    for(int _x=-1;_x<2;_x++){
                        for(int _y=-1;_y<2;_y++){
                            int X = (x+_x+f.w)%f.w;
                            int Y = (y+_y+f.h)%f.h;
                            int EL = X+f.w*Y;
                            av += f.automate_network.layers[layer].at(EL,i,j);
                        }
                    }
                    av/=9;
                    f.automate_network.layers[layer].at(element,i,j) = av;
                }
            }
        }
    
    });
    e.wait_and_throw();
}
