#ifndef __MODEL_H__
#define __MODEL_H__
#include<vector>
#include<CL/sycl.hpp>
#include"network.h"
#include<random>
#include <ctime>
#include<cstring>
#include "vec.hpp"
#include"img.hpp"
#include <algorithm>
#include <fstream>


const int view_radius = 1;
const int visible_cells_count = (view_radius*2+1)*(view_radius*2+1);


template <int dim>
struct Fields
{
    int h, w;
    int time = 0;
    Image<dim> curr;
    Image<dim> next;
    Network automate_network;
    Fields(){};
    
};

template <int dim>
void collectData(int x,int y, int view_radius, float* to, const Image<dim>& from, const Fields<dim>& f){
    for(int _x=-view_radius;_x<view_radius+1;_x++){
        for(int _y=-view_radius;_y<view_radius+1;_y++){
            int X = (x+_x+f.w)%f.w;
            int Y = (y+_y+f.h)%f.h;
            for(int i=0;i<dim;i++){
                to[i+dim*(_x+view_radius+(view_radius*2+1)*(_y+view_radius))] = from[X+f.w*Y][i];
            }
        }
    }
}

template <int dim>
class Model{
   public:
    sycl::queue queue;
    std::mt19937 rand;
    Fields<dim> f;

    Model(int h, int w)
    {
        rand.seed(std::time(nullptr));
        f.h = h;
        f.w = w;
        auto device = sycl::platform::get_platforms()[0].get_devices()[0];
        std::cout<< device.get_info<sycl::info::device::name>()<<std::endl;
        queue = sycl::queue(device);
        f.curr = Image<dim>(h,w,&queue);
        f.next = Image<dim>(h,w,&queue);
        f.automate_network = Network(&queue,2,visible_cells_count*dim,std::vector<LayerDefinition>({{dim,Tanh01}}),false);
        f.curr.reset();
    }
    ~Model()
    {
        f.automate_network.deinit();
        f.curr.deinit();
        f.next.deinit();
    }

    void step()
    {
        automateStep();
        f.time++;
    }
    void automateStep()
    {
        Fields<dim>& f = this->f;
        sycl::event e = queue.parallel_for(sycl::range<2>(f.h,f.w), [=](sycl::item<2> item) {
            int x = item.get_id(1);
            int y = item.get_id(0);
            int element = item.get_linear_id();
            float input[visible_cells_count*dim];
            float middle2[50];
            float output[dim];
            collectData<dim>(x,y,view_radius,input,f.curr,f);
            float* layers_data[2] = {input, output};
            f.automate_network.calc(0,layers_data);
            f.next[x+f.w*y]=vec<dim>(output);//>0.5?0:1;
        });
        e.wait_and_throw();

        Image tmp(std::move(f.curr));
        f.curr = std::move(f.next);
        f.next = std::move(tmp);
    }
};
#endif // __MODEL_H__