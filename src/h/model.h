#ifndef __MODEL_H__
#define __MODEL_H__
#include<vector>
#include<CL/sycl.hpp>
#include"network.h"
#include<random>
#include <ctime>
#include "vec.hpp"


struct Fields
{
    int h, w;
    static const int dim = 20;
    typedef vec<dim> elType;
    int time = 0;
    elType* curr;
    elType* next;
    Network automate_network;
    
};
typedef Fields::elType elType;


class Model{
   public:
    sycl::queue queue;
    std::mt19937 rand;
    Fields f;
    Model(int h, int w);
    ~Model();
    void step();
    void automateStep();
    void reset();
};
#endif // __MODEL_H__