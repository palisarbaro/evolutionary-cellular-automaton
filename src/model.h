#ifndef __MODEL_H__
#define __MODEL_H__
#include<vector>
#include<CL/sycl.hpp>
#include"network.h"
#include<random>
#include <ctime>
typedef int elType;
struct Bot{
    int x;
    int y;
};
struct Fields
{
    int h, w;
    int typeNum;
    int time = 0;
    elType* curr;
    elType* next;
    Bot* curr_bots;
    Bot* next_bots;
    uint64_t* history;
    Network automate_network;
    Network spawn_network;
    Network action_network;
};

class Model{
   public:
    sycl::queue queue;
    std::mt19937 rand;
    Fields f;
    Model(int h, int w, int typeNum);
    ~Model();
    void step();
    void automateStep();
    void botStep();
    void spawnBots();
    void reset();
    void averageWeights();
};
#endif // __MODEL_H__