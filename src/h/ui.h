#ifndef __UI_H__
#define __UI_H__
#include "model.hpp"
class UI{
  public:
    static const int dim = 20;
    static Model<dim>* model;
    static bool showFPS;
    static bool stopped;
    static int targetMS;
    static bool showBots;
    static bool showAutomate;
};
void init(int argc, char* argv[],float cellSize);
void loop();
void display();
void keyboard(unsigned char key,int x,int y);
#endif // __UI_H__