#ifndef __UI_H__
#define __UI_H__
#include "model.h"
class UI{
  public:
    static Model* model;
    static bool showFPS;
    static int targetMS;
};
void init(int argc, char* argv[],float cellSize);
void loop();
void display();
void keyboard(unsigned char key,int x,int y);
#endif // __UI_H__