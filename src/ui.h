#ifndef __UI_H__
#define __UI_H__
#include "model.h"
class UI{
  public:
    static Model* model;
};
void init(int argc, char* argv[],float cellSize);
void loop();
void display();
#endif // __UI_H__