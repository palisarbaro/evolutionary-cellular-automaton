#include <GL/glut.h>
#include "h/ui.h"
#include<chrono>
#include<iostream>
#include<cctype>
Model* UI::model = nullptr;
bool UI::showFPS = false;
bool UI::stopped = false;
int UI::targetMS = 100;
bool UI::showBots = true;
bool UI::showAutomate = true;
std::chrono::steady_clock::time_point timeFPS = std::chrono::steady_clock::now();
void STEP(int v){
    if(UI::stopped){
        return;
    }

    auto t = timeFPS;
    timeFPS = std::chrono::steady_clock::now();
    if(UI::showFPS){
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(timeFPS - t).count();
        std::cout<< ms << " ms; "<<1000./ms<<" FPS" << std::endl;
    }
    UI::model->step();
    display();
    glutTimerFunc(UI::targetMS, STEP,1);
}
void init(int argc, char* argv[], float cellSize){
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE);
    glutInitWindowSize(UI::model->f.w*cellSize, UI::model->f.h*cellSize);
    glutInitWindowPosition(100, 100);
    glutCreateWindow("Hello world!");
    gluOrtho2D(0,UI::model->f.w,0,UI::model->f.h);
    glutDisplayFunc(display);
    glutKeyboardFunc(keyboard);
    glutTimerFunc(1000, STEP,1);
}

void loop()
{
    glutMainLoop();
}
void setColor(elType val){
    glColor3f(val.arr[0],val.arr[1],val.arr[2]);
}
void display()
{
    float d = 1;
	glClear(GL_COLOR_BUFFER_BIT);
    for(int x=0;x<UI::model->f.w;x++){
        for(int y=0;y<UI::model->f.h;y++){
            int element = x+UI::model->f.w*y;
            glColor3f(0,0,0);
            setColor(UI::model->f.curr[element]);
            glBegin(GL_QUADS);
                glVertex2f(x,y);
                glVertex2f(x+d,y);
                glVertex2f(x+d,y+d);
                glVertex2f(x,y+d);
            glEnd();
        }
    }
    glFlush();
    glutSwapBuffers();
}

void keyboard(unsigned char key,int x,int y)
{
    key = tolower(key);
    if(key=='f'){
        UI::showFPS = !UI::showFPS;
    }
    if(key=='r'){
        UI::model->reset();
    }
    if(key==' '){
        UI::stopped=!UI::stopped;
        if(!UI::stopped){
            glutTimerFunc(UI::targetMS, STEP,1);
        }
    }

    if(key=='m'){
        UI::model->f.automate_network.mutate(0.05);
        if(glutGetModifiers()&GLUT_ACTIVE_SHIFT){
            UI::model->f.automate_network.randomize();
        }
    }

    if(key=='s'){
        UI::model->f.automate_network.printStatistics();
    }

    if(key=='b'){
        UI::showBots = !UI::showBots;
    }
    if(key=='c'){
        UI::showAutomate = !UI::showAutomate;
    }
    if(key>='0' && key<='9'){ //fps
        int mss[10] = {0,20,50,100,200,300,500,1000,2000,3000};
        int d = key-'0';
        UI::targetMS = mss[d];
        std::cout<<"target fps: "<< mss[d] << " ms; "<<1000./mss[d]<<" FPS" << std::endl;

    }
    display();
}
