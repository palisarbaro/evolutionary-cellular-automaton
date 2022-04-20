#include <GL/glut.h>
#include "ui.h"
#include<chrono>
#include<iostream>
#include<cctype>
Model* UI::model = nullptr;
bool UI::showFPS = false;
int UI::targetMS = 100;
std::chrono::steady_clock::time_point timeFPS = std::chrono::steady_clock::now();
void STEP(int v){
    auto t = timeFPS;
    timeFPS = std::chrono::steady_clock::now();
    if(UI::showFPS){
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(timeFPS - t).count();
        std::cout<< ms << " ms; "<<1000./ms<<" FPS" << std::endl;
    }
    UI::model->step(1);
    display();
    glutTimerFunc(UI::targetMS, STEP,1);
}
void init(int argc, char* argv[], float cellSize){
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE);
    glutInitWindowSize(UI::model->w*cellSize, UI::model->h*cellSize);
    glutInitWindowPosition(100, 100);
    glutCreateWindow("Hello world!");
    gluOrtho2D(0,UI::model->w,0,UI::model->h);
    glutDisplayFunc(display);
    glutKeyboardFunc(keyboard);
    glutTimerFunc(1000, STEP,1);
}

void loop()
{
    glutMainLoop();
}
void setColor(elType val){
    switch (val)
    {
    case 0:
        glColor3f(0,0,0);
        break;
    case 1:
        glColor3f(0,1,0);
        break;  
    default:
        throw -111;
    }
}
void display()
{
    float d = 1;
	glClear(GL_COLOR_BUFFER_BIT);
    for(int x=0;x<UI::model->w;x++){
        for(int y=0;y<UI::model->h;y++){
            setColor(UI::model->curr[x+UI::model->w*y]);
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
        display();
    }
    if(key>='0' && key<='9'){ //fps
        int mss[10] = {0,20,50,100,200,300,500,1000,2000,3000};
        int d = key-'0';
        UI::targetMS = mss[d];
        std::cout<<"target fps: "<< mss[d] << " ms; "<<1000./mss[d]<<" FPS" << std::endl;

    }
}
