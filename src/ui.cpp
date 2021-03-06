#include <GL/glut.h>
#include "ui.h"
#include<chrono>
#include<iostream>
Model* UI::model = nullptr;
std::chrono::steady_clock::time_point timeFPS = std::chrono::steady_clock::now();
void STEP(int v){
    auto t = timeFPS;
    timeFPS = std::chrono::steady_clock::now();
    std::cout<<std::chrono::duration_cast<std::chrono::milliseconds>(timeFPS - t).count() << " ms" << std::endl;
    UI::model->step(1);
    display();
    glutTimerFunc(1, STEP,1);
}
void init(int argc, char* argv[], float cellSize){
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE);
    glutInitWindowSize(UI::model->w*cellSize, UI::model->h*cellSize);
    glutInitWindowPosition(100, 100);
    glutCreateWindow("Hello world!");
    gluOrtho2D(0,UI::model->w,0,UI::model->h);
    glutDisplayFunc(display);
    glutTimerFunc(1, STEP,1);
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
    float d = 0.9;
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
