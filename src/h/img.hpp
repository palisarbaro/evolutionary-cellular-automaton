#ifndef __IMG_H__
#define __IMG_H__
#include"vec.hpp"
template <int dim>
struct Image{
    int h, w;
    bool initialized = false;
    sycl::queue* queue;
    vec<dim>* image;
    SYCL_EXTERNAL vec<dim>& operator[](int element) const {
        return image[element];
    }
    Image(){

    }
    Image(int h, int w,sycl::queue* queue): h(h), w(w), queue(queue)
    {
        image = sycl::malloc_shared<vec<dim>>(h*w, *queue);
        initialized = true;
    }
    void deinit()
    {
        if(initialized){
            sycl::free(image,*queue);
        }
    }
    void reset()
    {
        for(int x=0;x<w;x++){
            for(int y=0;y<h;y++){
                float  r[dim];
                for(int i=0;i<dim;i++){
                    r[i] = rand()/static_cast<float>(RAND_MAX);
                }
                image[x+w*y] = vec<dim>(r);
            }
        }
    }
    void readImg(std::string path)
    {
        std::ifstream file;
        file.open(path);
        int w,h;
        file>>w;
        file>>h;
        for(int x=0;x<w;x++){
            for(int y=0;y<h;y++){
                int r,g,b,a;
                file>>r>>g>>b>>a;
                int  el = y+w*(w-1-x);
                image[el][0]=r/255.;
                image[el][1]=g/255.;
                image[el][2]=b/255.;
            }
        }
    }
};

#endif // __IMG_H__