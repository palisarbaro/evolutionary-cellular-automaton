#ifndef __VEC_H__
#define __VEC_H__
#include<CL/sycl.hpp>
template <int size>
struct vec{
    float arr[size];
    SYCL_EXTERNAL float& operator[](int n) {
        return arr[n];
    }
    SYCL_EXTERNAL vec(float* a){
        for(int i=0;i<size;i++){
            arr[i]=a[i];
        }
    }
    // SYCL_EXTERNAL vec();
    // SYCL_EXTERNAL void operator=(vec<size> other);

};

#endif // __VEC_H__