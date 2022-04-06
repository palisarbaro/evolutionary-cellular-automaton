#include "model.h"
Model::Model(int h, int w):h(h),w(w)
{
    auto device = sycl::platform::get_platforms()[0].get_devices()[0];
    queue = sycl::queue(device);
    curr = sycl::malloc_shared<elType>(h*w, queue);
    next = sycl::malloc_shared<elType>(h*w, queue);

     for(int x=0;x<w;x++){
        for(int y=0;y<h;y++){
            curr[x+w*y] = rand()/static_cast<float>(RAND_MAX)>0.5?1:0;
        }
    }
}

Model::~Model()
{
    sycl::free(curr,queue);
    sycl::free(next,queue);
}

void Model::step(int iterations)
{
    auto curr = this->curr;
    auto next = this->next;
    int w = this->w;
    int h = this->h;
    for (int iteration = 0; iteration < iterations; iteration++) {
		sycl::event e = queue.parallel_for(sycl::range<2>(w,h), [=](sycl::id<2> item) {
            int x = item.get(0);
            int y = item.get(1);
            int count = 0;
            for(int i=-1;i<2;i++){
                for(int j=-1;j<2;j++){
                    int X = (x+i+w)%w;
                    int Y = (y+j+h)%h;
                    count += curr[X+w*Y];
                }
            }
            int res = 0;
            if(curr[x+w*y]==0){
                if(count==3){
                    res=1;
                }
            }
            else{
                if(count==3 || count==2){
                    res = 1;
                }
            }
			next[x+w*y]=res;
		});
		e.wait_and_throw();

		elType* tmp = curr;
		curr = next;
		next = tmp;

	}
    this->curr = curr;
    this->next = next;
    
}
