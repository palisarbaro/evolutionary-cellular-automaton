#include <cstdlib>
#include "h/ui.h"
#include "h/model.hpp"

int main(int argc, char* argv[]) {
	std::cout<<std::is_trivially_copyable<Fields<20>>::value<<std::endl;
	std::srand(std::time(nullptr));
	Model<20> m(200,200);
	UI::model = &m;
	init(argc,argv,5);
	loop();
	return 0;
}
