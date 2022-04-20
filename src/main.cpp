#include <cstdlib>
#include "ui.h"
#include "model.h"

int main(int argc, char* argv[]) {
	std::srand(std::time(nullptr));
	Model m(200,200,2);
	UI::model = &m;
	init(argc,argv,5);
	loop();
	return 0;
}
