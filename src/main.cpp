#include <cstdlib>
#include "ui.h"
#include "model.h"

int main(int argc, char* argv[]) {
	std::srand(std::time(nullptr));
	Model m(1000,1000);
	UI::model = &m;
	init(argc,argv,1);
	loop();
	return 0;
}
