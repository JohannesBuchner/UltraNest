#include <iostream>
#include <cmath>

extern "C" {

void my_cpp_transform(double * params, size_t d) {
	std::cout << "C tranform called" << std::endl;
	for(size_t i=0; i < d; i++) {
		params[i] = params[i] * 2 - 1;
	}
}

double my_cpp_likelihood(double * params, size_t d) {
	double l = 0.0;
	for(size_t i=0; i < d; i++) {
		l += pow((params[i] - i * 0.1)/0.01, 2);
	}
	std::cout << " likelihood called: " << l << std::endl;
	return -0.5 * l;
}

}
