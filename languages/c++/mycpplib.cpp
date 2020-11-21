#include <iostream>
#include <cmath>

extern "C" {

void my_cpp_transform(double * params, size_t d) {
	std::cout << "C++ transform called" << std::endl;
	for(size_t i=0; i < d; i++) {
		params[i] = params[i] * 2 - 1;
	}
}

double my_cpp_likelihood(double * params, size_t d) {
	double l = 0.0;
	for(size_t i=0; i < d; i++) {
		l += pow((params[i] - i * 0.1)/0.01, 2);
	}
	std::cout << "C++ likelihood called: " << l << std::endl;
	return -0.5 * l;
}


void my_cpp_transform_vectorized(double * params, size_t d, size_t n) {
	std::cout << "vectorized C++ transform called" << std::endl;
	for(size_t j=0; j < n; j++) {
	    for(size_t i=0; i < d; i++) {
		    params[j * d + i] = params[j * d + i] * 2 - 1;
	    }
	}
}

void my_cpp_likelihood_vectorized(double * params, size_t d, size_t n, double * like) {
	for(size_t j=0; j < n; j++) {
		like[j] = 0.;
	    for(size_t i=0; i < d; i++) {
			like[j] -= 0.5 * pow((params[j * d + i] - i * 0.1)/0.01, 2);
		}
	}
	std::cout << "vectorized C++  likelihood called" << std::endl;
}

}
