#include <stdio.h>
#include <math.h>

void my_c_transform(double * params, size_t d) {
	// printf("C tranform called\n");
	for(size_t i=0; i < d; i++) {
		params[i] = params[i] * 2 - 1;
	}
}

double my_c_likelihood(double * params, size_t d) {
	double l = 0.0;
	for(size_t i=0; i < d; i++) {
		l += pow((params[i] - i * 0.1)/0.01, 2);
	}
	// printf("C likelihood called: %f\n", l);
	return -0.5 * l;
}
