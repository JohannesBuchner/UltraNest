#include <stdio.h>
#include <math.h>

// simple version: one parameter vector per function call

void my_c_transform(double * params, size_t d) {
	// printf("C transform called\n");
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

// vectorized version: multiple parameter vectors per function call

void my_c_transform_vectorized(double * params, size_t d, size_t n) {
	// printf("vectorized C transform called\n");
	for(size_t j=0; j < n; j++) {
	    for(size_t i=0; i < d; i++) {
		    params[j * d + i] = params[j * d + i] * 2 - 1;
	    }
	}
}

void my_c_likelihood_vectorized(double * params, size_t d, size_t n, double * like) {
	// printf("vectorized C likelihood called\n");
	for(size_t j=0; j < n; j++) {
		like[j] = 0.;
	    for(size_t i=0; i < d; i++) {
			like[j] -= 0.5 * pow((params[j * d + i] - i * 0.1)/0.01, 2);
		}
	}
}
