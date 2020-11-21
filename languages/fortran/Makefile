myfortlib.so: myfortlib.f90
	gfortran -ggdb -Wall -Wextra -fPIC -shared -o myfortlib.so myfortlib.f90

run: myfortlib.so
	LD_LIBRARY_PATH=. python3 runfort.py

clean: 
	rm myfortlib.so

.PHONY: run clean
