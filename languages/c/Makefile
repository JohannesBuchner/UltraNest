mylib.so: mylib.c
	gcc -Wall -Wextra -shared -o mylib.so mylib.c

run: mylib.so
	LD_LIBRARY_PATH=. python3 runcsimple.py
	LD_LIBRARY_PATH=. python3 runc.py

clean: 
	rm mylib.so

.PHONY: run clean
