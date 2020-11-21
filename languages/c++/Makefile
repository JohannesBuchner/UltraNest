mycpplib.so: mycpplib.cpp
	g++ -Wall -Wextra -fPIC -shared -o mycpplib.so mycpplib.cpp

run: mycpplib.so
	LD_LIBRARY_PATH=. python3 runcpp.py

runsimple: mycpplib.so
	LD_LIBRARY_PATH=. python3 runsimplecpp.py

clean: 
	rm mycpplib.so

.PHONY: run clean
