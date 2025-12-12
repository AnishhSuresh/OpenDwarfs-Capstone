# Makefile for SPMV 
# Parallel version uses OpenMP

CXX = g++
CXXFLAGS = -O3 -std=c++17 -Wall -Wextra -pedantic -fopenmp

all: spmv

spmv: main.o spmv_serial.o spmv_parallel.o
	$(CXX) $(CXXFLAGS) main.o spmv_serial.o spmv_parallel.o -o spmv

main.o: main.cpp
	$(CXX) $(CXXFLAGS) -c main.cpp

spmv_serial.o: spmv_serial.cpp
	$(CXX) $(CXXFLAGS) -c spmv_serial.cpp

spmv_parallel.o: spmv_parallel.cpp
	$(CXX) $(CXXFLAGS) -c spmv_parallel.cpp

clean:
	rm -f *.o spmv

.PHONY: all clean