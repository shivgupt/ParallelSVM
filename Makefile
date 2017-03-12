CXX = g++
CFLAGS = -Wall -Wconversion -O3 -fPIC -fopenmp
SHVER = 2
OS = $(shell uname)

all: svm-train svm-predict

svm-train: svm-train_parallel.c svm.o

	cc -openmp -Wl,svm.o svm-train_parallel.c -o svm-train -lm
svm.o: svm.cpp svm.h
	$(CXX) $(CFLAGS) -c svm.cpp
clean:
	rm -f *~ svm.o svm-train libsvm.so.$(SHVER)
