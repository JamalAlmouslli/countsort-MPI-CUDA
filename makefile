csort: csort_mpi.o csort_cuda.o
	mpic++ -march=native -O3 csort_mpi.o csort_cuda.o
	-lcudart -L /usr/local/cuda/lib64/ -o csort
	
csort_mpi.o: csort_mpi.cpp
	mpic++ -march=native -O3 -c csort_mpi.cpp
	
csort_cuda.o: csort_cuda.cu
	nvcc -O3 -arch=sm_20 -c csort_cuda.cu
	
clean:
	rm -f csort_mpi.o csort_cuda.o csort