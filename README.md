# countsort-mpi-cuda

This utilizes [counting sort][1] to sort integers in parallel with MPI and CUDA. The MPI code itself is written in C++ while the CUDA code (.cu) is written in CUDA C. It may get modified later to use C++11.

The included makefile may not work in all cases but generally shows what is required to build the project.

NOTE: This currently runs on Linux only but can be modified to run on Windows (for instance, replacing [`gettimeofday()`][2] with another function).


[1]: https://en.wikipedia.org/wiki/Counting_sort
[2]: http://pubs.opengroup.org/onlinepubs/009695399/functions/gettimeofday.html
