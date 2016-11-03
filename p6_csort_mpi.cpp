#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <new>
#include <sys/time.h>
#include <mpi.h>

unsigned short* allocGPU(const int gpuCut);
void deallocGPU(unsigned short* deviceData);
void runGPU(const int gpuCut, const int cpuCut, unsigned short* deviceData, unsigned short* hostData);

bool verifyResult(const long size, const unsigned short* globalData)
{
    long i = 1;
    
    while ((i < size) && (globalData[i - 1] <= globalData[i]))
    {
        i++;
    }

    return i < size;
}

bool isValidInput(const long size, long int processes, const int gpuPercentage)
{
    if ((size < 1) || (size % processes != 0) || ((gpuPercentage < 0) || (gpuPercentage > 100)))
    {
        return false;
    }

    return true;
}

int main(int argc, char* argv[])
{
    if (argc != 3)
    {
        std::cerr << "usage: " << argv[0] << " number_of_elements gpu_percentage\n";
        return EXIT_FAILURE;
    }

    MPI::Init();
    const int processes = MPI::COMM_WORLD.Get_size();
    const int rank = MPI::COMM_WORLD.Get_rank();

    const long size = std::atol(argv[1]);
    const int gpuPercentage = std::atoi(argv[2]);

    if (!isValidInput(size, processes, gpuPercentage))
    {
        if (rank == 0)
        {
            std::cerr << "invalid input\n";
        }

        MPI::Finalize();
        return EXIT_FAILURE;
    }

    const int processPortion = size / processes;
    const int gpuCut = static_cast<float>(gpuPercentage) / 100 * processPortion;
    const int cpuCut = processPortion - gpuCut;

    if (rank == 0)
    {
        std::cout << "CountSort [MPI][CUDA]\n\n";
        std::cout << "sorting " << size << " values with " << processes << " process(es)\n";
        std::cout << "each process receives " << processPortion << " values\n";
        std::cout << "the GPU will handle " << gpuPercentage << "% of the workload (" << gpuCut << " values)\n\n";
    }

    // allocate arrays
    unsigned short* globalData;

    if (rank == 0)
    {
        globalData = new(std::nothrow) unsigned short[size];
    }

    unsigned short* hostData = new(std::nothrow) unsigned short[processPortion];

    if (!globalData || !hostData)
    {
        std::cerr << "could not allocate a host array\n";
        MPI::Finalize();
        return EXIT_FAILURE;
    }

    unsigned short* deviceData;

    if (gpuPercentage > 0)
    {
        deviceData = allocGPU(gpuCut);
    }

    // generate input
    if (rank == 0)
    {
        for (long i = 0; i < size; i++)
        {
            globalData[i] = (size - i) & 65535;
        }
    }

    // start time
    MPI::COMM_WORLD.Barrier();
    struct timeval start;
    gettimeofday(&start, NULL);

    MPI::COMM_WORLD.Scatter(globalData, processPortion, MPI::UNSIGNED_SHORT, hostData, processPortion, MPI::UNSIGNED_SHORT, 0);

    if (gpuPercentage > 0)
    {
        runGPU(gpuCut, cpuCut, deviceData, hostData);
    }

    // initialize the counts
    long localCount[65536];

    for (long i = 0; i < 65536; i++)
    {
        localCount[i] = 0;
    }

    // count the values
    for (long i = 0; i < cpuCut; i++)
    {
        localCount[hostData[i]]++;
    }

    long globalCount[65536];

    MPI::COMM_WORLD.Reduce(localCount, globalCount, 65536, MPI::LONG, MPI::SUM, 0);

    // compute the positions
    if (rank == 0)
    {
        for (long i = 1; i < 65536; i++)
        {
            globalCount[i] += globalCount[i - 1];
        }
    }

    // write the sorted values
    if (rank == 0)
    {
        for (long i = 0; i < globalCount[0]; i++)
        {
            globalData[i] = 0;
        }

        for (long i = 1; i < 65536; i++)
        {
            for (long j = globalCount[i - 1]; j < globalCount[i]; j++)
            {
                globalData[j] = i;
            }
        }
    }

    // end time
    struct timeval end;
    gettimeofday(&end, NULL);

    if (rank == 0)
    {
        std::cout << "compute time was " << std::setprecision(4) << std::fixed
                  << end.tv_sec + end.tv_usec / 1000000.0 - start.tv_sec - start.tv_usec / 1000000.0 << "s\n";
    }

    // verify result
    if (rank == 0)
    {
        std::cout << (verifyResult(size, globalData) ? "NOT sorted\n" : "sorted\n");
        delete[] globalData;
    }

    if (gpuPercentage > 0)
    {
        deallocGPU(deviceData);
    }

    delete[] hostData;
    MPI::Finalize();
}