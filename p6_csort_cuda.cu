#include <cuda.h>
#include <stdio.h>

#define THREADS 256

__device__ unsigned long long deviceCount[65536];

__global__ void GenInputKernel(const int gpuCut, unsigned short* deviceData)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (tid < gpuCut)
    {
        deviceData[tid] = (gpuCut - tid) & 65535;
    }
}

__global__ void InitCountKernel()
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    deviceCount[tid] = 0;
}

__global__ void CountKernel(const int gpuCut, const unsigned short* deviceData)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (tid < gpuCut)
    {
        atomicAdd(&deviceCount[deviceData[tid]], 1ULL);
    }
}

static void CudaTest(const char* msg)
{
    cudaError_t e;
    cudaThreadSynchronize();
    
    if (cudaSuccess != (e = cudaGetLastError()))
    {
        fprintf(stderr, "%s: %d\n", msg, e);
        fprintf(stderr, "%s\n", cudaGetErrorString(e));
        exit(-1);
    }
}

unsigned short* allocGPU(const int gpuCut)
{
    unsigned short* deviceData;
    
    if (cudaSuccess != cudaMalloc((void **)&deviceData, gpuCut * sizeof(unsigned short)))
    {
        fprintf(stderr, "could not allocate GPU array\n");
        exit(-1);
    }
    
    return deviceData;
}

void deallocGPU(unsigned short* deviceData)
{
    cudaFree(deviceData);
}

void runGPU(const int gpuCut, const int cpuCut, unsigned short* deviceData, unsigned short* hostData)
{
    InitCountKernel<<<65536 / THREADS, THREADS>>>();
    CudaTest("InitCountKernel failed\n");
    
    GenInputKernel<<<(gpuCut + THREADS - 1) / THREADS, THREADS>>>(gpuCut, deviceData);
    CudaTest("GenInputKernel failed\n");
    
    CountKernel<<<(gpuCut + THREADS - 1) / THREADS, THREADS>>>(gpuCut, deviceData);
    CudaTest("CountKernel failed\n");
    
    cudaMemcpy(hostData + cpuCut, deviceData, gpuCut, cudaMemcpyDeviceToHost);
}

