#define _CRT_SECURE_NO_WARNINGS

#include <stdio.h>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
//2015253039 ±ÇÁø¿ì

__global__ void helloWorld(char* str) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	str[idx] += idx;
}

int main(int argc, char** argv)
{
	int i;
	char strin[12] = "Hello";
	char str[] = "Hello World!";
	printf("%s", strin);

	for (i = 0; i < 12; i++)
	{
		str[i] -= i;
	}

	printf("%s\n", str);

	char* d_str;
	size_t size = sizeof(str);
	cudaMalloc((void**)&d_str, size);

	cudaMemcpy(d_str, str, size, cudaMemcpyHostToDevice);

	dim3 dimBlock(2);
	dim3 dimThread(6);

	helloWorld<<< dimBlock, dimThread >>>(d_str);

	cudaMemcpy(str, d_str, size, cudaMemcpyDeviceToHost);

	cudaFree(d_str);

	printf("%s\n", str);
	return 0;
}