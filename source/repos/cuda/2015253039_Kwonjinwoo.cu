#define _CRT_SECURE_NO_WARNINGS

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
//2015253039 권진우
__global__ void VecMul(double* X, double* Y, double* Z) {
	//blockIdx.x는 해당 쓰레드가 있는 Block의 번호를 내뱉는다.
	//threadIdx.x는 해당 값이 수행 될 block내의 thread에 할당된 값을 뱉는다.
	//할당된 모든 쓰레드가 명령어에 대한 작업을 병렬로 한번에 수행한다.
	//Grid[x][y] > Block[x][y] > Thread[x][y] <- kernel함수를 호출 할 때의 인자에 맞게 각 1,2차원이 결정된다.
	//예를 들어 <2, 512> 를 넘겨주었으면 block은 1차원으로 2개[0번, 1번], block내 thread개수는 512개(1차원)가 된다.
	//이 때, blockDim.x = 512, blockIdx.x = 해당 쓰레드가 위치한 BlockIdx(번호) 이다.
	for (int i = 0; i < 10000; i++)
	{
		int tx = blockDim.x * blockIdx.x + threadIdx.x;
		int ty = blockDim.x * blockIdx.x + threadIdx.x;
		Z[blockDim.x * blockIdx.x + threadIdx.x] = X[tx] * Y[ty];
	}
	//Z[]에 넣어 줄때도 각 쓰레드가 계산한 것이 목적 행렬에 1:1로 맞도록 Z[blockDim.x + ThreadIdx.x]를 해주어야한다.
}

int main(int argc, char** argv)
{
	double* X, * Y, * Z; // GPU에서 사용할 변수 선언
	const int n = 60; // 행렬의 크기 NxN
	float timer;
	cudaEvent_t start, stop;
	clock_t tstart, tstop;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	//Device에 변수를 위한 memory 공간 할당
	cudaMalloc((void**)&X, n * n * sizeof(double)); // size : byte단위
	cudaMalloc((void**)&Y, n * n * sizeof(double)); // size : byte단위
	cudaMalloc((void**)&Z, n * n * sizeof(double)); // size : byte단위

	// a,b,c에 Memory 할당
	double a[n][n] = { 0, };
	double b[n][n] = { 0, };
	double c[n][n] = { 0, };
	int flag = 0;
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++)
		{
			flag += 1;
			a[i][j] = flag; // 값 생성 & 할당
		}
	}
	flag = 0;
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++)
		{
			flag += 1;
			b[i][j] = flag; // 값 생성 & 할당
		}
	}
	
	//Copy Input Data to Device
	cudaMemcpy(X, a, n * n * sizeof(double), cudaMemcpyHostToDevice);// X <- a
	cudaMemcpy(Y, b, n * n * sizeof(double), cudaMemcpyHostToDevice);// 전역 M로
	
	dim3 grid = 6; // M개의 block 할당
	dim3 block = 600; // N개의 쓰레드를 가지는 블럭 할당

	// event record start
	cudaEventRecord(start, 0);

	VecMul << < grid, block >> > (X, Y, Z);

	tstart = clock();
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			for (int k = 0; k < 10000; k++) {
				c[i][j] = a[i][j] * b[i][j];
			}
		}
	}
	tstop = clock();
	double t;
	t = (double)tstop - (double)tstart;
	printf("time(cpu) : %.0f ms\n", t);
	
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	cudaEventElapsedTime(&timer, start, stop);
	cudaMemcpy(c, Z, n * n * sizeof(double), cudaMemcpyDeviceToHost);

	/*for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			printf("%lf\n", c[i][j]);
		}
	}*/
	
	printf("time(gpu) : %f ms(Micro Second)\n", timer);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	cudaFree(X); cudaFree(Y); cudaFree(Z);

	return 0;
}