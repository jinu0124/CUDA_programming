#define _CRT_SECURE_NO_WARNINGS

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
//2015253039 ������
__global__ void VecMul(double* X, double* Y, double* Z) {
	//blockIdx.x�� �ش� �����尡 �ִ� Block�� ��ȣ�� ����´�.
	//threadIdx.x�� �ش� ���� ���� �� block���� thread�� �Ҵ�� ���� ��´�.
	//�Ҵ�� ��� �����尡 ��ɾ ���� �۾��� ���ķ� �ѹ��� �����Ѵ�.
	//Grid[x][y] > Block[x][y] > Thread[x][y] <- kernel�Լ��� ȣ�� �� ���� ���ڿ� �°� �� 1,2������ �����ȴ�.
	//���� ��� <2, 512> �� �Ѱ��־����� block�� 1�������� 2��[0��, 1��], block�� thread������ 512��(1����)�� �ȴ�.
	//�� ��, blockDim.x = 512, blockIdx.x = �ش� �����尡 ��ġ�� BlockIdx(��ȣ) �̴�.
	for (int i = 0; i < 10000; i++)
	{
		int tx = blockDim.x * blockIdx.x + threadIdx.x;
		int ty = blockDim.x * blockIdx.x + threadIdx.x;
		Z[blockDim.x * blockIdx.x + threadIdx.x] = X[tx] * Y[ty];
	}
	//Z[]�� �־� �ٶ��� �� �����尡 ����� ���� ���� ��Ŀ� 1:1�� �µ��� Z[blockDim.x + ThreadIdx.x]�� ���־���Ѵ�.
}

int main(int argc, char** argv)
{
	double* X, * Y, * Z; // GPU���� ����� ���� ����
	const int n = 60; // ����� ũ�� NxN
	float timer;
	cudaEvent_t start, stop;
	clock_t tstart, tstop;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	//Device�� ������ ���� memory ���� �Ҵ�
	cudaMalloc((void**)&X, n * n * sizeof(double)); // size : byte����
	cudaMalloc((void**)&Y, n * n * sizeof(double)); // size : byte����
	cudaMalloc((void**)&Z, n * n * sizeof(double)); // size : byte����

	// a,b,c�� Memory �Ҵ�
	double a[n][n] = { 0, };
	double b[n][n] = { 0, };
	double c[n][n] = { 0, };
	int flag = 0;
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++)
		{
			flag += 1;
			a[i][j] = flag; // �� ���� & �Ҵ�
		}
	}
	flag = 0;
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++)
		{
			flag += 1;
			b[i][j] = flag; // �� ���� & �Ҵ�
		}
	}
	
	//Copy Input Data to Device
	cudaMemcpy(X, a, n * n * sizeof(double), cudaMemcpyHostToDevice);// X <- a
	cudaMemcpy(Y, b, n * n * sizeof(double), cudaMemcpyHostToDevice);// ���� M��
	
	dim3 grid = 6; // M���� block �Ҵ�
	dim3 block = 600; // N���� �����带 ������ �� �Ҵ�

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