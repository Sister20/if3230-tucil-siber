#include <stdio.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>

#include <cuda.h>
#include <math.h>

using namespace std;


__global__ void nodiag_normalize(double *A, double *id_mat, int n, int i){
 int x = blockIdx.x * blockDim.x + threadIdx.x;
 int y = blockIdx.y * blockDim.y + threadIdx.y;

 if (x < n && y < n)
 if (x == i && x!=y){
  id_mat[x*n + y] /= A[i*n + i];
  A[x*n + y] /= A[i*n + i];
 }

}

__global__ void diag_normalize(double *A, double *id_mat, int n, int i){
 int x = blockIdx.x * blockDim.x + threadIdx.x;
 int y = blockIdx.y * blockDim.y + threadIdx.y;

 if (x < n && y < n)
 if (x == y && x == i){
  id_mat[x*n + y] /= A[i*n + i];
  A[x*n + y] /= A[i*n + i];
 }
}

__global__ void gaussjordan(double *A, double *id_mat, int n, int i)
{
 int x = blockIdx.x * blockDim.x + threadIdx.x;
 int y = blockIdx.y * blockDim.y + threadIdx.y;

 if (x < n && y < n){
  if (x != i){
   id_mat[x*n + y] -= id_mat[i*n + y] * A[x*n + i];
   if (y != i){
    A[x*n + y] -= A[i*n + y] * A[x*n + i];
   }
  }
 }

}

int main()
{
    int i = 0, j = 0, n = 0, blocksize = 0;

    // Get the dimension of the matrix
    cin >> n;

    if (n<8){
        blocksize = 2;
    } else if (n<32){
        blocksize = 4;
    } else if (n<128){
        blocksize = 8;
    } else if (n<512){
        blocksize = 16;
    } else if (n<2048){
        blocksize = 32;
    } else {
        blocksize = 64;
    }

    // Allocating memory for matrix array
    double *iL = new double[n*n];
    double *mat = new double[n*n]();

    // Read matrix from text file
    for (i = 0; i < n; ++i) {
        for(j = 0; j < n; ++j) {
            cin >> mat[i * n + j];
        }
    }

    double *mat_c, *id_mat, *id_mat_c;
    int size = n*n*sizeof(double);

    // Allocate memory on GPU
    dim3 threadsPerBlock(blocksize, blocksize);
    dim3 numBlocks((n) / blocksize, (n) / blocksize);
    cudaMalloc((void**)&mat_c, size);
    cudaMalloc((void**)&id_mat_c, size);
    id_mat = new double[n*n];


    // Initialize the identity matrix
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i == j) {
                id_mat[i*n + i] = 1.0;
            } else {
                id_mat[i*n + j] = 0.0;
            }
        }
    }

    // Copy data from CPU to GPU
    cudaMemcpy(mat_c, mat, size, cudaMemcpyHostToDevice);
    cudaMemcpy(id_mat_c, id_mat, size, cudaMemcpyHostToDevice);

    // Perform the Gauss-Jordan elimination
    for (int i = 0; i < n; i++){
        nodiag_normalize <<<numBlocks, threadsPerBlock >>>(mat_c, id_mat_c, n, i);
        diag_normalize <<<numBlocks, threadsPerBlock >>>(mat_c, id_mat_c, n, i);
        gaussjordan <<<numBlocks, threadsPerBlock >>>(mat_c, id_mat_c, n, i);
    }

    //copy data from GPU to CPU
    cudaMemcpy(iL, id_mat_c, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(id_mat, mat_c, size, cudaMemcpyDeviceToHost);

    cudaFree(mat_c);
    cudaFree(id_mat_c);

    //print the result
    for (i = 0; i < n; ++i) {
        for(j = 0; j < n; ++j)
        {
            cout << iL[i * n + j] << " ";
        }
        cout << endl;
    }

    delete[]id_mat;
    delete[]mat;
    delete[]iL;

    return 0;
}