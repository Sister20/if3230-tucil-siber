#include <stdio.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>

#include <cuda.h>
#include <math.h>

using namespace std;

__global__ void normalizeTransform(double *mat, double *id_mat, int n, int i){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < n && row < n)
        if (col == i && col!=row){
            id_mat[col*n + row] /= mat[i*n + i];
            mat[col*n + row] /= mat[i*n + i];
        }

}

__global__ void transformToUnit(double *mat, double *id_mat, int n, int i) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < n && col < n) {
        if (col == i && row == i) {
            id_mat[col * n + row] /= mat[i * n + i];
            mat[col * n + row] /= mat[i * n + i];
        }
    }
}

__global__ void transformToDiagonal(double *mat, double *id_mat, int n, int i) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < n && col < n) {
        if (col != i) {
            id_mat[col * n + row] -= id_mat[i * n + row] * mat[col * n + i];
            if (row != i) {
                mat[col * n + row] -= mat[i * n + row] * mat[col * n + i];
            }
        }
    }
}

int main()
{
	int i = 0, j = 0, n = 0, blocksize = 0;

    // Get the dimension of the matrix
    cin >> n;

    // Determine the block size
    if (n < 8) {
        blocksize = 2;
    } else if (n < 32) {
        blocksize = 4;
    } else if (n < 128) {
        blocksize = 8;
    } else if (n < 512) {
        blocksize = 16;
    } else if (n < 2048) {
        blocksize = 32;
    } else {
        blocksize = 64;
    }

    // Allocating memory for matrix array
    double *mat_res = new double[n*n];
    double *mat = new double[n*n];

    // Read matrix from text file
    for (i = 0; i < n; ++i) {
        for (j = 0; j < n; ++j) {
            cin >> mat[i * n + j];
        }
    }

	double *mat_c, *id_mat_c;
	int size = n*n*sizeof(double);

    // Allocate memory on GPU
	dim3 threadsPerBlock(blocksize, blocksize);
	dim3 numBlocks((n + blocksize - 1) / blocksize, (n + blocksize - 1) / blocksize);
	cudaMalloc((void**)&mat_c, size);
    cudaMalloc((void**)&id_mat_c, size);
    
    // Initialize the identity matrix
    double *id_mat = new double[n*n];
    for (i = 0; i < n; ++i) {
        for (j = 0; j < n; ++j) {
            if (i == j) {
                id_mat[i * n + i] = 1.0;
            } else {
                id_mat[i * n + j] = 0.0;
            }
        }
    }

    // Copy data from CPU to GPU
	cudaMemcpy(mat_c, mat, size, cudaMemcpyHostToDevice);
    cudaMemcpy(id_mat_c, id_mat, size, cudaMemcpyHostToDevice);

	// Perform the Gauss-Jordan elimination
    for (i = 0; i < n; ++i) {
        normalizeTransform <<<numBlocks, threadsPerBlock >>>(mat_c, id_mat_c, n, i);
        // Transform the matrix into a unit matrix
        transformToUnit<<<numBlocks, threadsPerBlock>>>(mat_c, id_mat_c, n, i);

        // Transform the matrix into a diagonal matrix
        transformToDiagonal<<<numBlocks, threadsPerBlock>>>(mat_c, id_mat_c, n, i);
    }

	// Copy data from GPU to CPU
	cudaMemcpy(mat_res, id_mat_c, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(id_mat, mat_c, size, cudaMemcpyDeviceToHost);

    // Free the memory
	cudaFree(mat_c);
    cudaFree(id_mat_c);

    // Output result
    for (i = 0; i < n; ++i) {
        for (j = 0; j < n; ++j) {
            cout << mat_res[i * n + j] << " ";
        }
        cout << endl;
    }

	delete[]mat;
	delete[]mat_res;

	return 0;
}