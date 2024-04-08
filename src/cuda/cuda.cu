#include <stdio.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>

#include <cuda.h>
#include <math.h>

using namespace std;

#define blocksize 32

int main()
{
	int i = 0, j = 0, n = 0;

    // Get the dimension of the matrix
    cin >> n;

    // Allocating memory for matrix array
    double *iL = new double[n*n];
    double *mat = new double[n*n]();

    // Read matrix from text file
    for (i = 0; i < n; ++i) {
        for (j = 0; j < n; ++j) {
            cin >> mat[i * n + j];
        }
    }

	double *mat_c, *id_mat, *id_mat_c;
	int size = n*n*sizeof(double);

    // Allocate memory on GPU
	dim3 threadsPerBlock(blocksize, blocksize);
	dim3 numBlocks((n + blocksize - 1) / blocksize, (n + blocksize - 1) / blocksize);
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
    // TODO

	// Copy data from GPU to CPU
	cudaMemcpy(iL, id_mat_c, size, cudaMemcpyDeviceToHost);
	cudaMemcpy(id_mat, mat_c, size, cudaMemcpyDeviceToHost);

    // Free the memory
	cudaFree(mat_c);
	cudaFree(id_mat_c);

    // Print the result
    for (i = 0; i < n; ++i) {
        for (j = 0; j < n; ++j) {
            cout << iL[i * n + j] << " ";
        }
        cout << endl;
    }

	delete[]id_mat;
	delete[]mat;
	delete[]iL;

	return 0;
}