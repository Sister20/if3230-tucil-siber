#include <stdio.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>

#include <cuda.h>
#include <math.h>

using namespace std;

int main()
{
	int i = 0, j = 0, n = 0, blocksize = 0;

    // Get the dimension of the matrix
    cin >> n;

    // Determine the block size
    if (n < 8){
        blocksize = 2;
    } else if (n < 32){
        blocksize = 4;
    } else if (n < 128){
        blocksize = 8;
    } else if (n < 512){
        blocksize = 16;
    } else if (n < 2048){
        blocksize = 32;
    } else {
        blocksize = 64;
    }

    // Allocating memory for matrix array
    double *mat_res = new double[n*2*n];
    double *mat = new double[n*2*n];

    // Read matrix from text file
    for (i = 0; i < n; ++i) {
        for (j = 0; j < n; ++j) {
            cin >> mat[i * 2 * n + j];
        }

        // Append identity matrix
        mat[i * 2 * n + n + i] = 1.0;
    }

	double *mat_c;
	int size = n*2*n*sizeof(double);

    // Allocate memory on GPU
	dim3 threadsPerBlock(blocksize, blocksize);
	dim3 numBlocks((n + blocksize - 1) / blocksize, (n + blocksize - 1) / blocksize);
	cudaMalloc((void**)&mat_c, size);

	// Copy data from CPU to GPU
	cudaMemcpy(mat_c, mat, size, cudaMemcpyHostToDevice);

	// Perform the Gauss-Jordan elimination
    // TODO

	// Copy data from GPU to CPU
	cudaMemcpy(mat_res, mat_c, size, cudaMemcpyDeviceToHost);

    // Free the memory
	cudaFree(mat_c);

    // Print the result
    for (i = 0; i < n; ++i) {
        for (j = 0; j < n; ++j) {
            cout << mat_res[i * 2 * n + n + j] << " ";
        }
        cout << endl;
    }

	delete[]mat;
	delete[]mat_res;

	return 0;
}