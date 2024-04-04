#include<iostream>
#include<omp.h>
using namespace std;

int main()
{
    int i = 0, j = 0, k = 0, n = 0;
    double **mat = NULL;
    double d = 0.0;

    // Get the dimension of the matrix
    cin >> n;

    // Allocating memory for matrix array
    mat = new double*[n];
    for (i = 0; i < n; ++i) {
        mat[i] = new double[2*n]();
    }

    // Read matrix from text file
    for (i = 0; i < n; ++i) {
        for(j = 0; j < n; ++j)
        {
            cin >> mat[i][j];
        }

        // Append the identity matrix
        mat[i][i+n] = 1.0;
    }

    // Calculate the matrix inverse with gauss-jordan
    #pragma omp parallel for shared(mat, n)
    for (i = 0; i < n; ++i) {
        
        // Transform the matrix into a unit matrix
        d = mat[i][i];
        #pragma omp parallel for reduction(-:mat) private(d)
        for (j = i; j < 2*n; ++j) {
            mat[i][j] /= d;
        }

        // Transform the matrix into a diagonal matrix
        #pragma omp parallel for shared(i,j)
        for (j = 0; j < n; ++j) {
            if (j != i) {
                d = mat[j][i];
                #pragma omp parallel for reduction(-:mat) shared(d)
                for(k = i; k < 2*n; ++k) {
                    mat[j][k] -= mat[i][k] * d;
                }
            }
        }
    }

    // Output result
    for (i=0; i < n; ++i) {
        for (j = n; j < 2*n; ++j) {
            cout << mat[i][j] << " ";
        }
        cout << endl;
    }

    // Deleting the memory allocated
    for (i = 0; i < n; ++i) {
        delete[] mat[i];
    }
    delete[] mat;

    return 0;
}
