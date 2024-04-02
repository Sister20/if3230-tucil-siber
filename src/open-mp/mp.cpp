#include <iostream>
#include <omp.h>

// Matrix struct
struct Matrix {
    double *data;
    int rows;
    int cols;

    // Constructor
    Matrix(int rows, int cols) : rows(rows), cols(cols) {
        data = new double[rows * cols];
    }

    // Destructor
    ~Matrix() {
        delete[] data;
    }

    // Overloaded operator []
    double &operator[](int index) {
        return data[index];
    }
};

int main()
{
    int m_dimension = 0, i = 0, j = 0, k = 0;
    double d = 0.0;

    // Get the dimension of the matrix
    std::cin >> m_dimension;

    // Read matrix from text file
    Matrix matrix(2 * m_dimension, 2 * m_dimension);
    for (int i = 0; i < m_dimension; ++i) {
        for (int j = 0; j < m_dimension; ++j) {
            std::cin >> matrix[i * (2 * m_dimension) + j];
        }

        // Append the identity matrix
        matrix[i * (2 * m_dimension) + m_dimension + i] = 1.0;
    }

    // Transform the matrix into a diagonal matrix
    for (i = 0; i < m_dimension; ++i) {
        for (j = 0; j < 2 * m_dimension; ++j) {
            if (j != i) {
                d = matrix[j * (2 * m_dimension) + i] / matrix[i * (2 * m_dimension) + i];
                for (k = 0; k < 2 * m_dimension; ++k) {
                    matrix[j * (2 * m_dimension) + k] -= matrix[i * (2 * m_dimension) + k] * d;
                }
            }
        }
    }
    
    // Transform the matrix into a unit matrix
    for (i = 0; i < m_dimension; ++i) {
        d = matrix[i * (2 * m_dimension) + i];
        for (j = 0; j < 2*m_dimension; ++j) {
            matrix[i * (2 * m_dimension) + j] /= d;
        }
    }

    // Print output to a text file
    for (int i = 0; i < m_dimension; ++i) {
        for (int j = m_dimension; j < 2 * m_dimension; ++j) {
            std::cout << matrix[i * (2 * m_dimension) + j] << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}
