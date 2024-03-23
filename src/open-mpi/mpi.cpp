#include "mpi.h"

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

    // Overload to access the matrix elements
    double *operator[](int row) {
        return data + row * cols;
    }
};


int main(int argc, char *argv[]) {
    // Initialize MPI
    MPI_Init(&argc, &argv);

    int num_processes, rank, m_dimension;

    // Get the total number of processes
    MPI_Comm_size(MPI_COMM_WORLD, &num_processes);

    // Get current rank (Process ID)
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Get the dimension of the matrix
    if (rank == 0) {
        std::cin >> m_dimension;
    }

    // Broadcast the dimension of the matrix to all processes
    MPI_Bcast(&m_dimension, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Calculate the number of rows for each matrix chunk
    const int chunk_rows = m_dimension / num_processes;

    // Read matrix from text file
    Matrix matrix(m_dimension, m_dimension * 2);
    if (rank == 0) {
        for (int i = 0; i < m_dimension; i++) {
            for (int j = 0; j < m_dimension; j++) {
                std::cin >> matrix[i][j];
            }

            // Append the identity matrix
            matrix[i][m_dimension + i] = 1.0;
        }
    }

    // Scatter the matrix into chunks (cyclic)
    double *chunk = new double[chunk_rows * m_dimension * 2];
    for (int i = 0; i < chunk_rows; i++) {
        MPI_Scatter(matrix[0] + i * m_dimension * 2 * num_processes, m_dimension * 2, MPI_DOUBLE,
                    chunk + i * m_dimension * 2, m_dimension * 2, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    // Allocate memory to store pivot row sent by other processes
    double *pivot_row = new double[m_dimension * 2];

    // Calculate the matrix inverse with gauss-jordan
    for (int row = 0; row < chunk_rows; row++) {
        // Assign each iteration (current rank) to the belonging process
        for (int current_rank = 0; current_rank < num_processes; current_rank++) {
            // Calculate the column of the currently evaluated pivot
            int elim_col = row * num_processes + current_rank;

            // Only execute matrix calculation if the current rank is equal to the belonging rank
            if (rank == current_rank) {
                // Transform the matrix into a unit matrix
                double pivot_value = chunk[row * m_dimension * 2 + elim_col];
                for (int col = elim_col; col < m_dimension * 2; col++) {
                    chunk[row * m_dimension * 2 + col] /= pivot_value;
                }

                // Send the pivot row to the other ranks
                MPI_Bcast(chunk + row * m_dimension * 2, m_dimension * 2, MPI_DOUBLE, current_rank, MPI_COMM_WORLD);

                // Transform the matrix into a diagonal matrix
                for (int elim_row = 0; elim_row < chunk_rows; elim_row++) {
                    if (elim_row != row) {
                        // Get the ratio for elimination
                        double scale = chunk[elim_row * m_dimension * 2 + elim_col];

                        // Eliminate elements in both directions
                        for (int col = elim_col; col < m_dimension * 2; col++) {
                            chunk[elim_row * m_dimension * 2 + col] -= chunk[row * m_dimension * 2 + col] * scale;
                        }
                    }
                }
            }
            else {
                // Receive the pivot from the sending rank
                MPI_Bcast(pivot_row, m_dimension * 2, MPI_DOUBLE, current_rank, MPI_COMM_WORLD);

                // Local start index for the forward and backward elimination
                int local_start_forward = (rank < current_rank) ? row + 1 : row;
                int local_start_backward = row;

                // Forward elimination (Elimination for the lower half of the matrix)
                for (int elim_row = local_start_forward; elim_row < chunk_rows; elim_row++) {
                    // Get the ratio for elimination
                    double scale = chunk[elim_row * m_dimension * 2 + elim_col];

                    // Reducing the lower half of the matrix to be 0
                    double *chunk_row = &chunk[elim_row * m_dimension * 2];
                    for (int col = elim_col; col < m_dimension * 2; col++) {
                        chunk_row[col] -= pivot_row[col] * scale;
                    }
                }

                // Backward elimination (Elimination for the upper half of the matrix)
                for (int elim_row = local_start_backward; elim_row >= 0; elim_row--) {
                    // Get the ratio for elimination
                    double scale = chunk[elim_row * m_dimension * 2 + elim_col];

                    // Reducing the upper half of the matrix to be 0
                    double *chunk_row = &chunk[elim_row * m_dimension * 2];
                    for (int col = elim_col; col < m_dimension * 2; col++) {
                        chunk_row[col] -= pivot_row[col] * scale;
                    }
                }
            }
        }
    }

    // Barrier to wait for all processes to finish the calculation
    MPI_Barrier(MPI_COMM_WORLD);

    // Gather the matrix chunks into a single matrix (cyclic)
    for (int i = 0; i < chunk_rows; i++) {
        MPI_Gather(chunk + i * m_dimension * 2, m_dimension * 2, MPI_DOUBLE,
                   matrix[i * num_processes], m_dimension * 2, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    // Print output to a text file
    if (rank == 0) {
        for (int i = 0; i < m_dimension; i++) {
            for (int j = 0; j < m_dimension; j++) {
                std::cout << matrix[i][m_dimension + j] << " ";
            }
            std::cout << std::endl;
        }
    }

    // Deallocate memory
    delete[] chunk;
    delete[] pivot_row;

    // Finalize MPI environment
    MPI_Finalize();

    return 0;
}