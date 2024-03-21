#include <algorithm>
#include <iostream>
#include <memory>

#include "mpi.h"

int main(int argc, char *argv[])
{
    // Initialize MPI
    MPI_Init(&argc, &argv);

    // Get the total number of processes
    int num_processes;
    MPI_Comm_size(MPI_COMM_WORLD, &num_processes);

    // Get current rank (Process ID)
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Get the dimension of the matrix
    int dim;
    if (rank == 0)
    {
        std::cin >> dim;
    }

    // Broadcast the dimension of the matrix to all processes
    MPI_Bcast(&dim, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Calculate the number of rows for each matrix chunk
    const int chunk_rows = dim / num_processes;

    // Read matrix from text file
    std::unique_ptr<double[]> matrix;
    if (rank == 0)
    {
        matrix = std::make_unique<double[]>((dim * 2) * dim);
        for (int i = 0; i < dim; i++)
        {
            for (int j = 0; j < dim; j++)
            {
                std::cin >> matrix[i * (dim * 2) + j];
            }

            // Append the identity matrix
            matrix[i * (dim * 2) + dim + i] = 1.0;
        }
    }

    // Scatter the matrix into chunks (cyclic)
    auto chunk = std::make_unique<double[]>(dim * chunk_rows * 2);
    for (int i = 0; i < chunk_rows; i++)
    {
        MPI_Scatter(matrix.get() + i * (dim * 2) * num_processes, dim * 2, MPI_DOUBLE,
                    chunk.get() + i * dim * 2, dim * 2, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    // Allocate memory to store pivot row sent by other processes
    auto pivot_row = std::make_unique<double[]>(dim * 2);

    // Calculate the matrix inverse with gauss-jordan
    for (int row = 0; row < chunk_rows; row++)
    {
        // Assign each iteration (current rank) to the belonging process
        for (int current_rank = 0; current_rank < num_processes; current_rank++)
        {
            // Calculate the column of the currently evaluated pivot
            int elim_col = row * num_processes + current_rank;

            // Only execute matrix calculation if the current_rank is equal to the belonging rank
            if (rank == current_rank)
            {
                // Transform the matrix into a unit matrix
                double pivot = chunk[row * dim * 2 + elim_col];
                for (int col = elim_col; col < dim * 2; col++)
                {
                    chunk[row * dim * 2 + col] /= pivot;
                }

                // Send the pivot row to the other ranks
                MPI_Bcast(chunk.get() + row * dim * 2, dim * 2, MPI_DOUBLE, current_rank, MPI_COMM_WORLD);

                // Forward elimination (Elimination for the lower half of the matrix)
                for (int elim_row = row + 1; elim_row < chunk_rows; elim_row++)
                {
                    // Get the ratio for elimination
                    double scale = chunk[elim_row * dim * 2 + elim_col];

                    // Reducing the lower half of the matrix to be 0
                    for (int col = elim_col; col < dim * 2; col++)
                    {
                        chunk[elim_row * dim * 2 + col] -= chunk[row * dim * 2 + col] * scale;
                    }
                }

                // Barrier to wait for all processes to finish the forward elimination
                MPI_Barrier(MPI_COMM_WORLD);

                // Backward elimination (Elimination for the upper half of the matrix)
                for (int elim_row = row - 1; elim_row >= 0; elim_row--)
                {
                    // Get the ratio for elimination
                    double scale = chunk[elim_row * dim * 2 + elim_col];

                    // Reducing the upper half of the matrix to be 0
                    for (int col = elim_col; col < dim * 2; col++)
                    {
                        chunk[elim_row * dim * 2 + col] -= chunk[row * dim * 2 + col] * scale;
                    }
                }
            }
            else
            {
                // Receive the pivot from the sending current_rank
                MPI_Bcast(pivot_row.get(), dim * 2, MPI_DOUBLE, current_rank, MPI_COMM_WORLD);

                int local_start = (rank < current_rank) ? row + 1 : row;

                // Forward elimination (Elimination for the lower half of the matrix)
                for (int elim_row = local_start; elim_row < chunk_rows; elim_row++)
                {
                    // Get the ratio for elimination
                    double scale = chunk[elim_row * dim * 2 + elim_col];

                    // Reducing the lower half of the matrix to be 0
                    for (int col = elim_col; col < dim * 2; col++)
                    {
                        chunk[elim_row * dim * 2 + col] -= pivot_row[col] * scale;
                    }
                }

                // Barrier to wait for all processes to finish the forward elimination
                MPI_Barrier(MPI_COMM_WORLD);

                // Backward elimination (Elimination for the upper half of the matrix)
                for (int elim_row = row; elim_row >= 0; elim_row--)
                {
                    // Get the ratio for elimination
                    double scale = chunk[elim_row * dim * 2 + elim_col];

                    // Reducing the upper half of the matrix to be 0
                    for (int col = elim_col; col < dim * 2; col++)
                    {
                        chunk[elim_row * dim * 2 + col] -= pivot_row[col] * scale;
                    }
                }
            }
        }
    }

    // Gather the matrix chunks into a single matrix (cyclic)
    for (int i = 0; i < chunk_rows; i++)
    {
        MPI_Gather(chunk.get() + i * dim * 2, dim * 2, MPI_DOUBLE,
                   matrix.get() + i * (dim * 2) * num_processes, dim * 2, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    // Print output to a text file
    if (rank == 0)
    {
        for (int i = 0; i < dim; i++)
        {
            for (int j = 0; j < dim; j++)
            {
                std::cout << matrix[i * (dim * 2) + dim + j] << " ";
            }
            std::cout << std::endl;
        }
    }

    // Finalize MPI environment
    MPI_Finalize();

    return 0;
}


