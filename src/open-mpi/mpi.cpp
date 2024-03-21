#include <algorithm>
#include <iostream>
#include <memory>
#include "mpi.h"

int main(int argc, char *argv[])
{
    // Initialize the MPI environment
    MPI_Init(&argc, &argv);

    // Get the number of processes
    int num_tasks;
    MPI_Comm_size(MPI_COMM_WORLD, &num_tasks);

    // Get the rank of the process
    int task_id;
    MPI_Comm_rank(MPI_COMM_WORLD, &task_id);

    // Get dimension of matrix
    int dim;
    if (task_id == 0)
    {
        std::cin >> dim;
    }

    // Broadcast the dimension of the matrix
    MPI_Bcast(&dim, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Calculate the number of rows each task will handle
    const int n_rows = dim / num_tasks;

    // Variable to store the chunk of the matrix
    auto m_chunk = std::make_unique<double[]>(dim * n_rows * 2);

    // Read in the matrix
    std::unique_ptr<double[]> matrix;
    if (task_id == 0)
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

    // Partial pivoting
    if (task_id == 0)
    {
        double d = 0.0;
        for (int i = dim; i > 1; --i)
        {
            if (matrix[(i - 1) * (dim * 2) + 1] < matrix[i * (dim * 2) + 1])
            {
                for (int j = 0; j < dim * 2; ++j)
                {
                    d = matrix[i * (dim * 2) + j];
                    matrix[i * (dim * 2) + j] = matrix[(i - 1) * (dim * 2) + j];
                    matrix[(i - 1) * (dim * 2) + j] = d;
                }
            }
        }
    }

    // Exexcution time
    double start;
    if (task_id == 0)
    {
        start = MPI_Wtime();
    }

    // Scatter the matrix into chunks
    for (int i = 0; i < n_rows; i++)
    {
        MPI_Scatter(matrix.get() + i * (dim * 2) * num_tasks, dim * 2, MPI_DOUBLE,
                    m_chunk.get() + i * dim * 2, dim * 2, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    // TODO: Implement the matrix inverse with gauss-jordan
    // for (int row = 0; row < n_rows; row++)
    // {
    //     for (int rank = 0; rank < num_tasks; rank++)
    //     {
    //         auto global_col = row * num_tasks + rank;
    //         if (rank == task_id)
    //         {
    //             auto pivot = m_chunk[row * dim * 2 + global_col];
    //             for (int col = global_col; col < dim * 2; col++)
    //             {
    //                 m_chunk[row * dim * 2 + col] /= pivot;
    //             }
    //         }
    //     }
    // }

    // Gather the matrix
    for (int i = 0; i < n_rows; i++)
    {
        MPI_Gather(m_chunk.get() + i * dim * 2, dim * 2, MPI_DOUBLE,
                   matrix.get() + i * (dim * 2) * num_tasks, dim * 2, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    // Output
    if (task_id == 0)
    {
        double end = MPI_Wtime();
        std::cout << "Time: " << end - start << '\n';
        // print the matrix
        for (int i = 0; i < dim; i++)
        {
            for (int j = 0; j < dim * 2; j++)
            {
                std::cout << matrix[i * (dim * 2) + j] << ' ';
            }
            std::cout << '\n';
        }
    }

    // Finalize the MPI environment.

    MPI_Finalize();

    return 0;
}