// Transform the matrix into a unit matrix
void transformToUnitMatrix(double *chunk, int row, int dim, int elim_col)
{
  int pivot_index = row * dim * 2 + elim_col;
  double pivot = chunk[pivot_index];
  for (int col = elim_col; col < dim * 2; col++)
  {
    chunk[row * dim * 2 + col] /= pivot;
  }
}

// Forward elimination (Elimination for the lower half of the matrix)
void forwardElimination(double *chunk, int dim, int row, int chunk_rows, int elim_col)
{
  for (int elim_row = row + 1; elim_row < chunk_rows; elim_row++)
  {
    // Get the ratio for elimination
    double ratio = chunk[elim_row * dim * 2 + elim_col];

    // Reducing the lower half of the matrix to be 0
    for (int col = elim_col; col < dim * 2; col++)
    {
      chunk[elim_row * dim * 2 + col] -= chunk[row * dim * 2 + col] * ratio;
    }
  }
}

// Backward elimination (Elimination for the upper half of the matrix)
void backwardElimination(double *chunk, int dim, int row, int elim_col) {
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