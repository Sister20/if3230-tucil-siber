class Matrix
{
    public:
        double *data;
        int rows;
        int cols;

        // Constructor
        Matrix(int rows, int cols) : rows(rows), cols(cols)
        {
            data = new double[rows * cols];
        }

        // Destructor
        ~Matrix()
        {
            delete[] data;
        }

        // Overload to access the matrix elements
        double *operator[](int row)
        {
            return data + row * cols;
        }
};