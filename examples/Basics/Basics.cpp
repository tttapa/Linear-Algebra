/**
 * @example Basics.cpp
 * 
 * Demonstrates basic matrix and vector operations using the library.
 * 
 * ***
 * 
 * Expected output:
 * 
 * ~~~
 * A = 
 *              11             12             13
 *              21             22             23
 *              31             32             33
 * 
 * Aᵀ = 
 *              11             21             31
 *              12             22             32
 *              13             23             33
 * 
 * AᵀA = 
 *            1523           1586           1649
 *            1586           1652           1718
 *            1649           1718           1787
 * 
 * B = 
 *         15.3692        8.42271        22.3086
 *          21.827        10.3061         22.406
 *         36.6208        31.6117         39.846
 * 
 * A(1, 2) = 23
 * Dimensions of C: 2×3
 * Number of elements of C: 6
 * 
 * D = 
 *               0              3              6              9
 *               1              4              7             10
 *               2              5              8             11
 * 
 * Sum of elements of D = 66
 * v = 
 *               1
 *               2
 *               3
 * 
 * a   =               4              6              5
 * b   =               1              2              3
 * a×b =               8             -7              2
 * a·b =              31
 * 
 * v(2) = 3
 * 
 * vvᵀ = 
 *               1              2              3
 *               2              4              6
 *               3              6              9
 * 
 * vᵀv = 14
 * ~~~
 */

#include <iomanip>  // std::setw
#include <iostream> // std::cout
#include <linalg/Matrix.hpp>

void matrix_operations() {
    // Initialize a matrix with an initializer list:
    Matrix A = {
        {11, 12, 13},
        {21, 22, 23},
        {31, 32, 33},
    };
    // Print a matrix:
    std::cout << "A = \n" << A << std::endl;

    // Transposing a matrix:
    Matrix AT = transpose(A);
    std::cout << "Aᵀ = \n" << AT << std::endl;
    // Matrix multiplication:
    std::cout << "AᵀA = \n" << AT * A << std::endl;

    // Creating special matrices:
    Matrix E = Matrix::ones(3, 2);  // 3×2 (rows×columns) matrix of all ones
    Matrix O = Matrix::zeros(2, 2); // 2×2 matrix of all zeros
    Matrix C = Matrix::constant(2, 3, 42.42);  // 2×3 all elements are 42.42
    Matrix I = Matrix::identity(3);            // 3×3 identity matrix
    Matrix R = Matrix::random(3, 3, -10, +10); // 3×3 uniform random in [-10,10]

    // Adding, subtracting, negation, scalar multiplication, etc.
    Matrix B = A + I * (-R) - 3 * I + E * O * C / 3.14;
    std::cout << "B = \n" << B << std::endl;

    // Element access:
    std::cout << "A(1, 2) = " << A(1, 2) << std::endl;
    // (row, column), indices are zero-based.

    // Matrix size:
    std::cout << "Dimensions of C: " << C.rows() << "×" << C.cols() << "\n"
              << "Number of elements of C: " << C.num_elems() << "\n"
              << std::endl;

    // Creating a matrix with a given size:
    Matrix D(3, 4); // Equivalent to `Matrix D = Matrix::zeros(3, 4)`

    // Iterators:
    std::iota(std::begin(D), std::end(D), 0.);
    std::cout << "D = \n" << D << std::endl;
    // Iterators go column-by-column (column major order).
    double D_sum = std::accumulate(std::begin(D), std::end(D), 0.);
    std::cout << "Sum of elements of D = " << D_sum << std::endl;
}

void vector_operations() {
    // Vectors can be initialized just like matrices:
    Vector v = {1, 2, 3}; // Column vector (3×1)
    std::cout << "v = \n" << v << std::endl;
    RowVector a = {4, 6, 5}; // Row vector (1×3)

    // Transpose from column to row vector:
    RowVector b = transpose(v);

    // Dot and cross products:
    std::cout << "a   = " << a          //
              << "b   = " << b          //
              << "a×b = " << a.cross(b) //
              << "a·b = " << std::setw(15) << a.dot(b) << "\n"
              << std::endl;

    // Element access:
    std::cout << "v(2) = " << v(2) << "\n" << std::endl;
    // indices are zero-based.

    // Rank-1 multiplication:
    Matrix V = v * transpose(v);
    std::cout << "vvᵀ = \n" << V << std::endl;
    // Dot product:
    double d = transpose(v) * v;
    std::cout << "vᵀv = " << d << "\n" << std::endl;
}

int main() {
    matrix_operations(); 
    vector_operations();
}