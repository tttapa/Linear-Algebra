#pragma once

#include <linalg/Arduino/ArduinoConfig.hpp>

#include <algorithm>  // std::fill, std::transform
#include <cassert>    // assert
#include <cmath>      // std::sqrt
#include <functional> // std::plus, std::minus
#include <numeric>    // std::inner_product
#include <utility>    // std::swap
#include <vector>     // std::vector

#ifndef NO_IOSTREAM_SUPPORT
#include <iosfwd> // std::ostream
#endif

#ifndef NO_RANDOM_SUPPORT
#include <random> // std::uniform_real_distribution
#endif

using std::size_t;

#include "util/MatrixStorage.hpp"

#ifndef COL_MAJ_ORDER
#define COL_MAJ_ORDER 1
#endif

/// @addtogroup MatVec
/// @{

#ifndef NO_ARDUINO_PRINT_SUPPORT
/// General matrix class.
class Matrix : public Printable {
#else
class Matrix {
#endif

    /// Container to store the elements of the matrix internally.
    using storage_t = util::storage_t<double>;

  protected:
    /// Convert raw storage to a matrix.
    explicit Matrix(storage_t &&storage, size_t rows, size_t cols);
    /// Convert raw storage to a matrix.
    explicit Matrix(const storage_t &storage, size_t rows, size_t cols);

  public:
    /// @name   Constructors and assignment
    /// @{

    /// Default constructor.
    Matrix() = default;

    /// Create a matrix of zeros with the given dimensions.
    Matrix(size_t rows, size_t cols);

    /// Create a matrix with the given values.
    Matrix(std::initializer_list<std::initializer_list<double>> init);
    /// Assign the given values to the matrix.
    Matrix &
    operator=(std::initializer_list<std::initializer_list<double>> init);

    /// Default copy constructor.
    Matrix(const Matrix &) = default;
    /// Move constructor.
    Matrix(Matrix &&);

    /// Default copy assignment.
    Matrix &operator=(const Matrix &) = default;
    /// Move assignment.
    Matrix &operator=(Matrix &&);

    /// @}

  public:
    /// @name   Matrix size
    /// @{

    /// Get the number of rows of the matrix.
    size_t rows() const { return rows_; }
    /// Get the number of columns of the matrix.
    size_t cols() const { return cols_; }
    /// Get the number of elements in the matrix:
    size_t num_elems() const { return storage.size(); }

    /// Reshape the matrix. The new size must have the same number of elements,
    /// and the result depends on the storage order (column major order or
    /// row major order).
    void reshape(size_t newrows, size_t newcols);
    /// Create a reshaped copy of the matrix.
    /// @see    @ref reshape
    Matrix reshaped(size_t newrows, size_t newcols) const;

    /// @}

  public:
    /// @name   Element access
    /// @{

    /// Get the element at the given position in the matrix.
    double &operator()(size_t row, size_t col);
    /// Get the element at the given position in the matrix.
    const double &operator()(size_t row, size_t col) const;

    /// Get the element at the given position in the linearized matrix.
    double &operator()(size_t index) { return storage[index]; }
    /// Get the element at the given position in the linearized matrix.
    const double &operator()(size_t index) const { return storage[index]; }

    /// @}

  public:
    /// @name   Memory management
    /// @{

    /// Set the number of rows and columns to zero, and deallocate the storage.
    void clear_and_deallocate();

    /// @}

  public:
    /// @name   Filling matrices
    /// @{

    /// Fill the matrix with a constant value.
    void fill(double value);

    /// Fill the matrix as an identity matrix (all zeros except the diagonal
    /// which is one).
    void fill_identity();

#ifndef NO_RANDOM_SUPPORT
    /// Fill the matrix with uniformly distributed random values.
    void fill_random(double min = 0, double max = 1,
                     std::default_random_engine::result_type seed =
                         std::default_random_engine::default_seed);
#endif

    /// @}

  public:
    /// @name   Create special matrices
    /// @{

    /// Create a matrix filled with ones.
    static Matrix ones(size_t rows, size_t cols);

    /// Create a matrix filled with zeros.
    static Matrix zeros(size_t rows, size_t cols);

    /// Create a matrix filled with a constant value.
    static Matrix constant(size_t rows, size_t cols, double value);

    /// Create an identity matrix.
    static Matrix identity(size_t rows, size_t cols);

    /// Create a square identity matrix.
    static Matrix identity(size_t rows);

#ifndef NO_RANDOM_SUPPORT
    /// Create a matrix with uniformly distributed random values.
    static Matrix random(size_t rows, size_t cols, double min = 0,
                         double max = 1,
                         std::default_random_engine::result_type seed =
                             std::default_random_engine::default_seed);
#endif

    /// @}

  public:
    /// @name   Swapping rows and columns
    /// @{

    /// Swap two rows of the matrix.
    void swap_rows(size_t a, size_t b);
    /// Swap two columns of the matrix.
    void swap_columns(size_t a, size_t b);

    /// @}

  public:
    /// @name   Comparison
    /// @{

    /// Check for equality of two matrices.
    /// @warning    Uses exact comparison, which is often not appropriate for
    ///             floating point numbers.
    bool operator==(const Matrix &other) const;
    /// Check for inequality of two matrices.
    /// @warning    Uses exact comparison, which is often not appropriate for
    ///             floating point numbers.
    bool operator!=(const Matrix &other) const { return !(*this == other); }

    /// @}

  public:
    /// @name   Matrix norms
    /// @{

    /// Compute the Frobenius norm of the matrix.
    double normFro() const &;
    /// Compute the Frobenius norm of the matrix.
    double normFro() &&;

    /// @}

  public:
    /// @name   Iterators
    /// @{

    /// Get the iterator to the first element of the matrix.
    storage_t::iterator begin() { return storage.begin(); }
    /// Get the iterator to the first element of the matrix.
    storage_t::const_iterator begin() const { return storage.begin(); }
    /// Get the iterator to the first element of the matrix.
    storage_t::const_iterator cbegin() const { return storage.begin(); }

    /// Get the iterator to the element past the end of the matrix.
    storage_t::iterator end() { return storage.end(); }
    /// Get the iterator to the element past the end of the matrix.
    storage_t::const_iterator end() const { return storage.end(); }
    /// Get the iterator to the element past the end of the matrix.
    storage_t::const_iterator cend() const { return storage.end(); }

    /// @}

  public:
    /// @name   Printing
    /// @{

#ifndef NO_IOSTREAM_SUPPORT
    /// Print a matrix.
    /// @param  os
    ///         The stream to print to.
    /// @param  precision
    ///         The number of significant figures to print.
    ///         (0 = auto)
    /// @param  width
    ///         The width of each element (number of characters).
    ///         (0 = auto)
    void print(std::ostream &os, uint8_t precision = 0,
               uint8_t width = 0) const;
#endif

#ifndef NO_ARDUINO_PRINT_SUPPORT
    /// Print a matrix
    /// @param  print
    ///         The printer to print to.
    /// @param  precision
    ///         The number of significant figures to print.
    ///         (0 = auto)
    /// @param  width
    ///         The width of each element (number of characters).
    ///         (0 = auto)
    void print(Print &print, uint8_t precision = 0, uint8_t width = 0) const;

    /// Implements the Arduino Printable interface.
    size_t printTo(Print &print) const override {
        this->print(print);
        return 0;
    }
#endif

    /// @}

  protected:
    size_t rows_ = 0, cols_ = 0;
    storage_t storage;

    friend class Vector;
    friend class RowVector;
};

// :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: //

/// A column vector (n×1 matrix).
class Vector : public Matrix {
  public:
    /// @name   Constructors and assignment
    /// @{

    /// Default constructor.
    Vector() = default;

    /// Create a column vector of the given size.
    Vector(size_t size) : Matrix(size, 1) {}

    /// Create a column vector from the given list of values.
    Vector(std::initializer_list<double> init) { *this = init; }

    /// Assign a list of values to the column vector.
    Vector &operator=(std::initializer_list<double> init);

    /// Convert an m×n matrix to a mn column vector.
    explicit Vector(const Matrix &matrix);
    /// Convert an m×n matrix to a mn column vector.
    explicit Vector(Matrix &&matrix);

    /// @}

  public:
    /// @name   Vector size
    /// @{

    /// Resize the vector.
    void resize(size_t size) { storage.resize(size); }

    /// Get the number of elements in the vector.
    size_t size() const { return num_elems(); }

    /// Reshaping a vector to a matrix requires an explicit cast.
    void reshape(size_t, size_t) = delete;
    /// Reshaping a vector to a matrix requires an explicit cast.
    Matrix reshaped(size_t, size_t) = delete;

    /// @}

  public:
    /// @name   Creating special vectors
    /// @{

    /// Create a vector filled with ones.
    static Vector ones(size_t size);
    /// Create a vector filled with zeros.
    static Vector zeros(size_t size);
    /// Create a vector filled with a constant value.
    static Vector constant(size_t size, double value);

#ifndef NO_RANDOM_SUPPORT
    /// Create a vector with uniformly distributed random values.
    static Vector random(size_t size, double min = 0, double max = 1,
                         std::default_random_engine::result_type seed =
                             std::default_random_engine::default_seed);
#endif

    /// @}

  public:
    /// @name Dot products
    /// @{

    /// Compute the dot product of two vectors. Reinterprets matrices as
    /// vectors.
    static double dot_unchecked(const Matrix &a, const Matrix &b);
    /// Compute the dot product of two vectors. Reinterprets matrices as
    /// vectors.
    static double dot_unchecked(Matrix &&a, const Matrix &b);
    /// Compute the dot product of two vectors. Reinterprets matrices as
    /// vectors.
    static double dot_unchecked(const Matrix &a, Matrix &&b);
    /// Compute the dot product of two vectors. Reinterprets matrices as
    /// vectors.
    static double dot_unchecked(Matrix &&a, Matrix &&b);

    /// Compute the dot product of two vectors.
    static double dot(const Vector &a, const Vector &b);
    /// Compute the dot product of two vectors.
    static double dot(Vector &&a, const Vector &b);
    /// Compute the dot product of two vectors.
    static double dot(const Vector &a, Vector &&b);
    /// Compute the dot product of two vectors.
    static double dot(Vector &&a, Vector &&b);

    /// Compute the dot product of this vector with another vector.
    double dot(const Vector &b) const & { return dot(*this, b); }
    /// Compute the dot product of this vector with another vector.
    double dot(const Vector &b) && { return dot(std::move(*this), b); }
    /// Compute the dot product of this vector with another vector.
    double dot(Vector &&b) const & { return dot(*this, std::move(b)); }
    /// Compute the dot product of this vector with another vector.
    double dot(Vector &&b) && { return dot(std::move(*this), std::move(b)); }

    /// @}

  public:
    /// @name Cross products
    /// @{

    /// Compute the cross product of two 3-vectors, overwriting the first vector
    /// with the result. Reinterprets matrices as vectors (so it can be used
    /// with row vectors as well).
    static void cross_inplace_unchecked(Matrix &a, const Matrix &b);
    /// Compute the opposite of the cross product of two 3-vectors, overwriting
    /// the first vector with the result. Reinterprets matrices as vectors
    /// (so it can be used with row vectors as well).
    static void cross_inplace_unchecked_neg(Matrix &a, const Matrix &b);

    /// Compute the cross product of two 3-vectors, overwriting the first vector
    /// with the result.
    static void cross_inplace(Vector &a, const Vector &b);
    /// Compute the cross product of two 3-vectors, overwriting the first vector
    /// with the result.
    static void cross_inplace(Vector &a, Vector &&b);
    /// Compute the opposite of the cross product of two 3-vectors, overwriting
    /// the first vector with the result.
    static void cross_inplace_neg(Vector &a, const Vector &b);
    /// Compute the opposite of the cross product of two 3-vectors, overwriting
    /// the first vector with the result.
    static void cross_inplace_neg(Vector &a, Vector &&b);

    /// Compute the cross product of two 3-vectors.
    static Vector cross(const Vector &a, const Vector &b);
    /// Compute the cross product of two 3-vectors.
    static Vector &&cross(Vector &&a, const Vector &b);
    /// Compute the cross product of two 3-vectors.
    static Vector &&cross(const Vector &a, Vector &&b);
    /// Compute the cross product of two 3-vectors.
    static Vector &&cross(Vector &&a, Vector &&b);

    /// Compute the cross product of this 3-vector with another 3-vector.
    Vector cross(const Vector &b) const & { return cross(*this, b); }
    /// Compute the cross product of this 3-vector with another 3-vector,
    Vector &&cross(const Vector &b) && { return cross(std::move(*this), b); }
    /// Compute the cross product of this 3-vector with another 3-vector,
    Vector &&cross(Vector &&b) const & { return cross(*this, std::move(b)); }
    /// Compute the cross product of this 3-vector with another 3-vector,
    Vector &&cross(Vector &&b) && {
        return cross(std::move(*this), std::move(b));
    }

    /// @}

  public:
    /// @name   Vector norms
    /// @{

    /// Compute the 2-norm of the vector.
    double norm2() const &;
    /// Compute the 2-norm of the vector.
    double norm2() &&;

    /// @}
};

/// A row vector (1×n matrix).
class RowVector : public Matrix {
  public:
    /// @name   Constructors and assignment
    /// @{

    /// Default constructor.
    RowVector() = default;

    /// Create a row vector of the given size.
    RowVector(size_t size) : Matrix(1, size) {}

    /// Create a row vector from the given list of values.
    RowVector(std::initializer_list<double> init) { *this = init; }

    /// Assign a list of values to the column vector.
    RowVector &operator=(std::initializer_list<double> init);

    /// Convert an m×n matrix to a mn row vector.
    explicit RowVector(const Matrix &matrix);
    /// Convert an m×n matrix to a mn row vector.
    explicit RowVector(Matrix &&matrix);

    /// @}

  public:
    /// @name   Vector size
    /// @{

    /// Resize the vector.
    void resize(size_t size) { storage.resize(size); }

    /// Get the number of elements in the vector.
    size_t size() const { return num_elems(); }

    /// Reshaping a vector to a matrix requires an explicit cast.
    void reshape(size_t, size_t) = delete;
    /// Reshaping a vector to a matrix requires an explicit cast.
    Matrix reshaped(size_t, size_t) = delete;

    /// @}

  public:
    /// @name   Creating special row vectors
    /// @{

    /// Create a row vector filled with ones.
    static RowVector ones(size_t size);
    /// Create a row vector filled with zeros.
    static RowVector zeros(size_t size);
    /// Create a row vector filled with a constant value.
    static RowVector constant(size_t size, double value);

#ifndef NO_RANDOM_SUPPORT
    /// Create a row vector with uniformly distributed random values.
    static RowVector random(size_t size, double min = 0, double max = 1,
                            std::default_random_engine::result_type seed =
                                std::default_random_engine::default_seed);
#endif

    /// @}

  public:
    /// @name   Dot products
    /// @{

    /// Compute the dot product of two vectors.
    static double dot(const RowVector &a, const RowVector &b);
    /// Compute the dot product of two vectors.
    static double dot(RowVector &&a, const RowVector &b);
    /// Compute the dot product of two vectors.
    static double dot(const RowVector &a, RowVector &&b);
    /// Compute the dot product of two vectors.
    static double dot(RowVector &&a, RowVector &&b);

    /// Compute the dot product of this vector with another vector.
    double dot(const RowVector &b) const & { return dot(*this, b); }
    /// Compute the dot product of this vector with another vector.
    double dot(const RowVector &b) && { return dot(std::move(*this), b); }
    /// Compute the dot product of this vector with another vector.
    double dot(RowVector &&b) const & { return dot(*this, std::move(b)); }
    /// Compute the dot product of this vector with another vector.
    double dot(RowVector &&b) && { return dot(std::move(*this), std::move(b)); }

    /// @}

  public:
    /// @name Cross products
    /// @{

    /// Compute the cross product of two 3-vectors, overwriting the first vector
    /// with the result.
    static void cross_inplace(RowVector &a, const RowVector &b);
    /// Compute the cross product of two 3-vectors, overwriting the first vector
    /// with the result.
    static void cross_inplace(RowVector &a, RowVector &&b);
    /// Compute the opposite of the cross product of two 3-vectors, overwriting
    /// the first vector with the result.
    static void cross_inplace_neg(RowVector &a, const RowVector &b);
    /// Compute the opposite of the cross product of two 3-vectors, overwriting
    /// the first vector with the result.
    static void cross_inplace_neg(RowVector &a, RowVector &&b);

    /// Compute the cross product of two 3-vectors.
    static RowVector cross(const RowVector &a, const RowVector &b);
    /// Compute the cross product of two 3-vectors.
    static RowVector &&cross(RowVector &&a, const RowVector &b);
    /// Compute the cross product of two 3-vectors.
    static RowVector &&cross(const RowVector &a, RowVector &&b);
    /// Compute the cross product of two 3-vectors.
    static RowVector &&cross(RowVector &&a, RowVector &&b);

    /// Compute the cross product of this 3-vector with another 3-vector.
    RowVector cross(const RowVector &b) const &;
    /// Compute the cross product of this 3-vector with another 3-vector,
    RowVector &&cross(const RowVector &b) &&;
    /// Compute the cross product of this 3-vector with another 3-vector,
    RowVector &&cross(RowVector &&b) const &;
    /// Compute the cross product of this 3-vector with another 3-vector,
    RowVector &&cross(RowVector &&b) &&;

    /// @}

  public:
    /// @name   Vector norms
    /// @{

    /// Compute the 2-norm of the vector.
    double norm2() const &;
    /// Compute the 2-norm of the vector.
    double norm2() &&;

    /// @}
};

/// Square matrix class.
class SquareMatrix : public Matrix {
  public:
    /// @name   Constructors and assignment
    /// @{

    /// Default constructor.
    SquareMatrix() = default;

    /// Create a square matrix of zeros.
    SquareMatrix(size_t size) : Matrix(size, size) {}

    /// Create a square matrix with the given values.
    SquareMatrix(std::initializer_list<std::initializer_list<double>> init);

    /// Convert a general matrix to a square matrix
    explicit SquareMatrix(Matrix &&matrix);
    /// Convert a general matrix to a square matrix
    explicit SquareMatrix(const Matrix &matrix);

    /// Assign the given values to the square matrix.
    SquareMatrix &
    operator=(std::initializer_list<std::initializer_list<double>> init);

    /// @}

  public:
    /// @name   Matrix size
    /// @{

    /// Reshaping a square matrix to a general matrix requires an explicit cast.
    void reshape(size_t, size_t) = delete;
    /// Reshaping a square matrix to a general matrix requires an explicit cast.
    Matrix reshaped(size_t, size_t) = delete;

    /// @}

  public:
    /// @name   Transposition
    /// @{

    /// Transpose the matrix in-place.
    static void transpose_inplace(Matrix &A);
    /// Transpose the matrix in-place.
    void transpose_inplace() { transpose_inplace(*this); }

    /// @}

  public:
    /// @name   Creating special matrices
    /// @{

    /// Create a square matrix filled with ones.
    static SquareMatrix ones(size_t rows);
    /// Create a square matrix filled with zeros.
    static SquareMatrix zeros(size_t rows);
    /// Create a square matrix filled with a constant value.
    static SquareMatrix constant(size_t rows, double value);

    /// Create a square identity matrix.
    static SquareMatrix identity(size_t rows);

#ifndef NO_RANDOM_SUPPORT
    /// Create a matrix with uniformly distributed random values.
    static SquareMatrix random(size_t rows, double min = 0, double max = 1,
                               std::default_random_engine::result_type seed =
                                   std::default_random_engine::default_seed);
#endif

    /// @}
};

/// @}

#ifndef NO_IOSTREAM_SUPPORT
/// Print a matrix.
/// @related    Matrix
std::ostream &operator<<(std::ostream &os, const Matrix &M);
#endif

#ifndef NO_ARDUINO_PRINT_SUPPORT
/// Print a matrix.
/// @related    Matrix
Print &operator<<(Print &p, const Matrix &M);
#endif

// :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: //

/// @addtogroup MatVecOp
/// @{

/// @defgroup MatMul    Matrix multiplication
/// @brief   Matrix-matrix, matrix-vector and vector-vector multiplication.
/// @{

/// Matrix multiplication.
Matrix operator*(const Matrix &A, const Matrix &B);
/// Matrix multiplication.
Matrix operator*(Matrix &&A, const Matrix &B);
/// Matrix multiplication.
Matrix operator*(const Matrix &A, Matrix &&B);
/// Matrix multiplication.
Matrix operator*(Matrix &&A, Matrix &&B);

/// Square matrix multiplication.
SquareMatrix operator*(const SquareMatrix &A, const SquareMatrix &B);
/// Square matrix multiplication.
SquareMatrix operator*(SquareMatrix &&A, const SquareMatrix &B);
/// Square matrix multiplication.
SquareMatrix operator*(const SquareMatrix &A, SquareMatrix &&B);
/// Square matrix multiplication.
SquareMatrix operator*(SquareMatrix &&A, SquareMatrix &&B);

/// Matrix-vector multiplication.
Vector operator*(const Matrix &A, const Vector &b);
/// Matrix-vector multiplication.
Vector operator*(Matrix &&A, const Vector &b);
/// Matrix-vector multiplication.
Vector operator*(const Matrix &A, Vector &&b);
/// Matrix-vector multiplication.
Vector operator*(Matrix &&A, Vector &&b);

/// Matrix-vector multiplication.
RowVector operator*(const RowVector &a, const Matrix &B);
/// Matrix-vector multiplication.
RowVector operator*(RowVector &&a, const Matrix &B);
/// Matrix-vector multiplication.
RowVector operator*(const RowVector &a, Matrix &&B);
/// Matrix-vector multiplication.
RowVector operator*(RowVector &&a, Matrix &&B);

/// Vector-vector multiplication.
double operator*(const RowVector &a, const Vector &b);
/// Vector-vector multiplication.
double operator*(RowVector &&a, const Vector &b);
/// Vector-vector multiplication.
double operator*(const RowVector &a, Vector &&b);
/// Vector-vector multiplication.
double operator*(RowVector &&a, Vector &&b);

/// @}

// -------------------------------------------------------------------------- //

/// @defgroup   MatAdd  Addition
/// @brief  Matrix and vector addition
/// @{

/// Matrix addition.
Matrix operator+(const Matrix &A, const Matrix &B);

void operator+=(Matrix &A, const Matrix &B);
Matrix &&operator+(Matrix &&A, const Matrix &B);
Matrix &&operator+(const Matrix &A, Matrix &&B);
Matrix &&operator+(Matrix &&A, Matrix &&B);
Vector &&operator+(Vector &&a, const Vector &b);
Vector &&operator+(const Vector &a, Vector &&b);
Vector &&operator+(Vector &&a, Vector &&b);
RowVector &&operator+(RowVector &&a, const RowVector &b);
RowVector &&operator+(const RowVector &a, RowVector &&b);
RowVector &&operator+(RowVector &&a, RowVector &&b);
SquareMatrix &&operator+(SquareMatrix &&a, const SquareMatrix &b);
SquareMatrix &&operator+(const SquareMatrix &a, SquareMatrix &&b);
SquareMatrix &&operator+(SquareMatrix &&a, SquareMatrix &&b);
Vector operator+(const Vector &a, const Vector &b);
RowVector operator+(const RowVector &a, const RowVector &b);
SquareMatrix operator+(const SquareMatrix &a, const SquareMatrix &b);

/// @}

// -------------------------------------------------------------------------- //

/// @defgroup   MatSub  Subtraction
/// @brief  Matrix and vector subtraction
/// @{

/// Matrix subtraction.
Matrix operator-(const Matrix &A, const Matrix &B);
void operator-=(Matrix &A, const Matrix &B);
Matrix &&operator-(Matrix &&A, const Matrix &B);
Matrix &&operator-(const Matrix &A, Matrix &&B);
Matrix &&operator-(Matrix &&A, Matrix &&B);
Vector &&operator-(Vector &&a, const Vector &b);
Vector &&operator-(const Vector &a, Vector &&b);
Vector &&operator-(Vector &&a, Vector &&b);
RowVector &&operator-(RowVector &&a, const RowVector &b);
RowVector &&operator-(const RowVector &a, RowVector &&b);
RowVector &&operator-(RowVector &&a, RowVector &&b);
SquareMatrix &&operator-(SquareMatrix &&a, const SquareMatrix &b);
SquareMatrix &&operator-(const SquareMatrix &a, SquareMatrix &&b);
SquareMatrix &&operator-(SquareMatrix &&a, SquareMatrix &&b);
Vector operator-(const Vector &a, const Vector &b);
RowVector operator-(const RowVector &a, const RowVector &b);
SquareMatrix operator-(const SquareMatrix &a, const SquareMatrix &b);

/// @}

// -------------------------------------------------------------------------- //

/// @defgroup   MatNeg  Negation
/// @brief  Matrix and vector negation
/// @{

/// Matrix negation.
Matrix operator-(const Matrix &A);
Matrix &&operator-(Matrix &&A);
Vector &&operator-(Vector &&a);
RowVector &&operator-(RowVector &&a);
SquareMatrix &&operator-(SquareMatrix &&a);
Vector operator-(const Vector &a);
RowVector operator-(const RowVector &a);
SquareMatrix operator-(const SquareMatrix &a);

/// @}

// -------------------------------------------------------------------------- //

/// @defgroup   ScalMul Scalar multiplication
/// @brief  Multiplication by a scalar
/// @{

/// Scalar multiplication.
Matrix operator*(const Matrix &A, double s);
void operator*=(Matrix &A, double s);
Matrix &&operator*(Matrix &&A, double s);
Vector operator*(const Vector &a, double s);
RowVector operator*(const RowVector &a, double s);
SquareMatrix operator*(const SquareMatrix &a, double s);
Vector &&operator*(Vector &&a, double s);
RowVector &&operator*(RowVector &&a, double s);
SquareMatrix &&operator*(SquareMatrix &&a, double s);

Matrix operator*(double s, const Matrix &A);
Matrix &&operator*(double s, Matrix &&A);
Vector operator*(double s, const Vector &a);
RowVector operator*(double s, const RowVector &a);
SquareMatrix operator*(double s, const SquareMatrix &a);
Vector &&operator*(double s, Vector &&a);
RowVector &&operator*(double s, RowVector &&a);
SquareMatrix &&operator*(double s, SquareMatrix &&a);

/// @}

// -------------------------------------------------------------------------- //

/// @defgroup   ScalDiv Scalar division
/// @brief  Division by a scalar
/// @{

/// Scalar division.
Matrix operator/(const Matrix &A, double s);
void operator/=(Matrix &A, double s);
Matrix &&operator/(Matrix &&A, double s);
Vector operator/(const Vector &a, double s);
RowVector operator/(const RowVector &a, double s);
SquareMatrix operator/(const SquareMatrix &a, double s);
Vector &&operator/(Vector &&a, double s);
RowVector &&operator/(RowVector &&a, double s);
SquareMatrix &&operator/(SquareMatrix &&a, double s);

/// @}

/// @}

// -------------------------------------------------------------------------- //

/// @addtogroup MatVecOp
/// @{

/// @defgroup   MatTrans    Transposition
/// @brief  Matrix and vector transposition
/// @{

/// Matrix transpose for general matrices.
Matrix explicit_transpose(const Matrix &in);

/// Matrix transpose for rectangular or square matrices and row or column
/// vectors.
Matrix transpose(const Matrix &in);
/// Matrix transpose for rectangular or square matrices and row or column
/// vectors.
Matrix &&transpose(Matrix &&in);

/// Square matrix transpose.
SquareMatrix transpose(const SquareMatrix &in);
/// Square matrix transpose.
SquareMatrix &&transpose(SquareMatrix &&in);

/// Vector transpose.
RowVector transpose(const Vector &in);
/// Vector transpose.
RowVector transpose(Vector &&in);
/// Vector transpose.
Vector transpose(const RowVector &in);
/// Vector transpose.
Vector transpose(RowVector &&in);

/// @}

/// @}
