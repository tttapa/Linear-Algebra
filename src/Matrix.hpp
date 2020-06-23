#pragma once

#include <util/ArduinoMacroFix.hpp>

#include <algorithm>  // std::fill, std::transform
#include <cassert>    // assert
#include <functional> // std::plus, std::minus
#include <iosfwd>     // std::ostream
#include <numeric>    // std::inner_product
#include <random>     // std::uniform_real_distribution
#include <utility>    // std::swap
#include <vector>     // std::vector

using std::size_t;

#ifndef COL_MAJ_ORDER
#define COL_MAJ_ORDER 1
#endif

#ifdef MATRIX_COUNT_ALLOCATIONS
#include <util/CountingAllocator.hpp>
#endif

/// @addtogroup MatVec
/// @{

/// General matrix class.
class Matrix {

#ifdef MATRIX_COUNT_ALLOCATIONS
    using storage_t = std::vector<double, CountingAllocator<double>>;
#else
    /// Container to store the elements of the matrix internally.
    using storage_t = std::vector<double>;
#endif

  protected:
    /// Convert raw storage to an n×1 matrix (column vector) or a 1×n matrix
    /// (row vector).
    explicit Matrix(storage_t &&storage, bool column = true)
        : rows_(column ? storage.size() : 1),
          cols_(column ? 1 : storage.size()), storage(std::move(storage)) {}
    /// Convert raw storage to an n×1 matrix (column vector) or a 1×n matrix
    /// (row vector).
    explicit Matrix(const storage_t &storage, bool column = true)
        : rows_(column ? storage.size() : 1),
          cols_(column ? 1 : storage.size()), storage(storage) {}
    /// Convert raw storage to a matrix.
    explicit Matrix(storage_t &&storage, size_t rows, size_t cols)
        : rows_(rows), cols_(cols), storage(std::move(storage)) {}
    /// Convert raw storage to a matrix.
    explicit Matrix(const storage_t &storage, size_t rows, size_t cols)
        : rows_(rows), cols_(cols), storage(storage) {}

  public:
    /// @name   Constructors and assignment
    /// @{

    /// Default constructor.
    Matrix() = default;

    /// Create a matrix of zeros with the given dimensions.
    Matrix(size_t rows, size_t cols)
        : rows_(rows), cols_(cols), storage(rows * cols) {}

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
    void reshape(size_t newrows, size_t newcols) {
        assert(newrows * newcols == rows() * cols());
        this->rows_ = newrows;
        this->cols_ = newcols;
    }
    /// Create a reshaped copy of the matrix.
    /// @see    @ref reshape
    Matrix reshaped(size_t newrows, size_t newcols) const {
        Matrix result = *this;
        result.reshape(newrows, newcols);
        return result;
    }

    /// @}

  public:
    /// @name   Element access
    /// @{

    /// Get the element at the given position in the matrix.
    double &operator()(size_t row, size_t col) {
#if COL_MAJ_ORDER == 1
        return storage[row + rows_ * col];
#else
        return storage[row * cols_ + col];
#endif
    }
    /// Get the element at the given position in the matrix.
    const double &operator()(size_t row, size_t col) const {
#if COL_MAJ_ORDER == 1
        return storage[row + rows_ * col];
#else
        return storage[row * cols_ + col];
#endif
    }

    /// Get the element at the given position in the linearized matrix.
    double &operator()(size_t index) { return storage[index]; }
    /// Get the element at the given position in the linearlized matrix.
    const double &operator()(size_t index) const { return storage[index]; }

    /// @}

  public:
    /// @name   Memory management
    /// @{

    /// Set the number of rows and columns to zero, and deallocate the storage.
    void clear_and_deallocate() {
        this->rows_ = 0;
        this->cols_ = 0;
        storage_t().swap(this->storage); // replace storage with empty storage
        // temporary storage goes out of scope and deallocates original storage
    }

    /// @}

  public:
    /// @name   Filling matrices
    /// @{

    /// Fill the matrix with a constant value.
    void fill(double value) {
        std::fill(storage.begin(), storage.end(), value);
    }

    /// Fill the matrix as an identity matrix (all zeros except the diagonal
    /// which is one).
    void fill_identity() {
        fill(0);
        for (size_t i = 0; i < std::min(rows(), cols()); ++i)
            (*this)(i, i) = 1;
    }

    /// Fill the matrix with uniformly distributed random values.
    void fill_random(double min = 0, double max = 1,
                     std::default_random_engine::result_type seed =
                         std::default_random_engine::default_seed) {
        std::default_random_engine gen(seed);
        std::uniform_real_distribution<double> dist(min, max);
        std::generate(storage.begin(), storage.end(),
                      [&] { return dist(gen); });
    }

    /// @}

  public:
    /// @name   Create special matrices
    /// @{

    /// Create a matrix filled with ones.
    static Matrix ones(size_t rows, size_t cols) {
        return constant(rows, cols, 1);
    }

    /// Create a matrix filled with zeros.
    static Matrix zeros(size_t rows, size_t cols) {
        Matrix m(rows, cols);
        return m;
    }

    /// Create a matrix filled with a constant value.
    static Matrix constant(size_t rows, size_t cols, double value) {
        Matrix m(rows, cols);
        m.fill(value);
        return m;
    }

    /// Create an identity matrix.
    static Matrix identity(size_t rows) {
        Matrix m(rows, rows);
        m.fill_identity();
        return m;
    }

    /// Create a matrix with uniformly distributed random values.
    static Matrix random(size_t rows, size_t cols, double min = 0,
                         double max = 1,
                         std::default_random_engine::result_type seed =
                             std::default_random_engine::default_seed) {
        Matrix m(rows, cols);
        m.fill_random(min, max, seed);
        return m;
    }

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

    /// @}

  protected:
    size_t rows_ = 0, cols_ = 0;
    storage_t storage;

    friend class Vector;
    friend class RowVector;
};

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
    Vector &operator=(std::initializer_list<double> init) {
        static_cast<Matrix &>(*this) = {init};
        std::swap(rows_, cols_);
        return *this;
    }

    /// Convert an m×n matrix to a mn column vector.
    explicit Vector(const Matrix &matrix) : Matrix(matrix.storage) {}
    /// Convert an m×n matrix to a mn column vector.
    explicit Vector(Matrix &&matrix) : Matrix(std::move(matrix.storage)) {}

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
    static Vector ones(size_t size) { return Vector(Matrix::ones(size, 1)); }
    /// Create a vector filled with zeros.
    static Vector zeros(size_t size) { return Vector(Matrix::zeros(size, 1)); }
    /// Create a vector filled with a constant value.
    static Vector constant(size_t size, double value) {
        return Vector(Matrix::constant(size, 1, value));
    }
    /// Create a vector with uniformly distributed random values.
    static Vector random(size_t size, double min = 0, double max = 1,
                         std::default_random_engine::result_type seed =
                             std::default_random_engine::default_seed) {
        return Vector(Matrix::random(size, 1, min, max, seed));
    }

    /// @}

  public:
    /// @name Dot products
    /// @{

    /// Compute the dot product of two vectors. Reinterprets matrices as
    /// vectors.
    static double dot_unchecked(const Matrix &a, const Matrix &b);
    /// Compute the dot product of two vectors. Reinterprets matrices as
    /// vectors.
    static double dot_unchecked(Matrix &&a, const Matrix &b) {
        auto result = dot_unchecked(static_cast<const Matrix &>(a), b);
        a.clear_and_deallocate();
        return result;
    }
    /// Compute the dot product of two vectors. Reinterprets matrices as
    /// vectors.
    static double dot_unchecked(const Matrix &a, Matrix &&b) {
        return dot_unchecked(std::move(b), a);
    }
    /// Compute the dot product of two vectors. Reinterprets matrices as
    /// vectors.
    static double dot_unchecked(Matrix &&a, Matrix &&b) {
        auto result = dot_unchecked(static_cast<const Matrix &>(a),
                                    static_cast<const Matrix &>(b));
        a.clear_and_deallocate();
        b.clear_and_deallocate();
        return result;
    }

    /// Compute the dot product of two vectors.
    static double dot(const Vector &a, const Vector &b) {
        return dot_unchecked(a, b);
    }
    /// Compute the dot product of two vectors.
    static double dot(Vector &&a, const Vector &b) {
        return dot_unchecked(std::move(a), b);
    }
    /// Compute the dot product of two vectors.
    static double dot(const Vector &a, Vector &&b) {
        return dot_unchecked(a, std::move(b));
    }
    /// Compute the dot product of two vectors.
    static double dot(Vector &&a, Vector &&b) {
        return dot_unchecked(std::move(a), std::move(b));
    }

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
    /// with the result. Reinterprets matrices as vectors.
    static void cross_inplace_unchecked(Matrix &a, const Matrix &b);
    /// Compute the opposite of the cross product of two 3-vectors, overwriting
    /// the first vector with the result. Reinterprets matrices as vectors.
    static void cross_inplace_unchecked_neg(Matrix &a, const Matrix &b);

    /// Compute the cross product of two 3-vectors, overwriting the first vector
    /// with the result.
    static void cross_inplace(Vector &a, const Vector &b) {
        cross_inplace_unchecked(a, b);
    }
    /// Compute the cross product of two 3-vectors, overwriting the first vector
    /// with the result.
    static void cross_inplace(Vector &a, Vector &&b) {
        cross_inplace_unchecked(a, b);
        b.clear_and_deallocate();
    }
    /// Compute the opposite of the cross product of two 3-vectors, overwriting
    /// the first vector with the result.
    static void cross_inplace_neg(Vector &a, const Vector &b) {
        cross_inplace_unchecked_neg(a, b);
    }
    /// Compute the opposite of the cross product of two 3-vectors, overwriting
    /// the first vector with the result.
    static void cross_inplace_neg(Vector &a, Vector &&b) {
        cross_inplace_unchecked_neg(a, b);
        b.clear_and_deallocate();
    }

    /// Compute the cross product of two 3-vectors.
    static Vector cross(const Vector &a, const Vector &b) {
        Vector result = a;
        cross_inplace(result, b);
        return result;
    }
    /// Compute the cross product of two 3-vectors.
    static Vector &&cross(Vector &&a, const Vector &b) {
        cross_inplace(a, b);
        return std::move(a);
    }
    /// Compute the cross product of two 3-vectors.
    static Vector &&cross(const Vector &a, Vector &&b) {
        cross_inplace_neg(b, a);
        return std::move(b);
    }
    /// Compute the cross product of two 3-vectors.
    static Vector &&cross(Vector &&a, Vector &&b) {
        cross_inplace(a, std::move(b));
        return std::move(a);
    }

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
    double norm2() const & { return std::sqrt(dot(*this)); }
    /// Compute the 2-norm of the vector.
    double norm2() && { return std::sqrt(dot(std::move(*this))); }

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
    RowVector &operator=(std::initializer_list<double> init) {
        static_cast<Matrix &>(*this) = {init};
        return *this;
    }

    /// Convert an m×n matrix to a mn row vector.
    explicit RowVector(const Matrix &matrix) : Matrix(matrix.storage, false) {}
    /// Convert an m×n matrix to a mn row vector.
    explicit RowVector(Matrix &&matrix)
        : Matrix(std::move(matrix.storage), false) {}

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
    static RowVector ones(size_t size) {
        return RowVector(Matrix::ones(1, size));
    }
    /// Create a row vector filled with zeros.
    static RowVector zeros(size_t size) {
        return RowVector(Matrix::zeros(1, size));
    }
    /// Create a row vector filled with a constant value.
    static RowVector constant(size_t size, double value) {
        return RowVector(Matrix::constant(1, size, value));
    }
    /// Create a row vector with uniformly distributed random values.
    static RowVector random(size_t size, double min = 0, double max = 1,
                            std::default_random_engine::result_type seed =
                                std::default_random_engine::default_seed) {
        return RowVector(Matrix::random(1, size, min, max, seed));
    }

    /// @}

  public:
    /// @name   Dot products
    /// @{

    /// Compute the dot product of two vectors.
    static double dot(const RowVector &a, const RowVector &b) {
        return Vector::dot_unchecked(a, b);
    }
    /// Compute the dot product of two vectors.
    static double dot(RowVector &&a, const RowVector &b) {
        return Vector::dot_unchecked(std::move(a), b);
    }
    /// Compute the dot product of two vectors.
    static double dot(const RowVector &a, RowVector &&b) {
        return Vector::dot_unchecked(a, std::move(b));
    }
    /// Compute the dot product of two vectors.
    static double dot(RowVector &&a, RowVector &&b) {
        return Vector::dot_unchecked(std::move(a), std::move(b));
    }

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
    static void cross_inplace(RowVector &a, const RowVector &b) {
        Vector::cross_inplace_unchecked(a, b);
    }
    /// Compute the cross product of two 3-vectors, overwriting the first vector
    /// with the result.
    static void cross_inplace(RowVector &a, RowVector &&b) {
        Vector::cross_inplace_unchecked(a, b);
        b.clear_and_deallocate();
    }
    /// Compute the opposite of the cross product of two 3-vectors, overwriting
    /// the first vector with the result.
    static void cross_inplace_neg(RowVector &a, const RowVector &b) {
        Vector::cross_inplace_unchecked_neg(a, b);
    }
    /// Compute the opposite of the cross product of two 3-vectors, overwriting
    /// the first vector with the result.
    static void cross_inplace_neg(RowVector &a, RowVector &&b) {
        Vector::cross_inplace_unchecked_neg(a, b);
        b.clear_and_deallocate();
    }

    /// Compute the cross product of two 3-vectors.
    static RowVector cross(const RowVector &a, const RowVector &b) {
        RowVector result = a;
        cross_inplace(result, b);
        return result;
    }
    /// Compute the cross product of two 3-vectors.
    static RowVector &&cross(RowVector &&a, const RowVector &b) {
        cross_inplace(a, b);
        return std::move(a);
    }
    /// Compute the cross product of two 3-vectors.
    static RowVector &&cross(const RowVector &a, RowVector &&b) {
        cross_inplace_neg(b, a);
        return std::move(b);
    }
    /// Compute the cross product of two 3-vectors.
    static RowVector &&cross(RowVector &&a, RowVector &&b) {
        cross_inplace(a, std::move(b));
        return std::move(a);
    }

    /// Compute the cross product of this 3-vector with another 3-vector.
    RowVector cross(const RowVector &b) const & { return cross(*this, b); }
    /// Compute the cross product of this 3-vector with another 3-vector,
    RowVector &&cross(const RowVector &b) && {
        return cross(std::move(*this), b);
    }
    /// Compute the cross product of this 3-vector with another 3-vector,
    RowVector &&cross(RowVector &&b) const & {
        return cross(*this, std::move(b));
    }
    /// Compute the cross product of this 3-vector with another 3-vector,
    RowVector &&cross(RowVector &&b) && {
        return cross(std::move(*this), std::move(b));
    }

    /// @}

  public:
    /// @name   Vector norms
    /// @{

    /// Compute the 2-norm of the vector.
    double norm2() const & { return std::sqrt(dot(*this)); }
    /// Compute the 2-norm of the vector.
    double norm2() && { return std::sqrt(dot(std::move(*this))); }

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
    SquareMatrix(std::initializer_list<std::initializer_list<double>> init) {
        *this = init;
    }

    /// Assign the given values to the square matrix.
    SquareMatrix &
    operator=(std::initializer_list<std::initializer_list<double>> init) {
        static_cast<Matrix &>(*this) = init;
        assert(rows() == cols());
        return *this;
    }

    /// Convert a general matrix to a square matrix
    explicit SquareMatrix(Matrix &&matrix) : Matrix(std::move(matrix)) {
        assert(rows() == cols());
    }
    /// Convert a general matrix to a square matrix
    explicit SquareMatrix(const Matrix &matrix) : Matrix(matrix) {
        assert(rows() == cols());
    }

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
    static void transpose_inplace(Matrix &A) {
        assert(A.cols() == A.rows() && "Matrix should be square.");
        for (size_t n = 0; n < A.rows() - 1; ++n)
            for (size_t m = n + 1; m < A.rows(); ++m)
                std::swap(A(n, m), A(m, n));
    }
    /// Transpose the matrix in-place.
    void transpose_inplace() { transpose_inplace(*this); }

    /// @}

  public:
    /// @name   Creating special matrices
    /// @{

    /// Create a square matrix filled with ones.
    static SquareMatrix ones(size_t rows) {
        return SquareMatrix(Matrix::ones(rows, rows));
    }
    /// Create a square matrix filled with zeros.
    static SquareMatrix zeros(size_t rows) {
        return SquareMatrix(Matrix::zeros(rows, rows));
    }
    /// Create a square matrix filled with a constant value.
    static SquareMatrix constant(size_t rows, double value) {
        return SquareMatrix(Matrix::constant(rows, rows, value));
    }

    /// Create a square identity matrix.
    static SquareMatrix identity(size_t rows) {
        SquareMatrix m(rows);
        m.fill_identity();
        return m;
    }

    /// Create a matrix with uniformly distributed random values.
    static SquareMatrix random(size_t rows, double min = 0, double max = 1,
                               std::default_random_engine::result_type seed =
                                   std::default_random_engine::default_seed) {
        return SquareMatrix(Matrix::random(rows, rows, min, max, seed));
    }

    /// @}
};

/// @}

// :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: //

/// @addtogroup MatVecOp
/// @{

/// @defgroup MatMul    Matrix multiplication
/// @brief   Matrix-matrix, matrix-vector and vector-vector multiplication.
/// @{

/// Matrix multiplication.
inline Matrix operator*(const Matrix &A, const Matrix &B) {
    assert(A.cols() == B.rows() && "Inner dimensions don't match");
    Matrix C = Matrix::zeros(A.rows(), B.cols());
    for (size_t j = 0; j < B.cols(); ++j)
        for (size_t k = 0; k < A.cols(); ++k)
            for (size_t i = 0; i < A.rows(); ++i)
                C(i, j) += A(i, k) * B(k, j);
    return C;
}
/// Matrix multiplication.
inline Matrix operator*(Matrix &&A, const Matrix &B) {
    Matrix result = static_cast<const Matrix &>(A) * //
                    static_cast<const Matrix &>(B);
    A.clear_and_deallocate();
    return result;
}
/// Matrix multiplication.
inline Matrix operator*(const Matrix &A, Matrix &&B) {
    Matrix result = static_cast<const Matrix &>(A) * //
                    static_cast<const Matrix &>(B);
    B.clear_and_deallocate();
    return result;
}
/// Matrix multiplication.
inline Matrix operator*(Matrix &&A, Matrix &&B) {
    Matrix result = static_cast<const Matrix &>(A) * //
                    static_cast<const Matrix &>(B);
    A.clear_and_deallocate();
    B.clear_and_deallocate();
    return result;
}

/// Square matrix multiplication.
inline SquareMatrix operator*(const SquareMatrix &A, const SquareMatrix &B) {
    return SquareMatrix(static_cast<const Matrix &>(A) *
                        static_cast<const Matrix &>(B));
}
/// Square matrix multiplication.
inline SquareMatrix operator*(SquareMatrix &&A, const SquareMatrix &B) {
    return SquareMatrix(static_cast<Matrix &&>(A) *
                        static_cast<const Matrix &>(B));
}
/// Square matrix multiplication.
inline SquareMatrix operator*(const SquareMatrix &A, SquareMatrix &&B) {
    return SquareMatrix(static_cast<const Matrix &>(A) *
                        static_cast<Matrix &&>(B));
}
/// Square matrix multiplication.
inline SquareMatrix operator*(SquareMatrix &&A, SquareMatrix &&B) {
    return SquareMatrix(static_cast<Matrix &&>(A) * //
                        static_cast<Matrix &&>(B));
}

/// Matrix-vector multiplication.
inline Vector operator*(const Matrix &A, const Vector &b) {
    return Vector(A * static_cast<const Matrix &>(b));
}
/// Matrix-vector multiplication.
inline Vector operator*(Matrix &&A, const Vector &b) {
    return Vector(std::move(A) * static_cast<const Matrix &>(b));
}
/// Matrix-vector multiplication.
inline Vector operator*(const Matrix &A, Vector &&b) {
    return Vector(A * static_cast<Matrix &&>(b));
}
/// Matrix-vector multiplication.
inline Vector operator*(Matrix &&A, Vector &&b) {
    return Vector(std::move(A) * static_cast<Matrix &&>(b));
}

/// Matrix-vector multiplication.
inline RowVector operator*(const RowVector &a, const Matrix &B) {
    return RowVector(static_cast<const Matrix &>(a) * B);
}
/// Matrix-vector multiplication.
inline RowVector operator*(RowVector &&a, const Matrix &B) {
    return RowVector(static_cast<Matrix &&>(a) * B);
}
/// Matrix-vector multiplication.
inline RowVector operator*(const RowVector &a, Matrix &&B) {
    return RowVector(static_cast<const Matrix &>(a) * std::move(B));
}
/// Matrix-vector multiplication.
inline RowVector operator*(RowVector &&a, Matrix &&B) {
    return RowVector(static_cast<Matrix &&>(a) * std::move(B));
}

/// Vector-vector multiplication.
inline double operator*(const Vector &a, const RowVector &b) {
    return Vector::dot_unchecked(a, b);
}
/// Vector-vector multiplication.
inline double operator*(Vector &&a, const RowVector &b) {
    return Vector::dot_unchecked(std::move(a), b);
}
/// Vector-vector multiplication.
inline double operator*(const Vector &a, RowVector &&b) {
    return Vector::dot_unchecked(a, std::move(b));
}
/// Vector-vector multiplication.
inline double operator*(Vector &&a, RowVector &&b) {
    return Vector::dot_unchecked(std::move(a), std::move(b));
}

/// Vector-vector multiplication.
inline double operator*(const RowVector &a, const Vector &b) {
    return Vector::dot_unchecked(a, b);
}
/// Vector-vector multiplication.
inline double operator*(RowVector &&a, const Vector &b) {
    return Vector::dot_unchecked(std::move(a), b);
}
/// Vector-vector multiplication.
inline double operator*(const RowVector &a, Vector &&b) {
    return Vector::dot_unchecked(a, std::move(b));
}
/// Vector-vector multiplication.
inline double operator*(RowVector &&a, Vector &&b) {
    return Vector::dot_unchecked(std::move(a), std::move(b));
}

/// @}

// -------------------------------------------------------------------------- //

/// @defgroup   MatAdd  Addition
/// @brief  Matrix and vector addition
/// @{

inline void operator+=(Matrix &A, const Matrix &B) {
    assert(A.rows() == B.rows());
    assert(A.cols() == B.cols());
    std::transform(A.begin(), A.end(), B.begin(), A.begin(),
                   std::plus<double>());
}
inline Matrix operator+(const Matrix &A, const Matrix &B) {
    assert(A.rows() == B.rows());
    assert(A.cols() == B.cols());
    Matrix C(A.rows(), A.cols());
    std::transform(A.begin(), A.end(), B.begin(), C.begin(),
                   std::plus<double>());
    return C;
}
inline Matrix &&operator+(Matrix &&A, const Matrix &B) {
    A += B;
    return std::move(A);
}
inline Matrix &&operator+(const Matrix &A, Matrix &&B) {
    B += A;
    return std::move(B);
}
inline Matrix &&operator+(Matrix &&A, Matrix &&B) {
    A += B;
    B.clear_and_deallocate();
    return std::move(A);
}
inline Vector &&operator+(Vector &&a, const Vector &b) {
    static_cast<Matrix &>(a) += static_cast<const Matrix &>(b);
    return std::move(a);
}
inline Vector &&operator+(const Vector &a, Vector &&b) {
    static_cast<Matrix &>(b) += static_cast<const Matrix &>(a);
    return std::move(b);
}
inline Vector &&operator+(Vector &&a, Vector &&b) {
    static_cast<Matrix &>(a) += static_cast<const Matrix &>(b);
    b.clear_and_deallocate();
    return std::move(a);
}
inline RowVector &&operator+(RowVector &&a, const RowVector &b) {
    static_cast<Matrix &>(a) += static_cast<const Matrix &>(b);
    return std::move(a);
}
inline RowVector &&operator+(const RowVector &a, RowVector &&b) {
    static_cast<Matrix &>(b) += static_cast<const Matrix &>(a);
    return std::move(b);
}
inline RowVector &&operator+(RowVector &&a, RowVector &&b) {
    static_cast<Matrix &>(a) += static_cast<const Matrix &>(b);
    b.clear_and_deallocate();
    return std::move(a);
}
inline SquareMatrix &&operator+(SquareMatrix &&a, const SquareMatrix &b) {
    static_cast<Matrix &>(a) += static_cast<const Matrix &>(b);
    return std::move(a);
}
inline SquareMatrix &&operator+(const SquareMatrix &a, SquareMatrix &&b) {
    static_cast<Matrix &>(b) += static_cast<const Matrix &>(a);
    return std::move(b);
}
inline SquareMatrix &&operator+(SquareMatrix &&a, SquareMatrix &&b) {
    static_cast<Matrix &>(a) += static_cast<const Matrix &>(b);
    b.clear_and_deallocate();
    return std::move(a);
}
inline Vector operator+(const Vector &a, const Vector &b) {
    return Vector(static_cast<const Matrix &>(a) +
                  static_cast<const Matrix &>(b));
}
inline RowVector operator+(const RowVector &a, const RowVector &b) {
    return RowVector(static_cast<const Matrix &>(a) +
                     static_cast<const Matrix &>(b));
}
inline SquareMatrix operator+(const SquareMatrix &a, const SquareMatrix &b) {
    return SquareMatrix(static_cast<const Matrix &>(a) +
                        static_cast<const Matrix &>(b));
}

/// @}

// -------------------------------------------------------------------------- //

/// @defgroup   MatSub  Subtraction
/// @brief  Matrix and vector subtraction
/// @{

inline void operator-=(Matrix &A, const Matrix &B) {
    assert(A.rows() == B.rows());
    assert(A.cols() == B.cols());
    std::transform(A.begin(), A.end(), B.begin(), A.begin(),
                   std::minus<double>());
}
inline Matrix operator-(const Matrix &A, const Matrix &B) {
    assert(A.rows() == B.rows());
    assert(A.cols() == B.cols());
    Matrix C(A.rows(), A.cols());
    std::transform(A.begin(), A.end(), B.begin(), C.begin(),
                   std::minus<double>());
    return C;
}
inline Matrix &&operator-(Matrix &&A, const Matrix &B) {
    A -= B;
    return std::move(A);
}
inline Matrix &&operator-(const Matrix &A, Matrix &&B) {
    std::transform(A.begin(), A.end(), B.begin(), B.begin(),
                   std::minus<double>());
    return std::move(B);
}
inline Matrix &&operator-(Matrix &&A, Matrix &&B) {
    A -= B;
    B.clear_and_deallocate();
    return std::move(A);
}
inline Vector &&operator-(Vector &&a, const Vector &b) {
    static_cast<Matrix &>(a) -= static_cast<const Matrix &>(b);
    return std::move(a);
}
inline Vector &&operator-(const Vector &a, Vector &&b) {
    static_cast<const Matrix &>(a) - static_cast<Matrix &&>(b);
    return std::move(b);
}
inline Vector &&operator-(Vector &&a, Vector &&b) {
    static_cast<Matrix &>(a) -= static_cast<const Matrix &>(b);
    b.clear_and_deallocate();
    return std::move(a);
}
inline RowVector &&operator-(RowVector &&a, const RowVector &b) {
    static_cast<Matrix &>(a) -= static_cast<const Matrix &>(b);
    return std::move(a);
}
inline RowVector &&operator-(const RowVector &a, RowVector &&b) {
    static_cast<const Matrix &>(a) - static_cast<Matrix &&>(b);
    return std::move(b);
}
inline RowVector &&operator-(RowVector &&a, RowVector &&b) {
    static_cast<Matrix &>(a) -= static_cast<const Matrix &>(b);
    b.clear_and_deallocate();
    return std::move(a);
}
inline SquareMatrix &&operator-(SquareMatrix &&a, const SquareMatrix &b) {
    static_cast<Matrix &>(a) -= static_cast<const Matrix &>(b);
    return std::move(a);
}
inline SquareMatrix &&operator-(const SquareMatrix &a, SquareMatrix &&b) {
    static_cast<const Matrix &>(a) - static_cast<Matrix &&>(b);
    return std::move(b);
}
inline SquareMatrix &&operator-(SquareMatrix &&a, SquareMatrix &&b) {
    static_cast<Matrix &>(a) -= static_cast<const Matrix &>(b);
    b.clear_and_deallocate();
    return std::move(a);
}
inline Vector operator-(const Vector &a, const Vector &b) {
    return Vector(static_cast<const Matrix &>(a) -
                  static_cast<const Matrix &>(b));
}
inline RowVector operator-(const RowVector &a, const RowVector &b) {
    return RowVector(static_cast<const Matrix &>(a) -
                     static_cast<const Matrix &>(b));
}
inline SquareMatrix operator-(const SquareMatrix &a, const SquareMatrix &b) {
    return SquareMatrix(static_cast<const Matrix &>(a) -
                        static_cast<const Matrix &>(b));
}

/// @}

// -------------------------------------------------------------------------- //

/// @defgroup   MatNeg  Negation
/// @brief  Matrix and vector negation
/// @{

inline Matrix operator-(const Matrix &A) {
    Matrix result(A.rows(), A.cols());
    std::transform(A.begin(), A.end(), result.begin(), std::negate<double>());
    return result;
}
inline Matrix &&operator-(Matrix &&A) {
    std::transform(A.begin(), A.end(), A.begin(), std::negate<double>());
    return std::move(A);
}
inline Vector &&operator-(Vector &&a) {
    -static_cast<Matrix &&>(a);
    return std::move(a);
}
inline RowVector &&operator-(RowVector &&a) {
    -static_cast<Matrix &&>(a);
    return std::move(a);
}
inline SquareMatrix &&operator-(SquareMatrix &&a) {
    -static_cast<Matrix &&>(a);
    return std::move(a);
}
inline Vector operator-(const Vector &a) {
    return Vector(-static_cast<const Matrix &>(a));
}
inline RowVector operator-(const RowVector &a) {
    return RowVector(-static_cast<const Matrix &>(a));
}
inline SquareMatrix operator-(const SquareMatrix &a) {
    return SquareMatrix(-static_cast<const Matrix &>(a));
}

/// @}

// -------------------------------------------------------------------------- //

/// @defgroup   ScalMul Scalar multiplication
/// @brief  Multiplication by a scalar
/// @{

inline void operator*=(Matrix &A, double s) {
    std::transform(A.begin(), A.end(), A.begin(),
                   [s](double a) { return a * s; });
}
inline Matrix operator*(const Matrix &A, double s) {
    Matrix C(A.rows(), A.cols());
    std::transform(A.begin(), A.end(), C.begin(),
                   [s](double a) { return a * s; });
    return C;
}
inline Matrix &&operator*(Matrix &&A, double s) {
    A *= s;
    return std::move(A);
}
inline Vector operator*(const Vector &a, double s) {
    return Vector(static_cast<const Matrix &>(a) * s);
}
inline RowVector operator*(const RowVector &a, double s) {
    return RowVector(static_cast<const Matrix &>(a) * s);
}
inline SquareMatrix operator*(const SquareMatrix &a, double s) {
    return SquareMatrix(static_cast<const Matrix &>(a) * s);
}
inline Vector &&operator*(Vector &&a, double s) {
    static_cast<Matrix &>(a) *= s;
    return std::move(a);
}
inline RowVector &&operator*(RowVector &&a, double s) {
    static_cast<Matrix &>(a) *= s;
    return std::move(a);
}
inline SquareMatrix &&operator*(SquareMatrix &&a, double s) {
    static_cast<Matrix &>(a) *= s;
    return std::move(a);
}

inline Matrix operator*(double s, const Matrix &A) { return A * s; }
inline Matrix &&operator*(double s, Matrix &&A) { return std::move(A) * s; }
inline Vector operator*(double s, const Vector &a) { return a * s; }
inline RowVector operator*(double s, const RowVector &a) { return a * s; }
inline SquareMatrix operator*(double s, const SquareMatrix &a) { return a * s; }
inline Vector &&operator*(double s, Vector &&a) { return std::move(a) * s; }
inline RowVector &&operator*(double s, RowVector &&a) {
    return std::move(a) * s;
}
inline SquareMatrix &&operator*(double s, SquareMatrix &&a) {
    return std::move(a) * s;
}

/// @}

// -------------------------------------------------------------------------- //

/// @defgroup   ScalDiv Scalar division
/// @brief  Division by a scalar
/// @{

inline void operator/=(Matrix &A, double s) {
    std::transform(A.begin(), A.end(), A.begin(),
                   [s](double a) { return a / s; });
}
inline Matrix operator/(const Matrix &A, double s) {
    Matrix C(A.rows(), A.cols());
    std::transform(A.begin(), A.end(), C.begin(),
                   [s](double a) { return a / s; });
    return C;
}
inline Matrix &&operator/(Matrix &&A, double s) {
    A /= s;
    return std::move(A);
}
inline Vector operator/(const Vector &a, double s) {
    return Vector(static_cast<const Matrix &>(a) / s);
}
inline RowVector operator/(const RowVector &a, double s) {
    return RowVector(static_cast<const Matrix &>(a) / s);
}
inline SquareMatrix operator/(const SquareMatrix &a, double s) {
    return SquareMatrix(static_cast<const Matrix &>(a) / s);
}
inline Vector &&operator/(Vector &&a, double s) {
    static_cast<Matrix &>(a) /= s;
    return std::move(a);
}
inline RowVector &&operator/(RowVector &&a, double s) {
    static_cast<Matrix &>(a) /= s;
    return std::move(a);
}
inline SquareMatrix &&operator/(SquareMatrix &&a, double s) {
    static_cast<Matrix &>(a) /= s;
    return std::move(a);
}

/// @}

// -------------------------------------------------------------------------- //

/// @defgroup   MatTrans    Transposition
/// @brief  Matrix and vector transposition
/// @{

namespace detail {
/// Matrix transpose.
inline Matrix explicit_transpose(const Matrix &in) {
    Matrix out(in.cols(), in.rows());
    for (size_t n = 0; n < in.rows(); ++n)
        for (size_t m = 0; m < in.cols(); ++m)
            out(m, n) = in(n, m);
    return out;
}
} // namespace detail

/// Matrix transpose.
inline Matrix &&transpose(Matrix &&in) {
    if (in.rows() == in.cols()) // Square matrices
        SquareMatrix::transpose_inplace(in);
    else if (in.rows() == 1 || in.cols() == 1) // Vectors
        in.reshape(in.cols(), in.rows());
    else // General rectangular matrices
        in = detail::explicit_transpose(in);
    return std::move(in);
}
/// Matrix transpose.
inline Matrix transpose(const Matrix &in) {
    if (in.rows() == 1 || in.cols() == 1) { // Vectors
        Matrix out = in;
        out.reshape(in.cols(), in.rows());
        return out;
    } else { // General matrices (square and rectangular)
        return detail::explicit_transpose(in);
    }
}

/// Square matrix transpose.
inline SquareMatrix transpose(const SquareMatrix &in) {
    SquareMatrix out = in;
    out.transpose_inplace();
    return out;
}
/// Square matrix transpose.
inline SquareMatrix &&transpose(SquareMatrix &&in) {
    in.transpose_inplace();
    return std::move(in);
}

inline RowVector transpose(const Vector &in) { return RowVector(in); }
inline RowVector transpose(Vector &&in) { return RowVector(std::move(in)); }
inline Vector transpose(const RowVector &in) { return Vector(in); }
inline Vector transpose(RowVector &&in) { return Vector(std::move(in)); }

/// @}

/// @}

/// Print a matrix.
/// @related    Matrix
std::ostream &operator<<(std::ostream &os, const Matrix &M);

//                              Implementations                               //
// -------------------------------------------------------------------------- //

inline Matrix::Matrix(Matrix &&other) { *this = std::move(other); }

inline Matrix &Matrix::operator=(Matrix &&other) {
    // By explicitly defining move assignment, we can be sure that the object
    // that's being moved from has a consistent state.
    this->storage = std::move(other.storage);
    this->rows_   = other.rows_;
    this->cols_   = other.cols_;
    other.clear_and_deallocate();
    return *this;
}

inline Matrix::Matrix(
    std::initializer_list<std::initializer_list<double>> init) {
    *this = init;
}

inline Matrix &
Matrix::operator=(std::initializer_list<std::initializer_list<double>> init) {
    // First determine the size of the initializer list matrix:
    this->rows_ = init.size();
    assert(rows() > 0);
    this->cols_ = init.begin()->size();
    assert(cols() > 0);

    // Ensure that each row has the same number of columns:
    auto same_number_of_columns =
        [&](const std::initializer_list<double> &row) {
            return row.size() == cols();
        };
    assert(std::all_of(init.begin(), init.end(), same_number_of_columns));

    // Finally, allocate memory and copy the data to the internal storage
    storage.resize(rows() * cols());
    size_t r = 0;
    for (const auto &row : init) {
        size_t c = 0;
        for (double el : row) {
            (*this)(r, c) = el;
            ++c;
        }
        ++r;
    }

    return *this;
}

inline bool Matrix::operator==(const Matrix &other) const {
    assert(this->rows() == other.rows());
    assert(this->cols() == other.cols());
    auto res = std::mismatch(begin(), end(), other.begin());
    return res.first == end();
}

inline double Matrix::normFro() const & {
    return std::sqrt(Vector::dot_unchecked(*this, *this));
}

inline double Matrix::normFro() && {
    return std::sqrt(Vector::dot_unchecked(std::move(*this), *this));
}

inline double Vector::dot_unchecked(const Matrix &a, const Matrix &b) {
    assert(a.num_elems() == b.num_elems());
    return std::inner_product(a.begin(), a.end(), b.begin(), double(0));
}

inline void Vector::cross_inplace_unchecked(Matrix &a, const Matrix &b) {
    assert(a.num_elems() == 3);
    assert(b.num_elems() == 3);
    double a0 = a(1) * b(2) - a(2) * b(1);
    double a1 = a(2) * b(0) - a(0) * b(2);
    double a2 = a(0) * b(1) - a(1) * b(0);
    a(0)      = a0;
    a(1)      = a1;
    a(2)      = a2;
}

inline void Vector::cross_inplace_unchecked_neg(Matrix &a, const Matrix &b) {
    assert(a.num_elems() == 3);
    assert(b.num_elems() == 3);
    double a0 = a(2) * b(1) - a(1) * b(2);
    double a1 = a(0) * b(2) - a(2) * b(0);
    double a2 = a(1) * b(0) - a(0) * b(1);
    a(0)      = a0;
    a(1)      = a1;
    a(2)      = a2;
}
