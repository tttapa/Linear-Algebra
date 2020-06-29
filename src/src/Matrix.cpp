#include <linalg/Matrix.hpp>

#pragma region // Constructors -------------------------------------------------

Matrix::Matrix(storage_t &&storage, size_t rows, size_t cols)
    : rows_(rows), //
      cols_(cols), //
      storage(std::move(storage)) {}

Matrix::Matrix(const storage_t &storage, size_t rows, size_t cols)
    : rows_(rows), //
      cols_(cols), //
      storage(storage) {}

Matrix::Matrix(size_t rows, size_t cols)
    : rows_(rows), //
      cols_(cols), //
      storage(rows * cols) {}

Matrix::Matrix(Matrix &&other) { *this = std::move(other); }

Matrix::Matrix(std::initializer_list<std::initializer_list<double>> init) {
    *this = init;
}

#pragma endregion // -----------------------------------------------------------

#pragma region // Assignment ---------------------------------------------------

Matrix &Matrix::operator=(Matrix &&other) {
    // By explicitly defining move assignment, we can be sure that the object
    // that's being moved from has a consistent state.
    this->storage = std::move(other.storage);
    this->rows_ = other.rows_;
    this->cols_ = other.cols_;
    other.clear_and_deallocate();
    return *this;
}

Matrix &
Matrix::operator=(std::initializer_list<std::initializer_list<double>> init) {
    // First determine the size of the initializer list matrix:
    this->rows_ = init.size();
    assert(rows() > 0);
    this->cols_ = init.begin()->size();
    assert(cols() > 0);

    // Ensure that each row has the same number of columns:
    [[maybe_unused]] auto same_number_of_columns =
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

#pragma endregion // -----------------------------------------------------------

#pragma region // Matrix size --------------------------------------------------

void Matrix::reshape(size_t newrows, size_t newcols) {
    assert(newrows * newcols == rows() * cols());
    this->rows_ = newrows;
    this->cols_ = newcols;
}

Matrix Matrix::reshaped(size_t newrows, size_t newcols) const {
    Matrix result = *this;
    result.reshape(newrows, newcols);
    return result;
}

#pragma endregion // -----------------------------------------------------------

#pragma region // Element access -----------------------------------------------

double &Matrix::operator()(size_t row, size_t col) {
#if COL_MAJ_ORDER == 1
    return storage[row + rows_ * col];
#else
    return storage[row * cols_ + col];
#endif
}

const double &Matrix::operator()(size_t row, size_t col) const {
#if COL_MAJ_ORDER == 1
    return storage[row + rows_ * col];
#else
    return storage[row * cols_ + col];
#endif
}

#pragma endregion // -----------------------------------------------------------

#pragma region // Memory management --------------------------------------------

void Matrix::clear_and_deallocate() {
    this->rows_ = 0;
    this->cols_ = 0;
    storage_t().swap(this->storage); // replace storage with empty storage
    // temporary storage goes out of scope and deallocates original storage
}

#pragma endregion // -----------------------------------------------------------

#pragma region // Filling matrices ---------------------------------------------

void Matrix::fill(double value) {
    std::fill(storage.begin(), storage.end(), value);
}

void Matrix::fill_identity() {
    fill(0);
    for (size_t i = 0; i < std::min(rows(), cols()); ++i)
        (*this)(i, i) = 1;
}

void Matrix::fill_random(double min, double max,
                         std::default_random_engine::result_type seed) {
    std::default_random_engine gen(seed);
    std::uniform_real_distribution<double> dist(min, max);
    std::generate(storage.begin(), storage.end(), [&] { return dist(gen); });
}

#pragma endregion // -----------------------------------------------------------

#pragma region // Creating special matrices ------------------------------------

Matrix Matrix::ones(size_t rows, size_t cols) {
    return constant(rows, cols, 1);
}

Matrix Matrix::zeros(size_t rows, size_t cols) {
    Matrix m(rows, cols);
    return m;
}

Matrix Matrix::constant(size_t rows, size_t cols, double value) {
    Matrix m(rows, cols);
    m.fill(value);
    return m;
}

Matrix Matrix::identity(size_t rows, size_t cols) {
    Matrix m(rows, cols);
    m.fill_identity();
    return m;
}

Matrix Matrix::identity(size_t rows) { return identity(rows, rows); }

Matrix Matrix::random(size_t rows, size_t cols, double min, double max,
                      std::default_random_engine::result_type seed) {
    Matrix m(rows, cols);
    m.fill_random(min, max, seed);
    return m;
}

#pragma endregion // -----------------------------------------------------------

#pragma region // Swapping rows and columns ------------------------------------

void Matrix::swap_columns(size_t a, size_t b) {
    for (size_t r = 0; r < rows(); ++r)
        std::swap((*this)(r, a), (*this)(r, b));
}

void Matrix::swap_rows(size_t a, size_t b) {
    for (size_t c = 0; c < cols(); ++c)
        std::swap((*this)(a, c), (*this)(b, c));
}

#pragma endregion // -----------------------------------------------------------

#pragma region // Equality -----------------------------------------------------

bool Matrix::operator==(const Matrix &other) const {
    // When comparing two matrices with a different size, this is most likely
    // a bug, so don't return false, fail instead.
    assert(this->rows() == other.rows());
    assert(this->cols() == other.cols());
    // Find the first element of the matrices that differs:
    auto res = std::mismatch(begin(), end(), other.begin());
    // If such an element doesn't exist (i.e. when the two matrices are the
    // same), std::mismatch returns the end iterator:
    return res.first == end();
}

#pragma endregion // -----------------------------------------------------------

#pragma region // Norms --------------------------------------------------------

/**
 * ## Implementation
 * @snippet this Matrix::normFro
 */
//! <!-- [Matrix::normFro] -->
double Matrix::normFro() const & {
    // Reinterpret the matrix as one big vector, and compute the dot product
    // with itself. This is the 2-norm of the vector squared, so the Frobenius
    // norm of the matrix is the square root of this dot product.
    // ‖A‖f = ‖vec(A)‖₂ = √(vec(A)ᵀvec(A))
    return std::sqrt(Vector::dot_unchecked(*this, *this));
}
//! <!-- [Matrix::normFro] -->

double Matrix::normFro() && {
    // Same as above, but cleans up its storage once it's done.
    return std::sqrt(Vector::dot_unchecked(std::move(*this), *this));
}

#pragma endregion // -----------------------------------------------------------

#pragma region // Printing -----------------------------------------------------

#include <iomanip>
#include <iostream>

void Matrix::print(std::ostream &os, uint8_t precision, uint8_t width) const {
    int backup_precision = os.precision();
    precision = precision > 0 ? precision : backup_precision;
    width = width > 0 ? width : precision + 9;
    os.precision(precision);
    for (size_t r = 0; r < rows(); ++r) {
        for (size_t c = 0; c < cols(); ++c)
            os << std::setw(width) << (*this)(r, c);
        os << std::endl;
    }
    os.precision(backup_precision);
}

std::ostream &operator<<(std::ostream &os, const Matrix &M) {
    M.print(os);
    return os;
}

#pragma endregion // -----------------------------------------------------------

//                                   Vector                                   //
// :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: //

#pragma region // Constructors and assignment ----------------------------------

Vector::Vector(const Matrix &matrix)
    : Matrix(matrix.storage, matrix.num_elems(), 1) {}

Vector::Vector(Matrix &&matrix)
    : Matrix(std::move(matrix.storage), matrix.num_elems(), 1) {}

Vector &Vector::operator=(std::initializer_list<double> init) {
    // Assign this as a 1×n matrix to reuse the matrix code:
    static_cast<Matrix &>(*this) = {init};
    // Then swap the rows and columns to make it a column vector.
    std::swap(rows_, cols_);
    return *this;
}

#pragma endregion // -----------------------------------------------------------

#pragma region // Creating special vectors -------------------------------------

Vector Vector::ones(size_t size) { return Vector(Matrix::ones(size, 1)); }

Vector Vector::zeros(size_t size) { return Vector(Matrix::zeros(size, 1)); }

Vector Vector::constant(size_t size, double value) {
    return Vector(Matrix::constant(size, 1, value));
}

Vector Vector::random(size_t size, double min, double max,
                      std::default_random_engine::result_type seed) {
    return Vector(Matrix::random(size, 1, min, max, seed));
}

#pragma endregion // -----------------------------------------------------------

#pragma region // Dot products -------------------------------------------------

double Vector::dot_unchecked(const Matrix &a, const Matrix &b) {
    assert(a.num_elems() == b.num_elems());
    return std::inner_product(a.begin(), a.end(), b.begin(), double(0));
}

double Vector::dot_unchecked(Matrix &&a, const Matrix &b) {
    auto result = dot_unchecked(static_cast<const Matrix &>(a), b);
    a.clear_and_deallocate();
    return result;
}

double Vector::dot_unchecked(const Matrix &a, Matrix &&b) {
    return dot_unchecked(std::move(b), a);
}

double Vector::dot_unchecked(Matrix &&a, Matrix &&b) {
    auto result = dot_unchecked(static_cast<const Matrix &>(a),
                                static_cast<const Matrix &>(b));
    a.clear_and_deallocate();
    b.clear_and_deallocate();
    return result;
}

double Vector::dot(const Vector &a, const Vector &b) {
    return dot_unchecked(a, b);
}

double Vector::dot(Vector &&a, const Vector &b) {
    return dot_unchecked(std::move(a), b);
}

double Vector::dot(const Vector &a, Vector &&b) {
    return dot_unchecked(a, std::move(b));
}

double Vector::dot(Vector &&a, Vector &&b) {
    return dot_unchecked(std::move(a), std::move(b));
}

#pragma endregion // -----------------------------------------------------------

#pragma region // Cross products -----------------------------------------------

void Vector::cross_inplace_unchecked(Matrix &a, const Matrix &b) {
    assert(a.num_elems() == 3);
    assert(b.num_elems() == 3);
    double a0 = a(1) * b(2) - a(2) * b(1);
    double a1 = a(2) * b(0) - a(0) * b(2);
    double a2 = a(0) * b(1) - a(1) * b(0);
    a(0) = a0;
    a(1) = a1;
    a(2) = a2;
}

void Vector::cross_inplace_unchecked_neg(Matrix &a, const Matrix &b) {
    assert(a.num_elems() == 3);
    assert(b.num_elems() == 3);
    double a0 = a(2) * b(1) - a(1) * b(2);
    double a1 = a(0) * b(2) - a(2) * b(0);
    double a2 = a(1) * b(0) - a(0) * b(1);
    a(0) = a0;
    a(1) = a1;
    a(2) = a2;
}

void Vector::cross_inplace(Vector &a, const Vector &b) {
    cross_inplace_unchecked(a, b);
}

void Vector::cross_inplace(Vector &a, Vector &&b) {
    cross_inplace_unchecked(a, b);
    b.clear_and_deallocate();
}
void Vector::cross_inplace_neg(Vector &a, const Vector &b) {
    cross_inplace_unchecked_neg(a, b);
}

void Vector::cross_inplace_neg(Vector &a, Vector &&b) {
    cross_inplace_unchecked_neg(a, b);
    b.clear_and_deallocate();
}

Vector Vector::cross(const Vector &a, const Vector &b) {
    Vector result = a;
    cross_inplace(result, b);
    return result;
}

Vector &&Vector::cross(Vector &&a, const Vector &b) {
    cross_inplace(a, b);
    return std::move(a);
}

Vector &&Vector::cross(const Vector &a, Vector &&b) {
    cross_inplace_neg(b, a);
    return std::move(b);
}

Vector &&Vector::cross(Vector &&a, Vector &&b) {
    cross_inplace(a, std::move(b));
    return std::move(a);
}

#pragma endregion // -----------------------------------------------------------

#pragma region // Vector norms -------------------------------------------------

double Vector::norm2() const & {
    // Compute the dot product of the vector with itself. This is the sum of
    // the squares of the elements, which is the 2-norm of the vector squared.
    // The 2-norm norm is the square root of this dot product.
    // ‖v‖₂ = √(vᵀv)
    return std::sqrt(dot(*this));
}

double Vector::norm2() && {
    // Same as above but cleans up its resources when it's done.
    return std::sqrt(dot(std::move(*this)));
}

#pragma endregion // -----------------------------------------------------------

//                                 RowVector                                  //
// :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: //

#pragma region // Constructors and assignment ----------------------------------

RowVector::RowVector(const Matrix &matrix)
    : Matrix(matrix.storage, 1, matrix.num_elems()) {}

RowVector::RowVector(Matrix &&matrix)
    : Matrix(std::move(matrix.storage), 1, matrix.num_elems()) {}

RowVector &RowVector::operator=(std::initializer_list<double> init) {
    static_cast<Matrix &>(*this) = {init};
    return *this;
}

#pragma endregion // -----------------------------------------------------------

#pragma region // Creating special row vectors ---------------------------------

RowVector RowVector::ones(size_t size) {
    return RowVector(Matrix::ones(1, size));
}

RowVector RowVector::zeros(size_t size) {
    return RowVector(Matrix::zeros(1, size));
}

RowVector RowVector::constant(size_t size, double value) {
    return RowVector(Matrix::constant(1, size, value));
}

RowVector RowVector::random(size_t size, double min, double max,
                            std::default_random_engine::result_type seed) {
    return RowVector(Matrix::random(1, size, min, max, seed));
}

#pragma endregion // -----------------------------------------------------------

#pragma region // Dot products -------------------------------------------------

double RowVector::dot(const RowVector &a, const RowVector &b) {
    return Vector::dot_unchecked(a, b);
}

double RowVector::dot(RowVector &&a, const RowVector &b) {
    return Vector::dot_unchecked(std::move(a), b);
}

double RowVector::dot(const RowVector &a, RowVector &&b) {
    return Vector::dot_unchecked(a, std::move(b));
}

double RowVector::dot(RowVector &&a, RowVector &&b) {
    return Vector::dot_unchecked(std::move(a), std::move(b));
}

#pragma endregion // -----------------------------------------------------------

#pragma region // Cross products -----------------------------------------------

void RowVector::cross_inplace(RowVector &a, const RowVector &b) {
    Vector::cross_inplace_unchecked(a, b);
}
void RowVector::cross_inplace(RowVector &a, RowVector &&b) {
    Vector::cross_inplace_unchecked(a, b);
    b.clear_and_deallocate();
}
void RowVector::cross_inplace_neg(RowVector &a, const RowVector &b) {
    Vector::cross_inplace_unchecked_neg(a, b);
}
void RowVector::cross_inplace_neg(RowVector &a, RowVector &&b) {
    Vector::cross_inplace_unchecked_neg(a, b);
    b.clear_and_deallocate();
}

RowVector RowVector::cross(const RowVector &a, const RowVector &b) {
    RowVector result = a;
    cross_inplace(result, b);
    return result;
}
RowVector &&RowVector::cross(RowVector &&a, const RowVector &b) {
    cross_inplace(a, b);
    return std::move(a);
}
RowVector &&RowVector::cross(const RowVector &a, RowVector &&b) {
    cross_inplace_neg(b, a);
    return std::move(b);
}
RowVector &&RowVector::cross(RowVector &&a, RowVector &&b) {
    cross_inplace(a, std::move(b));
    return std::move(a);
}

RowVector RowVector::cross(const RowVector &b) const & {
    return cross(*this, b);
}
RowVector &&RowVector::cross(const RowVector &b) && {
    return cross(std::move(*this), b);
}
RowVector &&RowVector::cross(RowVector &&b) const & {
    return cross(*this, std::move(b));
}
RowVector &&RowVector::cross(RowVector &&b) && {
    return cross(std::move(*this), std::move(b));
}

#pragma endregion // -----------------------------------------------------------

#pragma region // Norms --------------------------------------------------------

double RowVector::norm2() const & { return std::sqrt(dot(*this)); }

double RowVector::norm2() && { return std::sqrt(dot(std::move(*this))); }

#pragma endregion // -----------------------------------------------------------

//                                SquareMatrix                                //
// :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: //

#pragma region // Constructors and assignment ----------------------------------

SquareMatrix::SquareMatrix(
    std::initializer_list<std::initializer_list<double>> init) {
    *this = init;
}

SquareMatrix::SquareMatrix(Matrix &&matrix) : Matrix(std::move(matrix)) {
    assert(rows() == cols());
}

SquareMatrix::SquareMatrix(const Matrix &matrix) : Matrix(matrix) {
    assert(rows() == cols());
}

SquareMatrix &SquareMatrix::operator=(
    std::initializer_list<std::initializer_list<double>> init) {
    static_cast<Matrix &>(*this) = init;
    assert(rows() == cols());
    return *this;
}

#pragma endregion // -----------------------------------------------------------

#pragma region // Transposition ------------------------------------------------

void SquareMatrix::transpose_inplace(Matrix &A) {
    assert(A.cols() == A.rows() && "Matrix should be square.");
    for (size_t n = 0; n < A.rows() - 1; ++n)
        for (size_t m = n + 1; m < A.rows(); ++m)
            std::swap(A(n, m), A(m, n));
}

#pragma endregion // -----------------------------------------------------------

#pragma region // Special matrices ---------------------------------------------

SquareMatrix SquareMatrix::ones(size_t rows) {
    return SquareMatrix(Matrix::ones(rows, rows));
}
SquareMatrix SquareMatrix::zeros(size_t rows) {
    return SquareMatrix(Matrix::zeros(rows, rows));
}
SquareMatrix SquareMatrix::constant(size_t rows, double value) {
    return SquareMatrix(Matrix::constant(rows, rows, value));
}
SquareMatrix SquareMatrix::identity(size_t rows) {
    SquareMatrix m(rows);
    m.fill_identity();
    return m;
}
SquareMatrix
SquareMatrix::random(size_t rows, double min, double max,
                     std::default_random_engine::result_type seed) {
    return SquareMatrix(Matrix::random(rows, rows, min, max, seed));
}

#pragma endregion // -----------------------------------------------------------

//                             Matrix operations                              //
// :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: //

#pragma region // Matrix multiplication ----------------------------------------

/**
 * ## Implementation
 * @snippet this operator*(Matrix, Matrix)
 */
//! <!-- [operator*(Matrix, Matrix)] -->
Matrix operator*(const Matrix &A, const Matrix &B) {
    assert(A.cols() == B.rows() && "Inner dimensions don't match");
    Matrix C = Matrix::zeros(A.rows(), B.cols());
    for (size_t j = 0; j < B.cols(); ++j)
        for (size_t k = 0; k < A.cols(); ++k)
            for (size_t i = 0; i < A.rows(); ++i)
                C(i, j) += A(i, k) * B(k, j);
    return C;
}
//! <!-- [operator*(Matrix, Matrix)] -->

Matrix operator*(Matrix &&A, const Matrix &B) {
    Matrix result = static_cast<const Matrix &>(A) * //
                    static_cast<const Matrix &>(B);
    A.clear_and_deallocate();
    return result;
}

Matrix operator*(const Matrix &A, Matrix &&B) {
    Matrix result = static_cast<const Matrix &>(A) * //
                    static_cast<const Matrix &>(B);
    B.clear_and_deallocate();
    return result;
}

Matrix operator*(Matrix &&A, Matrix &&B) {
    Matrix result = static_cast<const Matrix &>(A) * //
                    static_cast<const Matrix &>(B);
    A.clear_and_deallocate();
    B.clear_and_deallocate();
    return result;
}

SquareMatrix operator*(const SquareMatrix &A, const SquareMatrix &B) {
    return SquareMatrix(static_cast<const Matrix &>(A) *
                        static_cast<const Matrix &>(B));
}
SquareMatrix operator*(SquareMatrix &&A, const SquareMatrix &B) {
    return SquareMatrix(static_cast<Matrix &&>(A) *
                        static_cast<const Matrix &>(B));
}
SquareMatrix operator*(const SquareMatrix &A, SquareMatrix &&B) {
    return SquareMatrix(static_cast<const Matrix &>(A) *
                        static_cast<Matrix &&>(B));
}
SquareMatrix operator*(SquareMatrix &&A, SquareMatrix &&B) {
    return SquareMatrix(static_cast<Matrix &&>(A) * //
                        static_cast<Matrix &&>(B));
}

Vector operator*(const Matrix &A, const Vector &b) {
    return Vector(A * static_cast<const Matrix &>(b));
}
Vector operator*(Matrix &&A, const Vector &b) {
    return Vector(std::move(A) * static_cast<const Matrix &>(b));
}
Vector operator*(const Matrix &A, Vector &&b) {
    return Vector(A * static_cast<Matrix &&>(b));
}
Vector operator*(Matrix &&A, Vector &&b) {
    return Vector(std::move(A) * static_cast<Matrix &&>(b));
}

RowVector operator*(const RowVector &a, const Matrix &B) {
    return RowVector(static_cast<const Matrix &>(a) * B);
}
RowVector operator*(RowVector &&a, const Matrix &B) {
    return RowVector(static_cast<Matrix &&>(a) * B);
}
RowVector operator*(const RowVector &a, Matrix &&B) {
    return RowVector(static_cast<const Matrix &>(a) * std::move(B));
}
RowVector operator*(RowVector &&a, Matrix &&B) {
    return RowVector(static_cast<Matrix &&>(a) * std::move(B));
}

double operator*(const Vector &a, const RowVector &b) {
    return Vector::dot_unchecked(a, b);
}
double operator*(Vector &&a, const RowVector &b) {
    return Vector::dot_unchecked(std::move(a), b);
}
double operator*(const Vector &a, RowVector &&b) {
    return Vector::dot_unchecked(a, std::move(b));
}
double operator*(Vector &&a, RowVector &&b) {
    return Vector::dot_unchecked(std::move(a), std::move(b));
}

double operator*(const RowVector &a, const Vector &b) {
    return Vector::dot_unchecked(a, b);
}
double operator*(RowVector &&a, const Vector &b) {
    return Vector::dot_unchecked(std::move(a), b);
}
double operator*(const RowVector &a, Vector &&b) {
    return Vector::dot_unchecked(a, std::move(b));
}
double operator*(RowVector &&a, Vector &&b) {
    return Vector::dot_unchecked(std::move(a), std::move(b));
}

#pragma endregion // -----------------------------------------------------------

#pragma region // Addition -----------------------------------------------------

/**
 * ## Implementation
 * @snippet this operator+(Matrix, Matrix)
 */
//! <!-- [operator+(Matrix, Matrix)] -->
Matrix operator+(const Matrix &A, const Matrix &B) {
    assert(A.rows() == B.rows());
    assert(A.cols() == B.cols());
    Matrix C(A.rows(), A.cols());
    std::transform(A.begin(), A.end(), B.begin(), C.begin(),
                   std::plus<double>());
    return C;
}
//! <!-- [operator+(Matrix, Matrix)] -->

void operator+=(Matrix &A, const Matrix &B) {
    assert(A.rows() == B.rows());
    assert(A.cols() == B.cols());
    std::transform(A.begin(), A.end(), B.begin(), A.begin(),
                   std::plus<double>());
}
Matrix &&operator+(Matrix &&A, const Matrix &B) {
    A += B;
    return std::move(A);
}
Matrix &&operator+(const Matrix &A, Matrix &&B) {
    B += A;
    return std::move(B);
}
Matrix &&operator+(Matrix &&A, Matrix &&B) {
    A += B;
    B.clear_and_deallocate();
    return std::move(A);
}
Vector &&operator+(Vector &&a, const Vector &b) {
    static_cast<Matrix &>(a) += static_cast<const Matrix &>(b);
    return std::move(a);
}
Vector &&operator+(const Vector &a, Vector &&b) {
    static_cast<Matrix &>(b) += static_cast<const Matrix &>(a);
    return std::move(b);
}
Vector &&operator+(Vector &&a, Vector &&b) {
    static_cast<Matrix &>(a) += static_cast<const Matrix &>(b);
    b.clear_and_deallocate();
    return std::move(a);
}
RowVector &&operator+(RowVector &&a, const RowVector &b) {
    static_cast<Matrix &>(a) += static_cast<const Matrix &>(b);
    return std::move(a);
}
RowVector &&operator+(const RowVector &a, RowVector &&b) {
    static_cast<Matrix &>(b) += static_cast<const Matrix &>(a);
    return std::move(b);
}
RowVector &&operator+(RowVector &&a, RowVector &&b) {
    static_cast<Matrix &>(a) += static_cast<const Matrix &>(b);
    b.clear_and_deallocate();
    return std::move(a);
}
SquareMatrix &&operator+(SquareMatrix &&a, const SquareMatrix &b) {
    static_cast<Matrix &>(a) += static_cast<const Matrix &>(b);
    return std::move(a);
}
SquareMatrix &&operator+(const SquareMatrix &a, SquareMatrix &&b) {
    static_cast<Matrix &>(b) += static_cast<const Matrix &>(a);
    return std::move(b);
}
SquareMatrix &&operator+(SquareMatrix &&a, SquareMatrix &&b) {
    static_cast<Matrix &>(a) += static_cast<const Matrix &>(b);
    b.clear_and_deallocate();
    return std::move(a);
}
Vector operator+(const Vector &a, const Vector &b) {
    return Vector(static_cast<const Matrix &>(a) +
                  static_cast<const Matrix &>(b));
}
RowVector operator+(const RowVector &a, const RowVector &b) {
    return RowVector(static_cast<const Matrix &>(a) +
                     static_cast<const Matrix &>(b));
}
SquareMatrix operator+(const SquareMatrix &a, const SquareMatrix &b) {
    return SquareMatrix(static_cast<const Matrix &>(a) +
                        static_cast<const Matrix &>(b));
}

#pragma endregion // -----------------------------------------------------------

#pragma region // Subtraction --------------------------------------------------

/**
 * ## Implementation
 * @snippet this operator-(Matrix, Matrix)
 */
//! <!-- [operator-(Matrix, Matrix)] -->
Matrix operator-(const Matrix &A, const Matrix &B) {
    assert(A.rows() == B.rows());
    assert(A.cols() == B.cols());
    Matrix C(A.rows(), A.cols());
    std::transform(A.begin(), A.end(), B.begin(), C.begin(),
                   std::minus<double>());
    return C;
}
//! <!-- [operator-(Matrix, Matrix)] -->

void operator-=(Matrix &A, const Matrix &B) {
    assert(A.rows() == B.rows());
    assert(A.cols() == B.cols());
    std::transform(A.begin(), A.end(), B.begin(), A.begin(),
                   std::minus<double>());
}
Matrix &&operator-(Matrix &&A, const Matrix &B) {
    A -= B;
    return std::move(A);
}
Matrix &&operator-(const Matrix &A, Matrix &&B) {
    std::transform(A.begin(), A.end(), B.begin(), B.begin(),
                   std::minus<double>());
    return std::move(B);
}
Matrix &&operator-(Matrix &&A, Matrix &&B) {
    A -= B;
    B.clear_and_deallocate();
    return std::move(A);
}
Vector &&operator-(Vector &&a, const Vector &b) {
    static_cast<Matrix &>(a) -= static_cast<const Matrix &>(b);
    return std::move(a);
}
Vector &&operator-(const Vector &a, Vector &&b) {
    static_cast<const Matrix &>(a) - static_cast<Matrix &&>(b);
    return std::move(b);
}
Vector &&operator-(Vector &&a, Vector &&b) {
    static_cast<Matrix &>(a) -= static_cast<const Matrix &>(b);
    b.clear_and_deallocate();
    return std::move(a);
}
RowVector &&operator-(RowVector &&a, const RowVector &b) {
    static_cast<Matrix &>(a) -= static_cast<const Matrix &>(b);
    return std::move(a);
}
RowVector &&operator-(const RowVector &a, RowVector &&b) {
    static_cast<const Matrix &>(a) - static_cast<Matrix &&>(b);
    return std::move(b);
}
RowVector &&operator-(RowVector &&a, RowVector &&b) {
    static_cast<Matrix &>(a) -= static_cast<const Matrix &>(b);
    b.clear_and_deallocate();
    return std::move(a);
}
SquareMatrix &&operator-(SquareMatrix &&a, const SquareMatrix &b) {
    static_cast<Matrix &>(a) -= static_cast<const Matrix &>(b);
    return std::move(a);
}
SquareMatrix &&operator-(const SquareMatrix &a, SquareMatrix &&b) {
    static_cast<const Matrix &>(a) - static_cast<Matrix &&>(b);
    return std::move(b);
}
SquareMatrix &&operator-(SquareMatrix &&a, SquareMatrix &&b) {
    static_cast<Matrix &>(a) -= static_cast<const Matrix &>(b);
    b.clear_and_deallocate();
    return std::move(a);
}
Vector operator-(const Vector &a, const Vector &b) {
    return Vector(static_cast<const Matrix &>(a) -
                  static_cast<const Matrix &>(b));
}
RowVector operator-(const RowVector &a, const RowVector &b) {
    return RowVector(static_cast<const Matrix &>(a) -
                     static_cast<const Matrix &>(b));
}
SquareMatrix operator-(const SquareMatrix &a, const SquareMatrix &b) {
    return SquareMatrix(static_cast<const Matrix &>(a) -
                        static_cast<const Matrix &>(b));
}

#pragma endregion // -----------------------------------------------------------

#pragma region // Negation -----------------------------------------------------

/**
 * @brief   Matrix negation.
 * 
 * ## Implementation
 * @snippet this operator-(Matrix)
 */
//! <!-- [operator-(Matrix)] -->
Matrix operator-(const Matrix &A) {
    Matrix result(A.rows(), A.cols());
    std::transform(A.begin(), A.end(), result.begin(), std::negate<double>());
    return result;
}
//! <!-- [operator-(Matrix)] -->

Matrix &&operator-(Matrix &&A) {
    std::transform(A.begin(), A.end(), A.begin(), std::negate<double>());
    return std::move(A);
}
Vector &&operator-(Vector &&a) {
    -static_cast<Matrix &&>(a);
    return std::move(a);
}
RowVector &&operator-(RowVector &&a) {
    -static_cast<Matrix &&>(a);
    return std::move(a);
}
SquareMatrix &&operator-(SquareMatrix &&a) {
    -static_cast<Matrix &&>(a);
    return std::move(a);
}
Vector operator-(const Vector &a) {
    return Vector(-static_cast<const Matrix &>(a));
}
RowVector operator-(const RowVector &a) {
    return RowVector(-static_cast<const Matrix &>(a));
}
SquareMatrix operator-(const SquareMatrix &a) {
    return SquareMatrix(-static_cast<const Matrix &>(a));
}

#pragma endregion // -----------------------------------------------------------

#pragma region // Scalar multiplication ----------------------------------------

/**
 * ## Implementation
 * @snippet this operator*(Matrix, double)
 */
//! <!-- [operator*(Matrix, double)] -->
Matrix operator*(const Matrix &A, double s) {
    Matrix C(A.rows(), A.cols());
    std::transform(A.begin(), A.end(), C.begin(),
                   [s](double a) { return a * s; });
    return C;
}
//! <!-- [operator*(Matrix, double)] -->

void operator*=(Matrix &A, double s) {
    std::transform(A.begin(), A.end(), A.begin(),
                   [s](double a) { return a * s; });
}
Matrix &&operator*(Matrix &&A, double s) {
    A *= s;
    return std::move(A);
}
Vector operator*(const Vector &a, double s) {
    return Vector(static_cast<const Matrix &>(a) * s);
}
RowVector operator*(const RowVector &a, double s) {
    return RowVector(static_cast<const Matrix &>(a) * s);
}
SquareMatrix operator*(const SquareMatrix &a, double s) {
    return SquareMatrix(static_cast<const Matrix &>(a) * s);
}
Vector &&operator*(Vector &&a, double s) {
    static_cast<Matrix &>(a) *= s;
    return std::move(a);
}
RowVector &&operator*(RowVector &&a, double s) {
    static_cast<Matrix &>(a) *= s;
    return std::move(a);
}
SquareMatrix &&operator*(SquareMatrix &&a, double s) {
    static_cast<Matrix &>(a) *= s;
    return std::move(a);
}

Matrix operator*(double s, const Matrix &A) { return A * s; }
Matrix &&operator*(double s, Matrix &&A) { return std::move(A) * s; }
Vector operator*(double s, const Vector &a) { return a * s; }
RowVector operator*(double s, const RowVector &a) { return a * s; }
SquareMatrix operator*(double s, const SquareMatrix &a) { return a * s; }
Vector &&operator*(double s, Vector &&a) { return std::move(a) * s; }
RowVector &&operator*(double s, RowVector &&a) { return std::move(a) * s; }
SquareMatrix &&operator*(double s, SquareMatrix &&a) {
    return std::move(a) * s;
}

#pragma endregion // -----------------------------------------------------------

#pragma region // Scalar division ----------------------------------------------

/**
 * ## Implementation
 * @snippet this operator/(Matrix, double)
 */
//! <!-- [operator/(Matrix, double)] -->
Matrix operator/(const Matrix &A, double s) {
    Matrix C(A.rows(), A.cols());
    std::transform(A.begin(), A.end(), C.begin(),
                   [s](double a) { return a / s; });
    return C;
}
//! <!-- [operator/(Matrix, double)] -->

void operator/=(Matrix &A, double s) {
    std::transform(A.begin(), A.end(), A.begin(),
                   [s](double a) { return a / s; });
}
Matrix &&operator/(Matrix &&A, double s) {
    A /= s;
    return std::move(A);
}
Vector operator/(const Vector &a, double s) {
    return Vector(static_cast<const Matrix &>(a) / s);
}
RowVector operator/(const RowVector &a, double s) {
    return RowVector(static_cast<const Matrix &>(a) / s);
}
SquareMatrix operator/(const SquareMatrix &a, double s) {
    return SquareMatrix(static_cast<const Matrix &>(a) / s);
}
Vector &&operator/(Vector &&a, double s) {
    static_cast<Matrix &>(a) /= s;
    return std::move(a);
}
RowVector &&operator/(RowVector &&a, double s) {
    static_cast<Matrix &>(a) /= s;
    return std::move(a);
}
SquareMatrix &&operator/(SquareMatrix &&a, double s) {
    static_cast<Matrix &>(a) /= s;
    return std::move(a);
}

#pragma endregion // -----------------------------------------------------------

#pragma region // Transposition ------------------------------------------------

/**
 * ## Implementation
 * @snippet this explicit_transpose
 */
//! <!-- [explicit_transpose] -->
Matrix explicit_transpose(const Matrix &in) {
    Matrix out(in.cols(), in.rows());
    for (size_t n = 0; n < in.rows(); ++n)
        for (size_t m = 0; m < in.cols(); ++m)
            out(m, n) = in(n, m);
    return out;
}
//! <!-- [explicit_transpose] -->

/**
 * ## Implementation
 * @snippet this transpose(const Matrix &)
 */
//! <!-- [transpose(const Matrix &)] -->
Matrix transpose(const Matrix &in) {
    if (in.rows() == 1 || in.cols() == 1) { // Vectors
        Matrix out = in;
        out.reshape(in.cols(), in.rows());
        return out;
    } else { // General matrices (square and rectangular)
        return explicit_transpose(in);
    }
}
//! <!-- [transpose(const Matrix &)] -->

/**
 * ## Implementation
 * @snippet this transpose(Matrix &&)
 */
//! <!-- [transpose(Matrix &&)] -->
Matrix &&transpose(Matrix &&in) {
    if (in.rows() == in.cols())                // Square matrices
        SquareMatrix::transpose_inplace(in);   //     → reuse storage
    else if (in.rows() == 1 || in.cols() == 1) // Vectors
        in.reshape(in.cols(), in.rows());      //     → reshape row ↔ column
    else                                       // General rectangular matrices
        in = explicit_transpose(in);           //     → full transpose
    return std::move(in);
}
//! <!-- [transpose(Matrix &&)] -->

SquareMatrix transpose(const SquareMatrix &in) {
    SquareMatrix out = in;
    out.transpose_inplace();
    return out;
}
SquareMatrix &&transpose(SquareMatrix &&in) {
    in.transpose_inplace();
    return std::move(in);
}

RowVector transpose(const Vector &in) { return RowVector(in); }
RowVector transpose(Vector &&in) { return RowVector(std::move(in)); }
Vector transpose(const RowVector &in) { return Vector(in); }
Vector transpose(RowVector &&in) { return Vector(std::move(in)); }

#pragma endregion // -----------------------------------------------------------
