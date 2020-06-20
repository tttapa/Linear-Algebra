#include <algorithm>
#include <cassert>
#include <cmath> // std::sqrt, std::copysign
#include <limits>
#include <random>
#include <vector>

#include <iomanip>
#include <iostream>

using std::size_t;

class Matrix {
    friend class Vector;

  protected:
    explicit Matrix(std::vector<double> &&storage)
        : rows_(storage.size()), cols_(1), storage(std::move(storage)) {}

    explicit Matrix(const std::vector<double> &storage)
        : rows_(storage.size()), cols_(1), storage(storage) {}

  public:
    Matrix(size_t rows = 0, size_t cols = 0)
        : rows_(rows), cols_(cols), storage(rows * cols) {}

    size_t rows() const { return rows_; }
    size_t cols() const { return cols_; }

    double &operator()(size_t row, size_t col) {
#ifdef COL_MAJ_ORDER
        return storage[row + rows_ * col];
#else
        return storage[row * cols_ + col];
#endif
    }
    const double &operator()(size_t row, size_t col) const {
#ifdef COL_MAJ_ORDER
        return storage[row + rows_ * col];
#else
        return storage[row * cols_ + col];
#endif
    }

    void fill(double value) {
        std::fill(storage.begin(), storage.end(), value);
    }

    void fill_identity() {
        fill(0);
        for (size_t i = 0; i < std::min(rows(), cols()); ++i)
            (*this)(i, i) = 1;
    }

    void fill_random(double min = 0, double max = 1) {
        std::default_random_engine gen;
        std::uniform_real_distribution<double> dist(min, max);
        std::generate(storage.begin(), storage.end(),
                      [&] { return dist(gen); });
    }

    static Matrix random(size_t rows, size_t cols, double min = 0,
                         double max = 1) {
        Matrix m(rows, cols);
        m.fill_random(min, max);
        return m;
    }

    static Matrix identity(size_t rows) {
        Matrix m(rows, rows);
        m.fill_identity();
        return m;
    }

    static Matrix zeros(size_t rows, size_t cols) {
        Matrix m(rows, cols);
        return m;
    }

    static Matrix constant(size_t rows, size_t cols, double value) {
        Matrix m(rows, cols);
        m.fill(value);
        return m;
    }

    static Matrix ones(size_t rows, size_t cols) {
        return constant(rows, cols, 1);
    }

  private:
    size_t rows_, cols_;
    std::vector<double> storage;
};

class Vector {
  public:
    Vector(size_t size = 0) : storage(size) {}

    explicit Vector(Matrix &&matrix) : storage(std::move(matrix.storage)) {}

    double &operator()(size_t index) { return storage[index]; }
    const double &operator()(size_t index) const { return storage[index]; }

    void resize(size_t size) { storage.resize(size); }

    size_t size() const { return storage.size(); }

    Matrix to_matrix() && { return Matrix(std::move(storage)); }
    Matrix to_matrix() const & { return Matrix(storage); }

  private:
    std::vector<double> storage;
};

class HouseholderQR {
  public:
    HouseholderQR() = default;

  private:
    /// Result of a Householder QR factorization: stores the strict
    /// upper-triangular part of matrix R and the full matrix of scaled
    /// Householder reflection vectors W. The reflection vectors have norm √2.
    Matrix RW;
    /// Contains the diagonal elements of R.
    Vector R_diag;

    enum {
        NotFactored = 0,
        Factored    = 1,
    } state = NotFactored;

  private:
    void factor_impl();
    void back_subs(const Matrix &B, Matrix &X) const;

  public:
    void factor(Matrix &&matrix) {
        RW = std::move(matrix);
        R_diag.resize(RW.cols());
        factor_impl();
    }

    void factor(const Matrix &matrix) {
        RW = matrix;
        R_diag.resize(RW.cols());
        factor_impl();
    }

    void apply_QT(Matrix &B) const;
    Matrix apply_QT_copy(const Matrix &B) const {
        Matrix result = B;
        apply_QT(result);
        return result;
    }

    void apply_Q(Matrix &X) const;
    Matrix apply_Q_copy(const Matrix &X) const {
        Matrix result = X;
        apply_Q(result);
        return result;
    }

    void get_R(Matrix &R) const;
    Matrix get_R_copy() const {
        Matrix R(RW.rows(), RW.cols());
        get_R(R);
        return R;
    }

    Matrix steal_R();

    void get_Q(Matrix &R) const;
    Matrix get_Q_copy() const {
        Matrix Q(RW.rows(), RW.rows());
        get_Q(Q);
        return Q;
    }

    bool is_factored() const { return state == Factored; }

    const Matrix &get_RW() const { return RW; }
    const Vector &get_R_diag() const { return R_diag; }

    void solve(Matrix &B) const;
    Matrix solve_copy(const Matrix &B) const;
    void solve(Vector &b) const {
        Matrix B = std::move(b).to_matrix();
        solve(B);
        b = Vector(std::move(B));
    }
    Vector solve_copy(const Vector &b) const {
        Matrix B = b.to_matrix();
        return Vector(solve_copy(B));
    }
};

std::ostream &operator<<(std::ostream &os, const HouseholderQR &qr);

void HouseholderQR::factor_impl() {
    // For the intermediate calculations, we'll be working with RW.
    // It is initialized to the rectangular matrix to be factored.
    // At the end of this function, RW will contain the strict
    // upper-triangular part of the matrix R (without the diagonal),
    // and the complete scaled matrix of reflection vectors W, which is a
    // lower-triangular matrix. The diagonal of R is stored separately in
    // R_diag.

    // Helper function to square a number
    auto sq = [](double x) { return x * x; };

    for (size_t k = 0; k < RW.cols(); ++k) {
        // Introduce a column vector x = A[k:M,k], it's the lower part of the
        // k-th column of the matrix.
        // First compute the norm of x:

        double sq_norm_x = 0;
        for (size_t i = k; i < RW.rows(); ++i)
            sq_norm_x += sq(RW(i, k));
        double norm_x = std::sqrt(sq_norm_x);

        // x consists of two parts: its first element, x₀, and the rest, xₛ
        //     x = (x₀, xₛ)
        // You can express the norm of x in terms of the norms of the two parts:
        //     ‖x‖² = x₀² + ‖xₛ‖²
        double &x_0 = RW(k, k);

        // The goal of QR factorization is to transform the vector x, to a new
        // vector that is all zero, except for the first component.
        // Call this vector xₕ, the Householder reflection of x.
        // Since the transformation has to be unitary (Q is a unitary matrix),
        // this operation has to preserve the 2-norm. This means that the
        // nonzero component of xₕ has to have the same 2-norm (energy) as x:
        //     xₕ = (±‖x‖, 0, ..., 0) = ±‖x‖·̅e₁
        // where ̅e₁ is the first standard basis vector (1, 0, ..., 0).
        //
        // There are two (real) vectors that have the same energy as x in their
        // first component, because the sign doesn't affect the energy.
        // For numerical reasons, it's best to pick the sign opposite to the
        // sign of the first component of x, so that x and xₕ are far apart.
        //
        // The reflector vector vₖ is the difference between the original
        // vector x and its Householder reflection xₕ:
        //     vₖ = x - xₕ
        //        = x - (-sign(x₀)·‖x‖·̅e₁)
        //        = x + sign(x₀)·‖x‖·̅e₁
        //        = (x₀ + sign(x₀)·‖x‖, xₛ)
        //
        // Since vₖ will later be used to construct a projection matrix, it
        // should be normalized.
        // Computing the full norm is not necessary, since
        //     ‖vₖ‖² = (x₀ + sign(x₀)·‖x‖)² + ‖xₛ‖²
        //           = x₀² + 2·x₀·sign(x₀)·‖x‖ + ‖x‖² + ‖xₛ‖²
        //           = 2·x₀·sign(x₀)·‖x‖ + 2·‖x‖²
        //           = 2·|x₀|·‖x‖ + 2·‖x‖²
        //     ‖vₖ‖  = √2·√(|x₀|·‖x‖ + ‖x‖²)
        //
        // Normalize vₖ and call this uₖ:
        //     uₖ = vₖ / ‖vₖ‖
        //
        // For reasons that will become apparent in a moment, we'll keep the
        // factor of √2 in our normalization of vₖ, call this vector wₖ:
        //     wₖ = √2·uₖ
        //        = √2·vₖ / ‖vₖ‖
        //        = vₖ / √(|x₀|·‖x‖ + ‖x‖²)
        // Note how the sum only adds up numbers with the same sign. This
        // prevents catastrophic cancelations.
        //
        // It is clear that if ‖x‖ = 0, this normalization will fail. In that
        // case, we set uₖ = ̅e₁ or wₖ = √2·̅e₁.
        //
        // x will be overwritten by wₖ. The vector xₕ only has a single nonzero
        // component. It is saved in the R_diag vector.

        if (std::abs(norm_x) > std::numeric_limits<double>::min() * 2) {
            double x_p = -std::copysign(norm_x, x_0); // -sign(x₀)·‖x‖
            double v_0 = x_0 - x_p;
            double norm_v_sq2 = std::sqrt(std::abs(x_0) * norm_x + sq_norm_x);

            // Overwrite x with vₖ:
            x_0 = v_0;
            // the other components of x (xₛ) are already equal to the bottom
            // part of vₖ, so they don't have to be overwritten explicitly.

            // Then normalize x (= vₖ) to obtain wₖ:
            for (size_t i = k; i < RW.rows(); ++i)
                RW(i, k) /= norm_v_sq2;

            // Save the first component of xₕ:
            R_diag(k) = x_p;
        } else {
            // Overwrite x with wₖ = √2·̅e₁:
            x_0 = std::sqrt(2);
            // the other components of x (xₛ) are already equal to zero, since
            // ‖x‖ = 0.

            // R_diag is already initialized to zero.
        }

        // Now that the reflection vector vₖ (wₖ) is known, the rest of the
        // matrix A can be updated, that is A[k:m,k:n].
        //
        // The reflection was defined in terms of the reflection vector
        // vₖ = x - xₕ. To reflect x onto xₕ, you can first project x onto the
        // orthogonal complement of the space spanned by vₖ, call this
        // projection xₚ, and then add the difference between xₚ and x to xₚ
        // again, in order to reflect about the orthogonal complement of vₖ,
        // rather than projecting onto it:
        //     xₕ = xₚ + (xₚ - x)
        //        = 2·xₚ - x
        // The projection matrix onto the orthogonal complement of the space
        // spanned by vₖ is:
        //     P⟂ = I - uₖ·uₖᵀ
        // where I is the identity matrix, and uₖ the unit vector in the
        // direction of vₖ, as defined above.
        // This allows us to compute xₚ:
        //     xₚ = P⟂·x
        //        = I·x - uₖ·uₖᵀ·x
        //        = x - uₖ·uₖᵀ·x
        //     xₕ = 2·xₚ - x
        //        = 2·x - 2·uₖ·uₖᵀ·x - x
        //        = x - 2·uₖ·uₖᵀ·x
        // Because of our choice of wₖ = √2·uₖ earlier, this simplifies to
        //     xₕ = x - wₖ·wₖᵀ·x
        // The Householder reflector is thus the linear transformation given by
        //     H = I - wₖ·wₖᵀ
        //
        // The final step is to apply this reflector to the remaining part of
        // the matrix A, i.e. A[k:m,k:n].
        // The first column x = A[k:m,k] has already been updated, that's how
        // the reflector was computed, so only the submatrix A[k+1:m,k:n] has
        // to be updated.
        //     A'[k+1:m,k:n] = H·A[k+1:m,k:n]
        //                   = A[k+1:m,k:n] - wₖ·wₖᵀ·A[k+1:m,k:n]
        // This can be computed column-wise:
        //     aᵢ' = aᵢ - wₖ·wₖᵀ·aᵢ
        // where aᵢ is the i-th column of A.

        for (size_t c = k + 1; c < RW.cols(); ++c) {
            // Compute wₖᵀ·aᵢ
            double dot_product = 0;
            for (size_t r = k; r < RW.rows(); ++r)
                dot_product += RW(r, k) * RW(r, c);
            // Subtract wₖ·wₖᵀ·aᵢ
            for (size_t r = k; r < RW.rows(); ++r)
                RW(r, c) -= RW(r, k) * dot_product;
        }
    }
    state = Factored;
}

void HouseholderQR::apply_QT(Matrix &B) const {
    assert(state == Factored);
    assert(RW.rows() == B.rows());
    for (size_t c = 0; c < B.cols(); ++c) {
        for (size_t r = 0; r < RW.cols(); ++r) {
            double dot_product = 0;
            for (size_t i = r; i < RW.rows(); ++i)
                dot_product += RW(i, r) * B(i, c);
            for (size_t i = r; i < RW.rows(); ++i)
                B(i, c) -= RW(i, r) * dot_product;
        }
    }
}

void HouseholderQR::apply_Q(Matrix &X) const {
    assert(state == Factored);
    assert(RW.rows() == X.rows());
    for (size_t c = 0; c < X.cols(); ++c) {
        for (size_t r = RW.cols(); r-- > 0;) {
            double dot_product = 0;
            for (size_t i = r; i < RW.rows(); ++i)
                dot_product += RW(i, r) * X(i, c);
            for (size_t i = r; i < RW.rows(); ++i)
                X(i, c) -= RW(i, r) * dot_product;
        }
    }
}

void HouseholderQR::get_R(Matrix &R) const {
    assert(R.rows() == RW.rows());
    assert(R.cols() == RW.cols());
    for (size_t r = 0; r < R.cols(); ++r) {
        for (size_t c = 0; c < r; ++c)
            R(r, c) = 0;
        R(r, r) = R_diag(r);
        for (size_t c = r + 1; c < R.cols(); ++c)
            R(r, c) = RW(r, c);
    }
    for (size_t r = R.cols(); r < R.rows(); ++r) {
        for (size_t c = 0; c < R.cols(); ++c)
            R(r, c) = 0;
    }
}

Matrix HouseholderQR::steal_R() {
    state    = NotFactored;
    Matrix R = std::move(RW);
    for (size_t r = 0; r < R.cols(); ++r) {
        for (size_t c = 0; c < r; ++c)
            R(r, c) = 0;
        R(r, r) = R_diag(r);
    }
    for (size_t r = R.cols(); r < R.rows(); ++r) {
        for (size_t c = 0; c < R.cols(); ++c)
            R(r, c) = 0;
    }
    return R;
}

void HouseholderQR::get_Q(Matrix &Q) const {
    assert(Q.rows() == RW.rows());
    assert(Q.cols() == RW.rows());
    Q.fill_identity();
    apply_Q(Q);
}

void HouseholderQR::back_subs(const Matrix &B, Matrix &X) const {
    for (size_t i = 0; i < B.cols(); ++i) {
        for (size_t k = RW.cols(); k-- > 0;) {
            X(k, i) = B(k, i);
            for (size_t j = k + 1; j < RW.cols(); ++j) {
                X(k, i) -= RW(k, j) * X(j, i);
            }
            X(k, i) /= R_diag(k);
        }
    }
}

void HouseholderQR::solve(Matrix &B) const {
    apply_QT(B);

    // If the matrix is square, operate on B directly
    if (RW.cols() == RW.rows()) {
        back_subs(B, B);
    }
    // If the matrix is rectangular, use a separate result variable
    else {
        Matrix X(RW.cols(), B.cols());
        back_subs(B, X);
        B = std::move(X);
    }
}

Matrix HouseholderQR::solve_copy(const Matrix &B) const {
    Matrix B_cpy = B;
    apply_QT(B_cpy);
    Matrix X(RW.cols(), B.cols());
    back_subs(B_cpy, X);
    return X;
}

// :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: //

void transpose(Matrix &A) {
    assert(A.cols() == A.rows());
    for (size_t n = 0; n < A.rows() - 1; ++n)
        for (size_t m = n + 1; m < A.rows(); ++m)
            std::swap(A(n, m), A(m, n));
}

void print(std::ostream &os, const Matrix &Q, int w) {
    for (size_t r = 0; r < Q.rows(); ++r) {
        for (size_t c = 0; c < Q.cols(); ++c)
            os << std::setw(w) << Q(r, c);
        os << std::endl;
    }
}

void print(std::ostream &os, const Vector &v, int w) {
    for (size_t r = 0; r < v.size(); ++r) {
        os << std::setw(w) << v(r);
        os << std::endl;
    }
}

std::ostream &operator<<(std::ostream &os, const HouseholderQR &qr) {
    if (!qr.is_factored()) {
        os << "Not factored." << std::endl;
        return os;
    }

    Matrix Q = qr.get_Q_copy();

    os << "Q = " << std::endl;
    print(os, Q, 17);

    const auto &RW     = qr.get_RW();
    const auto &R_diag = qr.get_R_diag();

    os << "R = " << std::endl;
    for (size_t r = 0; r < RW.cols(); ++r) {
        for (size_t c = 0; c < r; ++c)
            os << std::setw(17) << 0;
        os << std::setw(17) << R_diag(r);
        for (size_t c = r + 1; c < RW.cols(); ++c)
            os << std::setw(17) << RW(r, c);
        os << std::endl;
    }
    for (size_t r = RW.cols(); r < RW.rows(); ++r) {
        for (size_t c = 0; c < RW.cols(); ++c)
            os << std::setw(17) << 0;
    }
    return os;
}

Matrix operator*(const Matrix &A, const Matrix &B) {
    assert(A.cols() == B.rows());
    Matrix C(A.rows(), B.cols());
    for (size_t j = 0; j < B.cols(); ++j)
        for (size_t k = 0; k < A.cols(); ++k)
            for (size_t i = 0; i < A.rows(); ++i)
                C(i, j) += A(i, k) * B(k, j);
    return C;
}

int main() {
    constexpr size_t M = 4, N = 3;
    Matrix m(M, N);

    std::array<std::array<double, N>, M> init = {{
        {1, 2, 1},
        {3, 4, 3},
        {1, 2, 3},
        {6, 5, 4},
    }};

    for (size_t r = 0; r < M; ++r)
        for (size_t c = 0; c < N; ++c)
            m(r, c) = init[r][c];

    HouseholderQR qr;
    qr.factor(std::move(m));

    std::cout << std::scientific << std::setprecision(8);
    std::cout << qr << std::endl;

    Matrix R    = qr.get_R_copy();
    Matrix prod = qr.apply_Q_copy(R);

    std::cout << "Q×R = " << std::endl;
    print(std::cout, prod, 17);

    Vector b(4);
    b(0)     = 4;
    b(1)     = 3;
    b(2)     = 2;
    b(3)     = 1;
    Vector x = qr.solve_copy(b);
    std::cout << "A \\ b = " << std::endl;
    print(std::cout, x, 17);
}