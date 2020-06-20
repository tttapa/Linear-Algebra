#include <array>
#include <cmath> // std::sqrt, std::copysign
#include <limits> // std::numeric_limits

#include <iomanip>
#include <iostream>

using std::size_t;

template <size_t R, size_t C>
using Matrix = std::array<std::array<double, C>, R>;

template <size_t N>
using Vector = std::array<double, N>;

/// Result of a Householder QR factorization: stores the upper-triangular matrix
/// R and the matrix of scaled Householder reflection vectors W.
/// The vectors have norm √2.
template <size_t M, size_t N>
struct QR {
    static_assert(M >= N, "Matrix must be taller than it's wide");
    /// Contains the strict upper-triangular part of R and all of W.
    Matrix<M, N> RW;
    /// Contains the diagonal of R.
    Vector<N> R_diag;
};

template <size_t M, size_t N>
QR<M, N> householder_qr(const Matrix<M, N> &A) {
    // Copy the input matrix A to the output struct.
    // For the intermediate calculations, we'll be working with qr.RW.
    // At the end of the algorithm, qr.RW will contain the strict
    // upper-triangular part of the matrix R (without the diagonal),
    // and the complete scaled matrix of reflection vectors W, which is a
    // lower-triangular matrix. The diagonal of R is stored separately in
    // qr.R_diag.
    QR<M, N> qr = {A, {}};
    auto &RW    = qr.RW; // shorthand

    // Helper function to square a number
    auto sq = [](double x) { return x * x; };

    for (size_t k = 0; k < N; ++k) {
        // Introduce a column vector x = A[k:M,k], it's the lower part of the
        // k-th column of the matrix.
        // First compute the norm of x:

        double sq_norm_x = 0;
        for (size_t i = k; i < M; ++i)
            sq_norm_x += sq(RW[i][k]);
        double norm_x = std::sqrt(sq_norm_x);

        // x consists of two parts: its first element x₀ and the rest xₛ
        //     x = (x₀, xₛ)
        // You can express the norm of x in terms of the norms of the two parts:
        //     ‖x‖² = x₀² + ‖xₛ‖²
        double &x_0 = RW[k][k];

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
            for (size_t i = k; i < M; ++i)
                RW[i][k] /= norm_v_sq2;

            // Save the first component of xₕ:
            qr.R_diag[k] = x_p;
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
        // projection xₚ, and second, add the difference between xₚ and x to xₚ
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

        for (size_t c = k + 1; c < N; ++c) {
            // Compute wₖᵀ·aᵢ
            double dot_prod = 0;
            for (size_t r = k; r < M; ++r)
                dot_prod += RW[r][k] * RW[r][c];
            // Subtract wₖ·wₖᵀ·aᵢ
            for (size_t r = k; r < M; ++r)
                RW[r][c] -= RW[r][k] * dot_prod;
        }
    }

    return qr;
}

template <size_t M, size_t N, size_t K>
void apply_Q_transpose(const QR<M, N> &qr, Matrix<M, K> &b) {
    for (size_t c = 0; c < K; ++c) {
        for (size_t r = 0; r < N; ++r) {
            double dot_product = 0;
            for (size_t i = r; i < M; ++i)
                dot_product += qr.RW[i][r] * b[i][c];
            for (size_t i = r; i < M; ++i)
                b[i][c] -= qr.RW[i][r] * dot_product;
        }
    }
}

template <size_t M, size_t N, size_t K>
void apply_Q(const QR<M, N> &qr, Matrix<M, K> &b) {
    for (size_t c = 0; c < K; ++c) {
        for (size_t r = N; r-- > 0;) {
            double dot_product = 0;
            for (size_t i = r; i < M; ++i)
                dot_product += qr.RW[i][r] * b[i][c];
            for (size_t i = r; i < M; ++i)
                b[i][c] -= qr.RW[i][r] * dot_product;
        }
    }
}

template <size_t M, size_t N>
void extract_R(const QR<M, N> &qr, Matrix<M, N> &R) {
    for (size_t r = 0; r < N; ++r) {
        for (size_t c = 0; c < r; ++c)
            R[r][c] = 0;
        R[r][r] = qr.R_diag[r];
        for (size_t c = r + 1; c < N; ++c)
            R[r][c] = qr.RW[r][c];
    }
    for (size_t r = N; r < M; ++r) {
        for (size_t c = 0; c < N; ++c)
            R[r][c] = 0;
    }
}

template <size_t M, size_t N>
void extract_Q(const QR<M, N> &qr, Matrix<M, M> &Q) {
    Q = {};
    for (size_t i = 0; i < M; ++i)
        Q[i][i] = 1;
    apply_Q(qr, Q);
}

template <size_t M, size_t N>
void extract_Q_transpose(const QR<M, N> &qr, Matrix<M, M> &QT) {
    QT = {};
    for (size_t i = 0; i < M; ++i)
        QT[i][i] = 1;
    apply_Q_transpose(qr, QT);
}

template <size_t M>
void transpose(Matrix<M, M> &A) {
    for (size_t n = 0; n < M - 1; ++n)
        for (size_t m = n + 1; m < M; ++m)
            std::swap(A[n][m], A[m][n]);
}

template <size_t M, size_t N>
void print(std::ostream &os, const Matrix<M, N> &Q, int w) {
    for (size_t r = 0; r < M; ++r) {
        for (size_t c = 0; c < N; ++c)
            os << std::setw(w) << Q[r][c];
        os << std::endl;
    }
}

template <size_t M, size_t N>
std::ostream &operator<<(std::ostream &os, const QR<M, N> &qr) {
    os << std::scientific << std::setprecision(3);

    Matrix<M, M> Q;
    extract_Q(qr, Q);

    os << "Q = " << std::endl;
    print(os, Q, 17);

    os << "R = " << std::endl;
    for (size_t r = 0; r < N; ++r) {
        for (size_t c = 0; c < r; ++c)
            os << std::setw(17) << 0;
        os << std::setw(17) << qr.R_diag[r];
        for (size_t c = r + 1; c < N; ++c)
            os << std::setw(17) << qr.RW[r][c];
        os << std::endl;
    }
    for (size_t r = N; r < M; ++r) {
        for (size_t c = 0; c < N; ++c)
            os << std::setw(17) << 0;
    }
    return os;
}

template <size_t M, size_t N, size_t K>
Matrix<M, N> operator*(const Matrix<M, K> &A, const Matrix<K, N> &B) {
    Matrix<M, N> C = {};
    for (size_t j = 0; j < N; ++j)
        for (size_t k = 0; k < K; ++k)
            for (size_t i = 0; i < M; ++i)
                C[i][j] += A[i][k] * B[k][j];
    return C;
}

int main() {
    constexpr size_t M = 4, N = 3;
    Matrix<M, N> m = {{
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9},
        {10, 11, 12},
    }};

    auto qr = householder_qr(m);
    std::cout << qr << std::endl;

    Matrix<M, M> Q;
    Matrix<M, N> R;
    extract_Q(qr, Q);
    extract_R(qr, R);
    auto prod = Q * R;
    print(std::cout, prod, 17);
}