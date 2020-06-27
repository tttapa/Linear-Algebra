#include "LU.hpp"

#include <cassert>
#include <iomanip>
#include <iostream>

/**
 * @pre     `LU_` contains the matrix A to be factorized
 * @pre     `LU_.rows() == LU_.cols()`
 * 
 * @post    The complete upper-triangular part of `LU_` contains the full 
 *          upper-triangular matrix U and the strict lower-triangular part of 
 *          matrix L. The diagonal elements of L are implicitly 1.
 * @post    `get_L() * get_U() == A`
 *          (up to rounding errors)
 */
void LU::compute_factorization() {
    // For the intermediate calculations, we'll be working with LU_.
    // It is initialized to the square n×n matrix to be factored.

    assert(LU_.rows() == LU_.cols());

    // The goal of the LU factorization algorithm is to repeatedly apply
    // transformations Lₖ to the matrix A to eventually end up with an upper-
    // triangular matrix U:
    //
    //     Lₙ⋯L₂L₁A = U
    //
    // The first transformation L₁ will introduce zeros below the diagonal in
    // the first column of A, L₂ will introduce zeros below the diagonal in the
    // second column of L₁A (while preserving the zeros introduced by L₁), and
    // so on, until all elements below the diagonal are zero.

    // Loop over all columns of A:
    for (size_t k = 0; k < LU_.cols(); ++k) {
        // In the following comments, k = [1, n], because this is more intuitive
        // and it follows the usual mathematical convention.
        // In the code, however, array indices start at zero, so k = [0, n-1].

        // In order to introduce a zero in the i-th row and the k-th column of
        // the matrix, subtract a multiple of the k-th row from the i-th row,
        // using a scaling factor lᵢₖ such that
        //
        //     A(i,k) - lᵢₖ·A(k,k) = 0
        //     lᵢₖ = A(i,k) / A(k,k)
        // 
        // This is the typical Gaussian elimination algorithm.
        // The element A(k,k) is often called the pivot.

        // The first update step (k=1) subtracts multiples of the first row from
        // the rows below it.
        // It can be represented as a lower-triangular matrix L₁:
        //     ┌                ┐
        //     │  1             │
        //     │ -l₁₁ 1         │
        //     │ -l₂₁ 0   1     │
        //     │ -l₃₁ 0   0   1 │
        //     └                ┘
        // Compute the product L₁A to verify this!
        // You can see that the diagonal just preserves all rows, and the
        // factors in the first columns of L₁ subtract a multiple of the first
        // row from the other rows.
        //
        // The next step (k=2) follows exactly the same principle, but now
        // subtracts multiples of the second row, resulting in a matrix L₂:
        //     ┌               ┐
        //     │ 1             │
        //     │ 0   1         │
        //     │ 0  -l₁₂ 1     │
        //     │ 0  -l₂₂ 0   1 │
        //     └               ┘
        // This can be continued for all n columns of A.
        //
        // After applying matrices L₁ through Lₙ to A, it has been transformed
        // into an upper-triangular matrix U:
        //
        //     Lₙ⋯L₂L₁A = U
        //     A = L₁⁻¹L₂⁻¹⋯Lₙ⁻¹U
        //     A = LU
        //
        // Where
        //     L = L₁⁻¹L₂⁻¹⋯Lₙ⁻¹
        //
        // Luckily, inverting the matrices Lₖ is trivial, since they are sparse
        // lower-triangular matrices with a determinant of 1, so their inverses
        // will be equal to their adjugates.
        // As an example, and without loss of generality, computing the adjugate
        // of L₂ results in:
        //     ┌               ┐
        //     │ 1             │
        //     │ 0   1         │
        //     │ 0   l₁₂ 1     │
        //     │ 0   l₂₂ 0   1 │
        //     └               ┘
        // This result is not immediately obvious, you have to write out some
        // 3×3 determinants, but it's clear that many of them will be 0 or 1.
        // Recall that the adjugate of a matrix is the transpose of the cofactor
        // matrix. 
        //
        // Finally, computing the product of all Lₖ⁻¹ factors is trivial as
        // well, because of the structure of the factors. For example,
        //     L₁⁻¹L₂⁻¹ =
        //     ┌               ┐
        //     │ 1             │
        //     │ l₁₁ 1         │
        //     │ l₂₁ l₁₂ 1     │
        //     │ l₃₁ l₂₂ 0   1 │
        //     └               ┘
        // In conclusion, we can just combine all factors lᵢₖ into the matrix L,
        // without explicitly having to invert or multiply any matrices. Even
        // the minus signs cancel out!

        // Note that by applying this transformation Lₖ to A, you'll always end
        // up with zeros below the diagonal in the current column k, so there's
        // no point in saving these zeros explicitly. Instead, this space is
        // used to store the scaling factors lᵢₖ, i.e. the strict lower-
        // triangular elements of L. The diagonal elements of L don't have to be
        // stored either, because they're always 1.

        // Use the diagonal element as the pivot:
        double pivot = LU_(k, k);

        // Compute the k-th column of L, the coefficients lᵢₖ:
        for (size_t i = k + 1; i < LU_.rows(); ++i)
            LU_(i, k) /= pivot;

        // Now update the rest of the matrix, we already (implicitly) introduced
        // zeros below the diagonal in the k-th column, and we also stored the
        // scaling factors for each row that determine Lₖ, but we haven't 
        // actually subtracted the multiples of the pivot row from the rest of 
        // the matrix yet, or in other words, we haven't multiplied the bottom
        // right block of the matrix by Lₖ yet.
        //
        // Again, Lₖ has already been implicitly applied to the k-th column by
        // setting all values below the diagonal to zero. Also the k-th row is
        // unaffected by Lₖ because the k-th row of Lₖ is always equal to
        // ̅eₖ = (0 ... 0 1 0 ... 0).
        // This means only the rows and columns k+1 through n have to be 
        // updated.

        // Update the trailing submatrix A'(k+1:n,k+1:n) = LₖA(k+1:n,k+1:n):
        for (size_t c = k + 1; c < LU_.cols(); ++c)
            // Subtract lᵢₖ times the current pivot row A(k,:):
            for (size_t i = k + 1; i < LU_.rows(); ++i)
                // A'(i,c) = 1·A(i,c) - lᵢₖ·A(k,c)
                LU_(i, c) -= LU_(i, k) * LU_(k, c);

        // We won't handle this here explicitly, but notice how the algorithm
        // fails when the value of the pivot is zero (or very small), as this
        // will cause a division by zero. When using IEEE 754 floating point
        // numbers, this means that the factors lᵢₖ will overflow to ±∞,
        // and during later calculations, infinities might be subtracted from
        // eachother, resulting in many elements becoming NaN (Not a Number).
        //
        // Even ignoring the numerical issues with LU factorization, this is a
        // huge dealbreaker: if a zero pivot is encountered anywhere during the
        // factorization, it fails.
        // Zero pivots occur even when the matrix is non-singular.
    }
    state = Factored;
}

void LU::back_subs(const Matrix &B, Matrix &X) const {
    // Solve upper triangular system UX = B by solving each column of B as a
    // vector system Uxᵢ = bᵢ
    //
    //     ┌                 ┐┌     ┐   ┌     ┐
    //     │ u₁₁ u₁₂ u₁₃ u₁₄ ││ x₁ᵢ │   │ b₁ᵢ │
    //     │     u₂₂ u₂₃ u₂₄ ││ x₂ᵢ │ = │ b₂ᵢ │
    //     │         u₃₃ u₃₄ ││ x₃ᵢ │   │ b₃ᵢ │
    //     │             u₄₄ ││ x₄ᵢ │   │ b₄ᵢ │
    //     └                 ┘└     ┘   └     ┘
    //
    // b₄ᵢ = u₄₄·x₄ᵢ                     ⟺ x₄ᵢ = b₄ᵢ/u₄₄
    // b₃ᵢ = u₃₃·x₃ᵢ + u₃₄·x₄ᵢ           ⟺ x₃ᵢ = (b₃ᵢ - u₃₄·x₄ᵢ)/u₃₃
    // b₂ᵢ = u₂₂·x₂ᵢ + u₂₃·x₃ᵢ + u₂₄·x₄ᵢ ⟺ x₂ᵢ = (b₂ᵢ - u₂₃·x₃ᵢ + u₂₄·x₄ᵢ)/u₂₂
    // ...

    for (size_t i = 0; i < B.cols(); ++i) {
        for (size_t r = LU_.rows(); r-- > 0;) {
            X(r, i) = B(r, i);
            for (size_t c = r + 1; c < LU_.cols(); ++c)
                X(r, i) -= LU_(r, c) * X(c, i);
            X(r, i) /= LU_(r, r);
        }
    }
}

void LU::forward_subs(const Matrix &B, Matrix &X) const {
    // Solve lower triangular system LX = B by solving each column of B as a
    // vector system Lxᵢ = bᵢ.
    // The diagonal is always 1, due to the construction of the L matrix in the
    // LU algorithm.
    //
    //     ┌               ┐┌     ┐   ┌     ┐
    //     │ 1             ││ x₁ᵢ │   │ b₁ᵢ │
    //     │ l₂₁ 1         ││ x₂ᵢ │ = │ b₂ᵢ │
    //     │ l₃₁ l₃₂ 1     ││ x₃ᵢ │   │ b₃ᵢ │
    //     │ l₄₁ l₄₂ l₄₃ 1 ││ x₄ᵢ │   │ b₄ᵢ │
    //     └               ┘└     ┘   └     ┘
    //
    // b₁ᵢ =   1·x₁ᵢ                   ⟺ x₁ᵢ = b₁ᵢ
    // b₂ᵢ = l₂₁·x₁ᵢ +   1·x₂ᵢ         ⟺ x₂ᵢ = b₂ᵢ - l₂₁·x₁ᵢ
    // b₃ᵢ = l₃₁·x₁ᵢ + l₃₂·x₂ᵢ + 1·x₃ᵢ ⟺ x₃ᵢ = b₃ᵢ - l₃₂·x₂ᵢ - l₃₁·x₁ᵢ
    // ...

    for (size_t i = 0; i < B.cols(); ++i) {
        for (size_t r = 0; r < LU_.rows(); ++r) {
            X(r, i) = B(r, i);
            for (size_t c = 0; c < r; ++c)
                X(r, i) -= LU_(r, c) * X(c, i);
        }
    }
}

void LU::compute(SquareMatrix &&matrix) {
    LU_ = std::move(matrix);
    compute_factorization();
}

void LU::compute(const SquareMatrix &matrix) {
    LU_ = matrix;
    compute_factorization();
}

SquareMatrix &&LU::steal_L() {
    state = NotFactored;
    for (size_t c = 0; c < LU_.cols(); ++c) {
        // Elements above the diagonal are zero
        for (size_t r = 0; r < c; ++r)
            LU_(r, c) = 0;
        // Diagonal elements are one
        LU_(c, c) = 1;
        // Elements below the diagonal are stored in LU_ already
    }
    return std::move(LU_);
}

void LU::get_L_inplace(Matrix &L) const {
    assert(is_factored());
    assert(L.rows() == LU_.rows());
    assert(L.cols() == LU_.cols());
    for (size_t c = 0; c < L.cols(); ++c) {
        // Elements above the diagonal are zero
        for (size_t r = 0; r < c; ++r)
            L(r, c) = 0;
        // Diagonal elements are one
        L(c, c) = 1;
        // Elements below the diagonal are stored in LU_
        for (size_t r = c + 1; r < L.rows(); ++r)
            L(r, c) = LU_(r, c);
    }
}

SquareMatrix LU::get_L() const & {
    SquareMatrix L(LU_.rows());
    get_L_inplace(L);
    return L;
}

SquareMatrix &&LU::steal_U() {
    state = NotFactored;
    for (size_t c = 0; c < LU_.cols(); ++c) {
        // Elements above and on the diagonal are stored in LU_ already
        // Elements below the diagonal are zero
        for (size_t r = c + 1; r < LU_.rows(); ++r)
            LU_(r, c) = 0;
    }
    return std::move(LU_);
}

void LU::get_U_inplace(Matrix &U) const {
    assert(is_factored());
    assert(U.rows() == LU_.rows());
    assert(U.cols() == LU_.cols());
    for (size_t c = 0; c < U.cols(); ++c) {
        // Elements above and on the diagonal are stored in LU_
        for (size_t r = 0; r <= c; ++r)
            U(r, c) = LU_(r, c);
        // Elements below the diagonal are zero
        for (size_t r = c + 1; r < U.rows(); ++r)
            U(r, c) = 0;
    }
}

SquareMatrix LU::get_U() const & {
    SquareMatrix U(LU_.rows());
    get_U_inplace(U);
    return U;
}

void LU::solve_inplace(Matrix &B) const {
    // Solve the system AX = B, or LUX = B.
    //
    // Let UX = Z, and first solve LZ = B, which is a simple lower-triangular
    // system of equations.
    // Now that Z is known, solve UX = Z, which is a simple upper-triangular
    // system of equations.

    forward_subs(B, B); // overwrite B with Z
    back_subs(B, B);    // overwrite B (Z) with X
}

Matrix LU::solve(const Matrix &B) const {
    Matrix B_cpy = B;
    solve_inplace(B_cpy);
    return B_cpy;
}

Matrix &&LU::solve(Matrix &&B) const {
    solve_inplace(B);
    return std::move(B);
}

Vector LU::solve(const Vector &b) const {
    return Vector(solve(static_cast<const Matrix &>(b)));
}

Vector &&LU::solve(Vector &&b) const {
    solve_inplace(b);
    return std::move(b);
}

// LCOV_EXCL_START

std::ostream &operator<<(std::ostream &os, const LU &lu) {
    if (!lu.is_factored()) {
        os << "Not factored." << std::endl;
        return os;
    }

    // Output field width (characters)
    int w = os.precision() + 9;
    auto &LU_ = lu.get_LU();

    os << "L = " << std::endl;
    for (size_t r = 0; r < LU_.cols(); ++r) {
        for (size_t c = 0; c < r; ++c)
            os << std::setw(w) << 0;
        for (size_t c = r; c < LU_.cols(); ++c)
            os << std::setw(w) << LU_(r, c);
        os << std::endl;
    }

    os << "U = " << std::endl;
    for (size_t r = 0; r < LU_.cols(); ++r) {
        for (size_t c = 0; c < r; ++c)
            os << std::setw(w) << LU_(r, c);
        os << std::setw(w) << 1;
        for (size_t c = r; c < LU_.cols(); ++c)
            os << std::setw(w) << 0;
        os << std::endl;
    }
    return os;
}

// LCOV_EXCL_STOP
