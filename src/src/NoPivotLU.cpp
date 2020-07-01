#ifndef ARDUINO
#include <linalg/NoPivotLU.hpp>
#else
#include <include/linalg/NoPivotLU.hpp>
#endif

/**
 * @pre     `LU` contains the matrix A to be factorized
 * @pre     `LU.rows() == LU.cols()`
 * 
 * @post    The complete upper-triangular part of `LU` contains the full 
 *          upper-triangular matrix U and the strict lower-triangular part of 
 *          matrix L. The diagonal elements of L are implicitly 1.
 * @post    `get_L() * get_U() == A`
 *          (up to rounding errors)
 * 
 * ## Implementation
 * @snippet this NoPivotLU::compute_factorization
 */
//! <!-- [NoPivotLU::compute_factorization] -->
void NoPivotLU::compute_factorization() {
    // For the intermediate calculations, we'll be working with LU.
    // It is initialized to the square n×n matrix to be factored.

    assert(LU.rows() == LU.cols());

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
    for (size_t k = 0; k < LU.cols(); ++k) {
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
        double pivot = LU(k, k);

        // Compute the k-th column of L, the coefficients lᵢₖ:
        for (size_t i = k + 1; i < LU.rows(); ++i)
            LU(i, k) /= pivot;

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
        for (size_t c = k + 1; c < LU.cols(); ++c)
            // Subtract lᵢₖ times the current pivot row A(k,:):
            for (size_t i = k + 1; i < LU.rows(); ++i)
                // A'(i,c) = 1·A(i,c) - lᵢₖ·A(k,c)
                LU(i, c) -= LU(i, k) * LU(k, c);

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
//! <!-- [NoPivotLU::compute_factorization] -->

/**
 * ## Implementation
 * @snippet this NoPivotLU::back_subs
 */
//! <!-- [NoPivotLU::back_subs] -->
void NoPivotLU::back_subs(const Matrix &B, Matrix &X) const {
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
        for (size_t r = LU.rows(); r-- > 0;) {
            X(r, i) = B(r, i);
            for (size_t c = r + 1; c < LU.cols(); ++c)
                X(r, i) -= LU(r, c) * X(c, i);
            X(r, i) /= LU(r, r);
        }
    }
}
//! <!-- [NoPivotLU::back_subs] -->

/**
 * ## Implementation
 * @snippet this NoPivotLU::forward_subs
 */
//! <!-- [NoPivotLU::forward_subs] -->
void NoPivotLU::forward_subs(const Matrix &B, Matrix &X) const {
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
        for (size_t r = 0; r < LU.rows(); ++r) {
            X(r, i) = B(r, i);
            for (size_t c = 0; c < r; ++c)
                X(r, i) -= LU(r, c) * X(c, i);
        }
    }
}
//! <!-- [NoPivotLU::forward_subs] -->

/**
 * ## Implementation
 * @snippet this NoPivotLU::solve_inplace
 */
//! <!-- [NoPivotLU::solve_inplace] -->
void NoPivotLU::solve_inplace(Matrix &B) const {
    // Solve the system AX = B, or LUX = B.
    //
    // Let UX = Z, and first solve LZ = B, which is a simple lower-triangular
    // system of equations.
    // Now that Z is known, solve UX = Z, which is a simple upper-triangular
    // system of equations.
    assert(is_factored());

    forward_subs(B, B); // overwrite B with Z
    back_subs(B, B);    // overwrite B (Z) with X
}
//! <!-- [NoPivotLU::solve_inplace] -->

// All implementations of the less interesting functions can be found here:
#include "boilerplate/NoPivotLU.ipp"
