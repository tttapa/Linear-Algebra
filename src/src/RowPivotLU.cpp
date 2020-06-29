#include <linalg/RowPivotLU.hpp>

#include <cassert>

/**
 * @pre     `LU` contains the matrix A to be factorized
 * @pre     `P` contains the identity matrix (no permutations)
 * @pre     `LU.rows() == LU.cols()`
 * @pre     `P.size() == LU.rows()`
 * 
 * @post    The complete upper-triangular part of `LU` contains the full 
 *          upper-triangular matrix U and the strict lower-triangular part of 
 *          matrix L. The diagonal elements of L are implicitly 1.
 * @post    `get_L() * get_U() == get_P() * A`
 *          (up to rounding errors)
 * 
 * @see     @ref NoPivotLU::compute_factorization() for an introduction to the
 *          LU factorization. The basics presented there will not be reiterated
 *          here.
 * 
 * ## Implementation
 * @snippet this RowPivotLU::compute_factorization
 */
//! <!-- [RowPivotLU::compute_factorization] -->
void RowPivotLU::compute_factorization() {
    // For the intermediate calculations, we'll be working with LU.
    // It is initialized to the square n×n matrix to be factored.

    assert(LU.rows() == LU.cols());
    assert(P.size() == LU.rows());

    // The goal of the LU factorization algorithm is to repeatedly apply
    // transformations Lₖ to the matrix A to eventually end up with an upper-
    // triangular matrix U. When row pivoting is used, the rows of A are
    // permuted using a permutation matrix P:
    //
    //     Lₙ'⋯L₂'L₁'PA = U
    //
    // The main steps of the algorithm are exactly the same as the original
    // LU algorithm explained in LU.cpp, and will not be repeated here.
    // The only difference is that instead of using the diagonal element as the
    // pivot, rows are swapped so that the element with the largest magnitude
    // ends up on the diagonal and can be used as the pivot.

    // Loop over all columns of A:
    for (size_t k = 0; k < LU.cols(); ++k) {
        // In the following comments, k = [1, n], because this is more intuitive
        // and it follows the usual mathematical convention.
        // In the code, however, array indices start at zero, so k = [0, n-1].

        // On each iteration, the largest element on or below the diagonal in
        // the current (k-th) column will be used as the pivot.
        // To this end, the k-th row and the row that contains the largest
        // element are swapped, and the swapping is stored in the permutation
        // matrix, so that it can later be undone, when solving systems of
        // equations for example.

        // Find the largest element (in absolute value)
        double max_elem = std::abs(LU(k, k));
        size_t max_index = k;
        for (size_t i = k + 1; i < LU.rows(); ++i) {
            double abs_elem = std::abs(LU(i, k));
            if (abs_elem > max_elem) {
                max_elem = abs_elem;
                max_index = i;
            }
        }

        // Select the index of the element that is largest in absolute value as
        // the new pivot index.
        // If this index is not the diagonal element, rows have to be swapped:
        if (max_index != k) {
            P(k) = max_index;           // save the permutation
            LU.swap_rows(k, max_index); // actually perfrom the permutation
        }

        // Note how all columns of the two rows are permuted, not just the
        // columns greater than k. You might wonder how that'll ever work out
        // correctly in the end.
        // Recall that for the LU factorization without pivoting, the result was
        //     Lₙ⋯L₂L₁A = U.
        // When pivoting is used, however, the matrix is permuted before each
        // elimination step:
        //
        //     LₙPₙ⋯L₂P₂L₁P₁A = U
        //
        // Luckily, the product can be reordered, and all permutation matrices
        // Pₖ can be grouped together.
        // Without loss of generality, consider the pivoted LU factorization of
        // a 3×3 matrix:
        //
        //     L₃P₃L₂P₂L₁P₁A = U
        //
        // Now introduce the following permuted matrices Lₖ':
        //
        //     L₃' = L₃
        //     L₂' = P₃L₂P₃⁻¹
        //     L₁' = P₃P₂L₁P₂⁻¹P₃⁻¹
        //
        // You can then easily see that the following equation holds:
        //
        //     L₃'L₂'L₁'P₂P₃P₁A = U
        //   ⇔ L₃(P₃L₂P₃⁻¹)(P₃P₂L₁P₂⁻¹P₃⁻¹)P₃P₂P₁A = U
        //   ⇔ L₃P₃L₂P₂L₁P₁A = U
        //
        // Furthermore, the matrices Lₖ' have the same structure as Lₖ, because
        // only rows below the k-th pivot are permuted.
        //
        // All of this allows us to group all row permutations into a single
        // permutation matrix P, and use the same algorithm and storage format
        // as for the unpivoted case.
        //
        // The factors Lₖ' are computed implicitly by applying the row
        // permutations to the entire matrix that stores both the U and L
        // factors, rather than just to the elements of the trailing submatrix.

        // The rest of the algorithm is identical to the one explained in
        // NoPivotLU.cpp.

        double pivot = LU(k, k);

        // Compute the k-th column of L, the coefficients lᵢₖ:
        for (size_t i = k + 1; i < LU.rows(); ++i)
            LU(i, k) /= pivot;

        // Update the trailing submatrix A'(k+1:n,k+1:n) = LₖA(k+1:n,k+1:n):
        for (size_t c = k + 1; c < LU.cols(); ++c)
            // Subtract lᵢₖ times the current pivot row A(k,:):
            for (size_t i = k + 1; i < LU.rows(); ++i)
                // A'(i,c) = 1·A(i,c) - lᵢₖ·A(k,c)
                LU(i, c) -= LU(i, k) * LU(k, c);

        // Because of the row pivoting, zero pivots are no longer an issue,
        // since the pivot is always chosen to be the largest possible element.
        // When the matrix is singular, the algorithm will still fail, of
        // course.
    }
    state = Factored;
    valid_LU = true;
    valid_P = true;
}
//! <!-- [RowPivotLU::compute_factorization] -->

/**
 * ## Implementation
 * @snippet this RowPivotLU::back_subs
 */
//! <!-- [RowPivotLU::back_subs] -->
void RowPivotLU::back_subs(const Matrix &B, Matrix &X) const {
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
//! <!-- [RowPivotLU::back_subs] -->

/**
 * ## Implementation
 * @snippet this RowPivotLU::forward_subs
 */
//! <!-- [RowPivotLU::forward_subs] -->
void RowPivotLU::forward_subs(const Matrix &B, Matrix &X) const {
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
//! <!-- [RowPivotLU::forward_subs] -->

/**
 * ## Implementation
 * @snippet this RowPivotLU::solve_inplace
 */
//! <!-- [RowPivotLU::solve_inplace] -->
void RowPivotLU::solve_inplace(Matrix &B) const {
    // Solve the system AX = B, PAX = PB or LUX = PB.
    //
    // Let UX = Z, and first solve LZ = PB, which is a simple lower-triangular
    // system of equations.
    // Now that Z is known, solve UX = Z, which is a simple upper-triangular
    // system of equations.
    assert(is_factored());

    P.permute_rows(B);
    forward_subs(B, B); // overwrite B with Z
    back_subs(B, B);    // overwrite B (Z) with X
}
//! <!-- [RowPivotLU::solve_inplace] -->

// All implementations of the less interesting functions can be found here:
#include "boilerplate/RowPivotLU.ipp"