#pragma once

#include "Matrix.hpp"
#include "PermutationMatrix.hpp"

/** 
 * @brief   LU factorization with row pivoting.
 * 
 * Factorizes a square matrix into a lower triangular and an upper-triangular
 * factor.
 * 
 * This version uses row pivoting, but it is not rank-revealing.
 * 
 * @ingroup Factorizations
 */
class RowPivotLU {
  public:
    /// @name Constructors
    /// @{

    /// Default constructor.
    RowPivotLU() = default;
    /// Factorize the given matrix.
    RowPivotLU(const SquareMatrix &matrix) { compute(matrix); }
    /// Factorize the given matrix.
    RowPivotLU(SquareMatrix &&matrix) { compute(std::move(matrix)); }

    /// @}

  public:
    /// @name Factorization
    /// @{

    /// Perform the LU factorization of the given matrix.
    void compute(SquareMatrix &&matrix);
    /// Perform the LU factorization of the given matrix.
    void compute(const SquareMatrix &matrix);

    /// @}

  public:
    /// @name   Retrieving the L factor
    /// @{

    /// Get the lower-triangular matrix L, reusing the internal storage.
    /// @warning    After calling this function, the LU object is no
    ///             longer valid, because this function steals its storage.
    ///             Stealing both L and P is allowed (if you do not steal U,
    ///             because it shares storage with L).
    SquareMatrix &&steal_L();

    /// Copy the lower-triangular matrix L to the given matrix.
    void get_L_inplace(Matrix &L) const;
    /// Get a copy of the lower-triangular matrix L.
    SquareMatrix get_L() const &;
    /// Get the lower-triangular matrix L.
    SquareMatrix &&get_L() && { return steal_L(); }

    /// @}

  public:
    /// @name   Retrieving the U factor
    /// @{

    /// Get the upper-triangular matrix U, reusing the internal storage.
    /// @warning    After calling this function, the LU object is no
    ///             longer valid, because this function steals its storage.
    ///             Stealing both U and P is allowed (if you do not steal L,
    ///             because it shares storage with U).
    SquareMatrix &&steal_U();

    /// Copy the upper-triangular matrix U to the given matrix.
    void get_U_inplace(Matrix &U) const;
    /// Get a copy of the upper-triangular matrix U.
    SquareMatrix get_U() const &;
    /// Get the upper-triangular matrix U.
    SquareMatrix &&get_U() && { return steal_U(); }

    /// @}

  public:
    /// @name   Retrieving the P factor
    /// @{

    /// Get the permutation matrix P, reusing the internal storage.
    /// @warning    After calling this function, the LU object is no
    ///             longer valid, because this function steals its storage.
    ///             Stealing P and either L or U (not both) is allowed.
    PermutationMatrix &&steal_P();

    /// Get a copy of the permutation matrix P.
    PermutationMatrix get_P() const & { return P; }
    /// Get the permutation matrix P.
    PermutationMatrix &&get_P() && { return steal_P(); }

    /// @}

  public:
    /// @name   Solving systems of equations problems
    /// @{

    /// Solve the system AX = B or LUX = B.
    /// Matrix B is overwritten with the result X.
    void solve_inplace(Matrix &B) const;
    /// Solve the system AX = B or LUX = B.
    Matrix solve(const Matrix &B) const;
    /// Solve the system AX = B or LUX = B.
    Matrix &&solve(Matrix &&B) const;
    /// Solve the system Ax = b or LUx = b.
    Vector solve(const Vector &B) const;
    /// Solve the system Ax = b or LUx = b.
    Vector &&solve(Vector &&B) const;

    /// @}

  public:
    /// @name   Access to internal representation
    /// @{

    /// Check if this object contains a factorization.
    bool is_factored() const { return state == Factored; }

    /// Check if this object contains valid L and U factors.
    bool has_LU() const { return has_LU_; }

    /// Check if this object contains a valid permutation matrix P.
    bool has_P() const { return has_P_; }

    /// Get the internal storage of the upper-triangular matrix U and the strict
    /// lower-triangular part of matrix L.
    /// @warning    After calling this function, the LU object is no longer
    ///             valid, because this function steals its storage.
    ///             Stealing both LU and P is allowed (but not L or U
    ///             individually).
    SquareMatrix &&steal_LU();

    /// Get a copy of the internal storage of the upper-triangular matrix U and
    /// the strict lower-triangular part of matrix L.
    const SquareMatrix &get_LU() const & { return LU_; }
    /// Get the internal storage of the upper-triangular matrix U and the strict
    /// lower-triangular part of matrix L.
    SquareMatrix &&get_LU() && { return steal_LU(); }

    /// @}

  private:
    /// The actual LU factorization algorithm.
    void compute_factorization();
    /// Back substitution algorithm for solving upper-triangular systems UX = B.
    void back_subs(const Matrix &B, Matrix &X) const;
    /// Forward substitution algorithm for solving lower-triangular systems
    /// LX = B.
    void forward_subs(const Matrix &B, Matrix &X) const;

  private:
    /// Result of a LU factorization: stores the upper-triangular
    /// matrix U and the strict lower-triangular part of matrix L. The diagonal
    /// elements of L are implicitly 1.
    SquareMatrix LU_;
    /// The permutation of A that maximizes pivot size.
    PermutationMatrix P = PermutationMatrix::RowPermutation;

    enum State {
        NotFactored = 0,
        Factored = 1,
    } state = NotFactored;

    bool has_LU_ = false;
    bool has_P_ = false;
};

/// Print the L and U matrices of an LU object.
/// @related    RowPivotLU
std::ostream &operator<<(std::ostream &os, const RowPivotLU &lu);
