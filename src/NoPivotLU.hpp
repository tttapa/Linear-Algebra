#pragma once

#include "Matrix.hpp"

/** 
 * @brief   LU factorization without pivoting.
 * 
 * Factorizes a square matrix into a lower triangular and an upper-triangular
 * factor.
 * 
 * This version does not use row pivoting, and is not rank-revealing.
 * 
 * @warning **Never** use this factorization, it is not numerically stable and
 *          will fail completely if a zero pivot is encountered. This algorithm
 *          is included for educational purposes only. Use a pivoted LU 
 *          factorization or a QR factorization instead.
 * 
 * @ingroup Factorizations
 */
class NoPivotLU {
  public:
    /// @name Constructors
    /// @{

    /// Default constructor.
    NoPivotLU() = default;
    /// Factorize the given matrix.
    NoPivotLU(const SquareMatrix &matrix) { compute(matrix); }
    /// Factorize the given matrix.
    NoPivotLU(SquareMatrix &&matrix) { compute(std::move(matrix)); }

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
    /// @warning    After calling this function, the NoPivotLU object is no
    ///             longer valid, because this function steals its storage.
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
    /// @warning    After calling this function, the NoPivotLU object is no
    ///             longer valid, because this function steals its storage.
    SquareMatrix &&steal_U();

    /// Copy the upper-triangular matrix U to the given matrix.
    void get_U_inplace(Matrix &U) const;
    /// Get a copy of the upper-triangular matrix U.
    SquareMatrix get_U() const &;
    /// Get the upper-triangular matrix U.
    SquareMatrix &&get_U() && { return steal_U(); }

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
    bool has_LU() const { return is_factored(); }

    /// Get the internal storage of the upper-triangular matrix U and the strict
    /// lower-triangular part of matrix L.
    /// @warning    After calling this function, the LU object is no longer
    ///             valid, because this function steals its storage.
    SquareMatrix &&steal_LU();

    /// Get a copy of the internal storage of the upper-triangular matrix U and
    /// the strict lower-triangular part of matrix L.
    const SquareMatrix &get_LU() const & { return LU; }
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
    SquareMatrix LU;

    enum State {
        NotFactored = 0,
        Factored = 1,
    } state = NotFactored;
};

/// Print the L and U matrices of an NoPivotLU object.
/// @related    NoPivotLU
std::ostream &operator<<(std::ostream &os, const NoPivotLU &lu);
