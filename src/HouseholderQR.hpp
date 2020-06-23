#include "Matrix.hpp"

/** 
 * @brief   QR factorization using Householder reflectors.
 * 
 * Factorizes an m×n matrix with m >= n into an m×m unitary factor Q and an m×n 
 * upper triangular factor R.
 * 
 * It can be used for solving square systems of equations or for finding a least
 * squares solution to an overdetermined system of equations.
 * 
 * This version does not use column pivoting, and is not rank-revealing.
 * 
 * @ingroup Factorizations
 */
class HouseholderQR {
  public:
    /// @name Constructors
    /// @{

    /// Default constructor.
    HouseholderQR() = default;
    /// Factorize the given matrix.
    HouseholderQR(const Matrix &matrix) { compute(matrix); }
    /// Factorize the given matrix.
    HouseholderQR(Matrix &&matrix) { compute(std::move(matrix)); }

    /// @}

  public:
    /// @name Factorization
    /// @{

    /// Perform the QR factorization of the given matrix.
    void compute(Matrix &&matrix);
    /// Perform the QR factorization of the given matrix.
    void compute(const Matrix &matrix);

    /// @}

  public:
    /// @name   Retrieving the Q factor
    /// @{

    /// Compute the product QᵀB, overwriting B with the result.
    void apply_QT_inplace(Matrix &B) const;
    /// Compute the product QᵀB.
    Matrix apply_QT(const Matrix &B) const;
    /// Compute the product QᵀB.
    Matrix &&apply_QT(Matrix &&B) const;

    /// Compute the product QB, overwriting B with the result.
    void apply_Q_inplace(Matrix &X) const;
    /// Compute the product QB.
    Matrix apply_Q(const Matrix &X) const;
    /// Compute the product QB.
    Matrix &&apply_Q(Matrix &&B) const;

    /// Compute the unitary matrix Q and copy it to the given matrix.
    void get_Q_inplace(SquareMatrix &Q) const;
    /// Compute the unitary matrix Q.
    SquareMatrix get_Q() const;

    /// @}

  public:
    /// @name   Retrieving the R factor
    /// @{

    /// Get the upper-triangular matrix R, reusing the internal storage.
    /// @warning    After calling this function, the HouseholderQR object is no
    ///             longer valid, because this function steals its storage.
    Matrix &&steal_R();

    /// Copy the upper-triangular matrix R to the given matrix.
    void get_R_inplace(Matrix &R) const;
    /// Get a copy of the upper-triangular matrix R.
    Matrix get_R() const &;
    /// Get the upper-triangular matrix R.
    Matrix &&get_R() && { return steal_R(); }

    /// @}

  public:
    /// @name   Solving systems of equations and least-squares problems
    /// @{

    /// Solve the system AX = B or QRX = B.
    /// Matrix B is overwritten with the result X. If the matrix A is square,
    /// no new allocations occur, and the storage of B is reused for X.
    /// If A is not square, new storage will be allocated for X.
    void solve_inplace(Matrix &B) const;
    /// Solve the system AX = B or QRX = B.
    Matrix solve(const Matrix &B) const;
    /// Solve the system AX = B or QRX = B.
    Matrix &&solve(Matrix &&B) const;
    /// Solve the system Ax = b or QRx = b.
    Vector solve(const Vector &B) const;
    /// Solve the system Ax = b or QRx = b.
    Vector &&solve(Vector &&B) const;

    /// @}

  public:
    /// @name   Access to internal representation
    /// @{

    /// Check if this object contains a valid factorization.
    bool is_factored() const { return state == Factored; }

    /// Get the internal storage of the strict upper-triangular part of R and
    /// the Householder reflector vectors W.
    const Matrix &get_RW() const & { return RW; }
    /// @copydoc    get_RW
    Matrix &&get_RW() && { return std::move(RW); }
    /// Get the internal storage of the diagonal elements of R.
    const Vector &get_R_diag() const & { return R_diag; }
    /// @copydoc    get_R_diag
    Vector &&get_R_diag() && { return std::move(R_diag); }

    /// @}

  private:
    /// The actual QR factorization algorithm.
    void compute_factorization();
    /// Back substitution algorithm for solving upper-triangular systems RX = B.
    void back_subs(const Matrix &B, Matrix &X) const;

  private:
    /// Result of a Householder QR factorization: stores the strict
    /// upper-triangular part of matrix R and the full matrix of scaled
    /// Householder reflection vectors W. The reflection vectors have norm √2.
    Matrix RW;
    /// Contains the diagonal elements of R.
    Vector R_diag;

    enum State {
        NotFactored = 0,
        Factored = 1,
    } state = NotFactored;
};

/// Print the Q and R matrices of a HouseholderQR object.
/// @related    HouseholderQR
std::ostream &operator<<(std::ostream &os, const HouseholderQR &qr);
