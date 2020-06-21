#include "Matrix.hpp"

class HouseholderQR {
  public:
    HouseholderQR() = default;
    HouseholderQR(Matrix &&matrix) { compute(std::move(matrix)); }
    HouseholderQR(const Matrix &matrix) { compute(matrix); }

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
    /// The actual QR factorization algorithm.
    void compute_impl();
    /// Back substitution algorithm for solving upper-triangular systems RX = B.
    void back_subs(const Matrix &B, Matrix &X) const;

  public:
    /// Perform the QR factorization of the given matrix.
    void compute(Matrix &&matrix) {
        RW = std::move(matrix);
        R_diag.resize(RW.cols());
        compute_impl();
    }

    /// Perform the QR factorization of the given matrix.
    void compute(const Matrix &matrix) {
        RW = matrix;
        R_diag.resize(RW.cols());
        compute_impl();
    }

    /// Compute the product QᵀB, overwriting B with the result.
    void apply_QT_inplace(Matrix &B) const;
    /// Compute the product QᵀB.
    Matrix apply_QT(const Matrix &B) const {
        Matrix result = B;
        apply_QT_inplace(result);
        return result;
    }

    /// Compute the product QB, overwriting B with the result.
    void apply_Q_inplace(Matrix &X) const;
    /// Compute the product QB.
    Matrix apply_Q(const Matrix &X) const {
        Matrix result = X;
        apply_Q_inplace(result);
        return result;
    }

    /// Get the upper-triangular matrix R, reusing the internal storage.
    /// @warning    After calling this function, the HouseholderQR object is no
    ///             longer valid, because this function steals its storage.
    Matrix &&steal_R();

    /// Copy the upper-triangular matrix R to the given matrix.
    void get_R_inplace(Matrix &R) const;
    /// Get a copy of the upper-triangular matrix R.
    Matrix get_R() const & {
        Matrix R(RW.rows(), RW.cols());
        get_R_inplace(R);
        return R;
    }
    /// Get the upper-triangular matrix R.
    Matrix &&get_R() && { return steal_R(); }

    /// Compute the unitary matrix Q and copy it to the given matrix.
    void get_Q_inplace(SquareMatrix &R) const;
    /// Compute the unitary matrix Q.
    SquareMatrix get_Q() const {
        SquareMatrix Q(RW.rows());
        get_Q_inplace(Q);
        return Q;
    }

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

    /// Solve the system AX = B or QRX = B.
    /// Matrix B is overwritten with the result X. If the matrix A is square,
    /// no new allocations occur, and the storage of B is reused for X.
    /// If A is not square, new storage will be allocated for X.
    void solve_inplace(Matrix &B) const;
    /// Solve the system AX = B or QRX = B.
    Matrix solve(const Matrix &B) const;
    /// Solve the system AX = B or QRX = B.
    Matrix solve(Matrix &&B) const;
    /// Solve the system Ax = b or QRx = b.
    Vector solve(const Vector &B) const;
    /// Solve the system Ax = b or QRx = b.
    Vector solve(Vector &&B) const;
};

/// Print the Q and R matrices of a HouseholderQR object.
/// @related    HouseholderQR
std::ostream &operator<<(std::ostream &os, const HouseholderQR &qr);
