#include <linalg/HouseholderQR.hpp>

#include <cassert>
#include <iomanip>
#include <iostream>

void HouseholderQR::compute(Matrix &&matrix) {
    RW = std::move(matrix);
    R_diag.resize(RW.cols());
    compute_factorization();
}

void HouseholderQR::compute(const Matrix &matrix) {
    RW = matrix;
    R_diag.resize(RW.cols());
    compute_factorization();
}

Matrix HouseholderQR::apply_QT(const Matrix &B) const {
    Matrix result = B;
    apply_QT_inplace(result);
    return result;
}

Matrix &&HouseholderQR::apply_QT(Matrix &&B) const {
    apply_QT_inplace(B);
    return std::move(B);
}

Matrix HouseholderQR::apply_Q(const Matrix &X) const {
    Matrix result = X;
    apply_Q_inplace(result);
    return result;
}

Matrix &&HouseholderQR::apply_Q(Matrix &&B) const {
    apply_Q_inplace(B);
    return std::move(B);
}

void HouseholderQR::get_R_inplace(Matrix &R) const {
    assert(is_factored());
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

Matrix HouseholderQR::get_R() const & {
    Matrix R(RW.rows(), RW.cols());
    get_R_inplace(R);
    return R;
}

Matrix &&HouseholderQR::steal_R() {
    state = NotFactored;
    for (size_t r = 0; r < RW.cols(); ++r) {
        for (size_t c = 0; c < r; ++c)
            RW(r, c) = 0;
        RW(r, r) = R_diag(r);
    }
    for (size_t r = RW.cols(); r < RW.rows(); ++r) {
        for (size_t c = 0; c < RW.cols(); ++c)
            RW(r, c) = 0;
    }
    return std::move(RW);
}

void HouseholderQR::get_Q_inplace(SquareMatrix &Q) const {
    assert(Q.rows() == RW.rows());
    assert(Q.cols() == RW.rows());
    Q.fill_identity();
    apply_Q_inplace(Q);
}

SquareMatrix HouseholderQR::get_Q() const {
    SquareMatrix Q(RW.rows());
    get_Q_inplace(Q);
    return Q;
}

Matrix HouseholderQR::solve(const Matrix &B) const {
    Matrix B_cpy = apply_QT(B);
    Matrix X(RW.cols(), B.cols());
    back_subs(B_cpy, X);
    return X;
}

Matrix &&HouseholderQR::solve(Matrix &&B) const {
    solve_inplace(B);
    return std::move(B);
}

Vector HouseholderQR::solve(const Vector &b) const {
    return Vector(solve(static_cast<const Matrix &>(b)));
}

Vector &&HouseholderQR::solve(Vector &&b) const {
    solve_inplace(b);
    return std::move(b);
}

// LCOV_EXCL_START

std::ostream &operator<<(std::ostream &os, const HouseholderQR &qr) {
    if (!qr.is_factored()) {
        os << "Not factored." << std::endl;
        return os;
    }

    Matrix Q = qr.get_Q();
    os << "Q = " << std::endl;
    Q.print(os);

    // Output field width (characters)
    int w = os.precision() + 9;

    const auto &RW = qr.get_RW();
    const auto &R_diag = qr.get_R_diag();
    os << "R = " << std::endl;
    for (size_t r = 0; r < RW.cols(); ++r) {
        for (size_t c = 0; c < r; ++c)
            os << std::setw(w) << 0;
        os << std::setw(w) << R_diag(r);
        for (size_t c = r + 1; c < RW.cols(); ++c)
            os << std::setw(w) << RW(r, c);
        os << std::endl;
    }
    for (size_t r = RW.cols(); r < RW.rows(); ++r) {
        for (size_t c = 0; c < RW.cols(); ++c)
            os << std::setw(w) << 0;
    }
    return os;
}

// LCOV_EXCL_STOP
