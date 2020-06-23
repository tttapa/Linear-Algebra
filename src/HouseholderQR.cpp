#include "HouseholderQR.hpp"

#include <cassert>
#include <iomanip>
#include <iostream>

void HouseholderQR::compute_impl() {
    // For the intermediate calculations, we'll be working with RW.
    // It is initialized to the rectangular matrix to be factored.
    // At the end of this function, RW will contain the strict
    // upper-triangular part of the matrix R (without the diagonal),
    // and the complete scaled matrix of reflection vectors W, which is a
    // lower-triangular matrix. The diagonal of R is stored separately in
    // R_diag.

    assert(RW.rows() >= RW.cols());
    assert(R_diag.size() == RW.cols());

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

            // Save the first component of xₕ:
            R_diag(k) = 0;
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

void HouseholderQR::apply_QT_inplace(Matrix &B) const {
    assert(is_factored());
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

void HouseholderQR::apply_Q_inplace(Matrix &X) const {
    assert(is_factored());
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

void HouseholderQR::solve_inplace(Matrix &B) const {
    apply_QT_inplace(B);

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

    int w = os.precision() + 9;

    Matrix Q = qr.get_Q();
    os << "Q = " << std::endl;
    Q.print(os, w);

    const auto &RW     = qr.get_RW();
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
