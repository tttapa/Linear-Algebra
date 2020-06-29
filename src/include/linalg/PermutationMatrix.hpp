#pragma once

#include "Matrix.hpp"

/// @addtogroup MatVec
/// @{

/// Class that represents matrices that permute the rows or columns of other
/// matrices.
/// Stored in an efficient manner with O(n) memory requirements.
class PermutationMatrix {

    /// Container to store the elements of the permutation matrix internally.
    using storage_t = util::storage_t<size_t>;

  public:
    enum Type {
        Unspecified = 0,       ///< Can be used for permuting rows or columns.
        RowPermutation = 1,    ///< Can be used for permuting rows only.
        ColumnPermutation = 2, ///< Can be used for permuting columns only.
    };

  public:
    /// @name   Constructors and assignment
    /// @{

    /// Default constructor.
    PermutationMatrix() = default;

    /// Create an empty permutation matrix with the given type.
    PermutationMatrix(Type type) : type(type) {}

    /// Create a permutation matrix without permutations.
    PermutationMatrix(size_t rows, Type type = Unspecified)
        : storage(rows), type(type) {
        fill_identity();
    }

    /// Create a permutation matrix with the given permutation.
    PermutationMatrix(std::initializer_list<size_t> init,
                      Type type = Unspecified);
    /// Assign the given permutation to the matrix.
    PermutationMatrix &operator=(std::initializer_list<size_t> init);

    /// Default copy constructor.
    PermutationMatrix(const PermutationMatrix &) = default;
    /// Move constructor.
    PermutationMatrix(PermutationMatrix &&);

    /// Default copy assignment.
    PermutationMatrix &operator=(const PermutationMatrix &) = default;
    /// Move assignment.
    PermutationMatrix &operator=(PermutationMatrix &&);

    /// @}

  public:
    /// @name   Matrix size
    /// @{

    /// Get the size of the permutation matrix.
    size_t size() const { return storage.size(); }
    /// Get the number of rows of the permutation matrix.
    size_t rows() const { return size(); }
    /// Get the number of columns of the permutation matrix.
    size_t cols() const { return size(); }
    /// Get the number of elements in the matrix:
    size_t num_elems() const { return size(); }
    /// Resize the permutation matrix.
    void resize(size_t size) { storage.resize(size); }

    /// @}

  public:
    /// @name   Element access
    /// @{

    /// Get the element at the given position in the swap sequence.
    /// If the k-th element is i, that is `P(k) == i`, this means that the k-th
    /// step of the swapping algorithm will swap `i` and `k`.
    size_t &operator()(size_t index) { return storage[index]; }
    /// @copydoc    operator()(size_t)
    const size_t &operator()(size_t index) const { return storage[index]; }

    /// @}

  public:
    /// @name   Transposition
    /// @{

    /// Reverse the order of the permutations.
    void reverse() { reverse_ = !reverse_; }

    /// Transpose or invert the permutation matrix.
    void transpose_inplace() { reverse(); }

    /// Check if the permutation should be applied in reverse.
    bool is_reversed() const { return reverse_; }

    /// Get the type of permutation matrix (whether it permutes rows or columns,
    /// or unspecified).
    Type get_type() const { return type; }
    /// Set the type of permutation matrix (whether it permutes rows or columns,
    /// or unspecified).
    void set_type(Type type) { this->type = type; }

    /// @}

  public:
    /// @name   Conversion to a full matrix or a permutation
    /// @{

    /// Convert a permutation matrix into a full matrix.
    SquareMatrix to_matrix(Type type = Unspecified) const;

    /// Type that represents a permutation (in the mathematical sense, a
    /// permutation of the integers 0 through n-1).
    using Permutation = storage_t;

    /// Convert a permutation matrix into a mathematical permutation
    Permutation to_permutation() const;

    /// @}

  public:
    /// @name   Applying the permutation to matrices
    /// @{

    /// Apply the permutation to the columns of matrix A.
    void permute_columns(Matrix &A) const;
    /// Apply the permutation to the rows of matrix A.
    void permute_rows(Matrix &A) const;

    /// @}

  public:
    /// @name   Memory management
    /// @{

    /// Set the size to zero, and deallocate the storage.
    void clear_and_deallocate() {
        storage_t().swap(this->storage); // replace storage with empty storage
        // temporary storage goes out of scope and deallocates original storage
    }

    /// @}

  public:
    /// @name    Generating permutations
    /// @{

    /// Return a random permutation of the integers 0 through length-1.
    static Permutation
    random_permutation(size_t length,
                       std::default_random_engine::result_type seed =
                           std::default_random_engine::default_seed);

    /// Return the identity permutation (0, 1, 2, 3, ..., length-1).
    static Permutation identity_permutation(size_t length);

    /// @}

  public:
    /// @name   Filling matrices
    /// @{

    /// Fill the matrix as an identity permutation.
    void fill_identity() { std::iota(begin(), end(), size_t(0)); }

    /// Create a permutation matrix from the given permutation.
    /// @note   This isn't a very fast method, it's mainly used for tests.
    ///         Internally, the permutation matrix is represented by a sequence
    ///         of swap operations. Converting from this representation to a
    ///         mathematical permutation is fast, but the other way around
    ///         requires O(n²) operations (with the naive implementation used 
    ///         here, anyway).
    void fill_from_permutation(Permutation permutation);

    /// Fill the matrix with a random permutation.
    /// @note   This isn't a very fast method, it's mainly used for tests.
    void fill_random(std::default_random_engine::result_type seed =
                         std::default_random_engine::default_seed) {
        fill_from_permutation(random_permutation(size(), seed));
    }

    /// @}

  public:
    /// @name   Create special matrices
    /// @{

    /// Create an identity permutation matrix.
    static PermutationMatrix identity(size_t rows, Type type = Unspecified) {
        PermutationMatrix p(rows, type);
        return p;
    }

    /// @copydoc    fill_from_permutation
    static PermutationMatrix from_permutation(Permutation permutation,
                                              Type type = Unspecified) {
        PermutationMatrix p(permutation.size(), type);
        p.fill_from_permutation(std::move(permutation));
        return p;
    }

    /// Create a random permutation matrix.
    /// @note   This isn't a very fast method, it's mainly used for tests.
    static PermutationMatrix
    random(size_t rows, Type type = Unspecified,
           std::default_random_engine::result_type seed =
               std::default_random_engine::default_seed) {
        PermutationMatrix p(rows, type);
        p.fill_random(seed);
        return p;
    }

    /// @}

  public:
    /// @name   Iterators
    /// @{

    /// Get the iterator to the first element of the swapping sequence.
    storage_t::iterator begin() { return storage.begin(); }
    /// Get the iterator to the first element of the swapping sequence.
    storage_t::const_iterator begin() const { return storage.begin(); }
    /// Get the iterator to the first element of the swapping sequence.
    storage_t::const_iterator cbegin() const { return storage.begin(); }

    /// Get the iterator to the element past the end of the swapping sequence.
    storage_t::iterator end() { return storage.end(); }
    /// Get the iterator to the element past the end of the swapping sequence.
    storage_t::const_iterator end() const { return storage.end(); }
    /// Get the iterator to the element past the end of the swapping sequence.
    storage_t::const_iterator cend() const { return storage.end(); }

    /// @}

  public:
    /// @name   Printing
    /// @{

    /// Print a permutation matrix.
    /// @param  os
    ///         The stream to print to.
    /// @param  precision
    ///         The number of significant figures to print.
    ///         (0 = auto)
    /// @param  width
    ///         The width of each element (number of characters).
    ///         (0 = auto)
    void print(std::ostream &os, uint8_t precision = 0,
               uint8_t width = 0) const;

    /// @}

  protected:
    storage_t storage;
    bool reverse_ = false;
    Type type = Unspecified;
};

/// @}

/// Print a permutation matrix.
/// @related    PermutationMatrix
std::ostream &operator<<(std::ostream &os, const PermutationMatrix &M);

// :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: //

/// @addtogroup MatMul
/// @{

/// Left application of permutation matrix (P permutes rows of A).
inline Matrix operator*(const PermutationMatrix &P, const Matrix &A) {
    Matrix result = A;
    P.permute_rows(result);
    return result;
}
/// Left application of permutation matrix (P permutes rows of A).
inline Matrix &&operator*(const PermutationMatrix &P, Matrix &&A) {
    P.permute_rows(A);
    return std::move(A);
}
/// Right application of permutation matrix (P permutes columns of A).
inline Matrix operator*(const Matrix &A, const PermutationMatrix &P) {
    Matrix result = A;
    P.permute_columns(result);
    return result;
}
/// Right application of permutation matrix (P permutes columns of A).
inline Matrix &&operator*(Matrix &&A, const PermutationMatrix &P) {
    P.permute_columns(A);
    return std::move(A);
}

/// Left application of permutation matrix (P permutes rows of A).
inline SquareMatrix operator*(const PermutationMatrix &P,
                              const SquareMatrix &A) {
    SquareMatrix result = A;
    P.permute_rows(result);
    return result;
}
/// Left application of permutation matrix (P permutes rows of A).
inline SquareMatrix &&operator*(const PermutationMatrix &P, SquareMatrix &&A) {
    P.permute_rows(A);
    return std::move(A);
}
/// Right application of permutation matrix (P permutes columns of A).
inline SquareMatrix operator*(const SquareMatrix &A,
                              const PermutationMatrix &P) {
    SquareMatrix result = A;
    P.permute_columns(result);
    return result;
}
/// Right application of permutation matrix (P permutes columns of A).
inline SquareMatrix &&operator*(SquareMatrix &&A, const PermutationMatrix &P) {
    P.permute_columns(A);
    return std::move(A);
}

/// Left application of permutation matrix (P permutes rows of v).
inline Vector operator*(const PermutationMatrix &P, const Vector &v) {
    Vector result = v;
    P.permute_rows(result);
    return result;
}
/// Left application of permutation matrix (P permutes rows of v).
inline Vector &&operator*(const PermutationMatrix &P, Vector &&v) {
    P.permute_rows(v);
    return std::move(v);
}

/// Right application of permutation matrix (P permutes columns of v).
inline RowVector operator*(const RowVector &v, const PermutationMatrix &P) {
    RowVector result = v;
    P.permute_columns(result);
    return result;
}
/// Right application of permutation matrix (P permutes columns of v).
inline RowVector &&operator*(RowVector &&v, const PermutationMatrix &P) {
    P.permute_columns(v);
    return std::move(v);
}

/// @}

/// @addtogroup MatTrans
/// @{

/// Transpose a permutation matrix (inverse permutation).
inline PermutationMatrix transpose(const PermutationMatrix &P) {
    PermutationMatrix result = P;
    result.transpose_inplace();
    return result;
}
/// Transpose a permutation matrix (inverse permutation).
inline PermutationMatrix &&transpose(PermutationMatrix &&P) {
    P.transpose_inplace();
    return std::move(P);
}

/// @}

//                              Implementations                               //
// -------------------------------------------------------------------------- //

inline PermutationMatrix::PermutationMatrix(PermutationMatrix &&other) {
    *this = std::move(other);
}

inline PermutationMatrix &
PermutationMatrix::operator=(PermutationMatrix &&other) {
    // By explicitly defining move assignment, we can be sure that the object
    // that's being moved from has a consistent state.
    this->storage = std::move(other.storage);
    std::swap(this->type, other.type);
    std::swap(this->reverse_, other.reverse_);
    other.clear_and_deallocate();
    return *this;
}

inline PermutationMatrix::PermutationMatrix(std::initializer_list<size_t> init,
                                            Type type)
    : type(type) {
    *this = init;
}

inline SquareMatrix PermutationMatrix::to_matrix(Type type) const {
    // TODO: I'm sure this can be sped up
    Type actual_type = type == Unspecified ? this->type : type;
    assert(actual_type != Unspecified);
    if (actual_type == RowPermutation) {
        SquareMatrix P = SquareMatrix::identity(size());
        permute_rows(P);
        return P;
    } else if (actual_type == ColumnPermutation) {
        SquareMatrix P = SquareMatrix::identity(size());
        permute_columns(P);
        return P;
    }
    assert(false);
    return {};
}

inline PermutationMatrix::Permutation
PermutationMatrix::to_permutation() const {
    Permutation p = identity_permutation(size());
    auto &This = *this;
    if (is_reversed()) {
        // Count down
        for (size_t i = size(); i-- > 0;)
            if (i != This(i))
                std::swap(p[i], p[This(i)]);
    } else {
        // Count up
        for (size_t i = 0; i < size(); ++i)
            if (i != This(i))
                std::swap(p[i], p[This(i)]);
    }
    return p;
}

inline PermutationMatrix &
PermutationMatrix::operator=(std::initializer_list<size_t> init) {
    storage_t permutation(init.size());
    std::copy(init.begin(), init.end(), permutation.begin());
    fill_from_permutation(std::move(permutation));

    return *this;
}

inline void PermutationMatrix::permute_columns(Matrix &A) const {
    assert(A.cols() == size());
    assert(get_type() != RowPermutation);
    auto &This = *this;
    if (is_reversed()) {
        // Count down
        for (size_t i = size(); i-- > 0;)
            if (i != This(i))
                A.swap_columns(i, This(i));
    } else {
        // Count up
        for (size_t i = 0; i < size(); ++i)
            if (i != This(i))
                A.swap_columns(i, This(i));
    }
}

inline void PermutationMatrix::permute_rows(Matrix &A) const {
    assert(A.rows() == size());
    assert(get_type() != ColumnPermutation);
    auto &This = *this;
    if (is_reversed()) {
        // Count down
        for (size_t i = size(); i-- > 0;)
            if (i != This(i))
                A.swap_rows(i, This(i));
    } else {
        // Count up
        for (size_t i = 0; i < size(); ++i)
            if (i != This(i))
                A.swap_rows(i, This(i));
    }
}
