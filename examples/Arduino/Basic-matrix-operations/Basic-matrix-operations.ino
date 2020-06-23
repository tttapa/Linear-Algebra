#include <Linear_Algebra.h> // Include the library first

#include <HouseholderQR.hpp>
#include <Matrix.hpp> // Then include the necessary headers
#include <iostream>   // For printing using std::cout

void setup() {
  Serial.begin(115200);
  std::cout.precision(6); // How many significant digits to print

  std::cout << "Solve system Ax = v \n"
               "------------------- \n\n";
  SquareMatrix A = {
    {1, 6, 2},
    {3, 2, 1},
    {5, 1, 3},
  };
  Vector x = Vector::ones(3);
  Vector v = A * x;

  HouseholderQR qr(A); // QR factorization
  Vector x_solution = qr.solve(v);

  std::cout << "A = \n"
            << A << "\n"
            << "v = \n"
            << v << "\n"
            << "QR factorization of A: \n"
            << qr << "\n"
            << "solution x = \n"
            << x << std::endl;

  std::cout << "Basic matrix operations \n"
               "----------------------- \n\n";
  Matrix B = {
    {1, 2},
    {3, 4},
    {5, 6},
  };
  Matrix C = {
    {1, 2, 3},
    {4, 5, 6},
  };
  Matrix D = {
    {10, 11},
    {12, 13},
  };
  std::cout << " C×B + D = \n" << (C * B + D) << std::endl;

  std::cout << "Basic vector operations \n"
               "----------------------- \n\n";
  RowVector a = {1, 2, 3};
  RowVector b = {4, 6, 5};
  std::cout << "a   =      " << a << "\n"
            << "b   =      " << b << "\n"
            << "a×b =      " << a.cross(b) << "\n"
            << "a·b =      " << a.dot(b) << "\n\n"
            << std::endl;

  std::cout << "Element access \n"
               "-------------- \n\n";
  std::cout << "A[2,0] = " << A(2, 0) << std::endl;
  //                            │  └─ column
  //                            └──── row
  std::cout << "A[2,0] ← 100" << std::endl;
  A(2, 0) = 100; // assign a new value to the element
  std::cout << "A = \n" << A << std::endl;
}

void loop() {}