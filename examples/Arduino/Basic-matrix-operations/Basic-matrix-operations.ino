#include <Linear_Algebra.h> // Include the library first

#include <HouseholderQR.hpp> // Then include the necessary headers
#include <Matrix.hpp>

#include <Arduino/ArduinoCout.hpp> // For printing using arduino::cout
using arduino::cout;

void setup() {
  Serial.begin(115200);
  while (!Serial)
    ;
  cout.precision(6); // How many significant digits to print

  cout << "Solve system Ax = v \n"
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

  cout << "A = \n"
       << A << "\n"
       << "v = \n"
       << v << "\n"
       << "QR factorization of A: \n"
       << qr << "\n"
       << "solution x = \n"
       << x << std::endl;

  cout << "Basic matrix operations \n"
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
  cout << " C×B + D = \n" << (C * B + D) << std::endl;

  cout << "Basic vector operations \n"
          "----------------------- \n\n";
  RowVector a = {1, 2, 3};
  RowVector b = {4, 6, 5};
  cout << "a   =      " << a << "\n"
       << "b   =      " << b << "\n"
       << "a×b =      " << a.cross(b) << "\n"
       << "a·b =      " << a.dot(b) << "\n\n"
       << std::endl;

  cout << "Element access \n"
          "-------------- \n\n";
  cout << "A[2,0] = " << A(2, 0) << std::endl;
  //                       │  └─ column
  //                       └──── row
  cout << "A[2,0] ← 100" << std::endl;
  A(2, 0) = 100; // assign a new value to the element
  cout << "A = \n" << A << std::endl;
}

void loop() {}