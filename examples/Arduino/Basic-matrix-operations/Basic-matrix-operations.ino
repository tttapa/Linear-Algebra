/**
 * @example Basic-matrix-operations.ino
 * @brief   Arduino sketch that demonstrates some of the basic features.
 * 
 * ---
 * 
 * Aimed at Arduino boards with STL ostream support. Tested on ESP32 and Teensy
 * 4.0 (beta). Also compiles for Arduino SAMD (Zero) and Arduino Nano 33 BLE,
 * but untested.
 * 
 * Expected output
 * ---------------
 * 
 * ~~~
 * Solve system Ax = v 
 * ------------------- 
 * 
 * A = 
 *               1              6              2
 *               3              2              1
 *               5              1              3
 * 
 * v = 
 *               9
 *               6
 *               9
 * 
 * QR factorization of A: 
 * Q = 
 *       -0.169031       0.963676      -0.206779
 *       -0.507093      0.0948696       0.856654
 *       -0.845154      -0.249657      -0.472637
 * R = 
 *        -5.91608       -2.87352       -3.38062
 *               0        5.72214        1.27325
 *               0              0      -0.974814
 * 
 * solution x = 
 *               1
 *               1
 *               1
 * 
 * Basic matrix operations 
 * ----------------------- 
 * 
 *  C×B + D = 
 *              32             39
 *              61             77
 * 
 * Basic vector operations 
 * ----------------------- 
 * 
 * a   =                    1              2              3
 * 
 * b   =                    4              6              5
 * 
 * a×b =                   -8              7             -2
 * 
 * a·b =      31
 * 
 * 
 * Element access 
 * -------------- 
 * 
 * A[2,0] = 5
 * A[2,0] ← 100
 * A = 
 *               1              6              2
 *               3              2              1
 *             100              1              3
 * ~~~
 */

#include <Linear_Algebra.h> // Include the library first

// Then include the necessary headers:
#include <include/linalg/HouseholderQR.hpp> 
#include <include/linalg/Matrix.hpp>

// For printing using arduino::cout:
#include <include/linalg/Arduino/ArduinoCout.hpp>
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