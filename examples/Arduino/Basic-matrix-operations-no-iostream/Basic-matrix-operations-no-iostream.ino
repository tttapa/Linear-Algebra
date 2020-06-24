/**
 * @example Basic-matrix-operations.ino
 * @brief   Arduino sketch that demonstrates some of the basic features.
 * 
 * ---
 * 
 * Aimed at Arduino boards without STL ostream support. Tested on Teensy 3.2.
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

#include <HouseholderQR.hpp> // Then include the necessary headers
#include <Matrix.hpp>

void setup() {
  Serial.begin(115200);
  while (!Serial)
    ;

  Serial.println("Solve system Ax = v");
  Serial.println("-------------------\n");
  SquareMatrix A = {
    {1, 6, 2},
    {3, 2, 1},
    {5, 1, 3},
  };
  Vector x = Vector::ones(3);
  Vector v = A * x;

  HouseholderQR qr(A); // QR factorization
  Vector x_solution = qr.solve(v);

  Serial.println("A = ");
  A.print(Serial);
  Serial.println("\nv =");
  v.print(Serial);
  Serial.println("\nQR factorization of A:");
  Serial.println("Q =");
  qr.get_Q().print(Serial);
  Serial.println("R =");
  qr.get_R().print(Serial);
  Serial.println("\nsolution x =");
  x_solution.print(Serial);
  Serial.println();

  Serial.println("Basic matrix operations");
  Serial.println("-----------------------\n");
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
  Matrix E = C * B + D;
  Serial.println("C×B + D =");
  E.print(Serial);

  Serial.println("Basic vector operations");
  Serial.println("-----------------------\n");
  RowVector a = {1, 2, 3};
  RowVector b = {4, 6, 5};
  RowVector c = a.cross(b);
  double d = a.dot(b);
  Serial.print("a   =      ");
  a.print(Serial);
  Serial.print("b   =      ");
  b.print(Serial);
  Serial.print("a×b =      ");
  c.print(Serial);
  Serial.print("a·b =      ");
  Serial.println(d);
  Serial.println();

  Serial.println("Element access");
  Serial.println("--------------\n");
  Serial.print("A[2,0] = ");
  Serial.println(A(2, 0));
  //               │  └─ column
  //               └──── row
  Serial.println("A[2,0] ← 100");
  A(2, 0) = 100; // assign a new value to the element
  A.print(Serial);
}

void loop() {}
