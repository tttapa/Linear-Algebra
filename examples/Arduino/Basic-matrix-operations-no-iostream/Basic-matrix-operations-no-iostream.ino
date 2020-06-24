/**
 * @example Basic-matrix-operations-no-iostream.ino
 * @brief   Arduino sketch that demonstrates some of the basic features.
 * 
 * ---
 * 
 * Aimed at Arduino boards without STL ostream support. Tested on Teensy 3.2,
 * compiles for Teensy 3.0, 3.5, 3.6 as well.
 * 
 * Expected output
 * ---------------
 * 
 * ~~~
 * Solve system Ax = v
 * -------------------
 * 
 * A = 
 *    1.000000e+00   6.000000e+00   2.000000e+00
 *    3.000000e+00   2.000000e+00   1.000000e+00
 *    5.000000e+00   1.000000e+00   3.000000e+00
 * 
 * v =
 *    9.000000e+00
 *    6.000000e+00
 *    9.000000e+00
 * 
 * QR factorization of A:
 * Q =
 *   -1.690309e-01   9.636759e-01  -2.067787e-01
 *   -5.070926e-01   9.486964e-02   8.566545e-01
 *   -8.451543e-01  -2.496570e-01  -4.726370e-01
 * R =
 *   -5.916080e+00  -2.873524e+00  -3.380617e+00
 *    0.000000e+00   5.722137e+00   1.273250e+00
 *    0.000000e+00   0.000000e+00  -9.748137e-01
 * 
 * solution x =
 *    1.000000e+00
 *    1.000000e+00
 *    1.000000e+00
 * 
 * Basic matrix operations
 * -----------------------
 * 
 * C×B + D =
 *    3.200000e+01   3.900000e+01
 *    6.100000e+01   7.700000e+01
 * 
 * Basic vector operations
 * -----------------------
 * 
 * a   =         1.000000e+00   2.000000e+00   3.000000e+00
 * b   =         4.000000e+00   6.000000e+00   5.000000e+00
 * a×b =        -8.000000e+00   7.000000e+00  -2.000000e+00
 * a·b =      31.00
 * 
 * Element access
 * --------------
 * 
 * A[2,0] = 5.00
 * A[2,0] ← 100
 *    1.000000e+00   6.000000e+00   2.000000e+00
 *    3.000000e+00   2.000000e+00   1.000000e+00
 *    1.000000e+02   1.000000e+00   3.000000e+00
 * 
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
  Serial.println();

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