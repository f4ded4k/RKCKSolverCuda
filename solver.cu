// source : https://arxiv.org/pdf/1611.02274.pdf

#include <solver.h>
#include <stdio.h>
#include <system_params.h>

// Utility function for ODE system.
__device__ double I(const double t) {
  if (t < 50.0) {
    return 0.01;
  } else {
    return 1.0;
  }
}

// F(t,y,g) = h*f(t,y,g) of Y' = f(t,y,g).
// t,y,g : Parameters required to compute f(t,y,g).
// h : Step size.
// F : Returns h*f(t,y,g).
__device__ void FuncF(const double t, const double *y, const double *g,
                      const double h, OUT double *F) {
  double A = y[0], B = y[1], C = y[2];
  double K_IA = g[0], K_FAA_ = g[1], K_FBB_ = g[2], K_CB = g[3], K_AC = g[4],
         K_BC_ = g[5], F_A = g[6], F_B = g[7];
  F[0] = h * (I(t) * K_IA * (1 - A) / (1 - A + K_IA) -
              F_A * K_FAA_ * A / (A + K_FAA_));
  F[1] = h * (-F_B * K_FBB_ * B / (B + K_FBB_) +
              C * K_CB * (1 - B) / (1 - B + K_CB));
  F[2] =
      h * (A * K_AC * (1 - C) / (1 - C + K_AC) - B * K_BC_ * C / (C + K_BC_));
}

// Computes y values for a single timestep.
// t,y,g : Parameters required to compute f(t,y,g).
// h : Step size.
// yEnd : Returns y values at (t+h).
// yErr : Returns error estimated using 4 & 5-th order methods.
__device__ void solverStep(const double t, const double *y, const double *g,
                           const double h, OUT double *yEnd, OUT double *yErr) {
  // source : http://www.elegio.it/mc2/rk/doc/p201-cash-karp.pdf
  // Butcher tablau for 5th order adaptive Cash-Karp method.
  const double a[6] = {0.0, 0.2, 0.3, 0.6, 1.0, 0.875};
  const double b1[1] = {0.2};
  const double b2[2] = {3.0 / 40, 9.0 / 40};
  const double b3[3] = {3.0 / 10, -9.0 / 10, 6.0 / 5};
  const double b4[4] = {-11.0 / 54, 5.0 / 2, -70.0 / 27, 35.0 / 27};
  const double b5[5] = {1631.0 / 55296, 175.0 / 512, 575.0 / 13824,
                        44275.0 / 110592, 253.0 / 4096};
  const double c[6] = {37.0 / 378,  0.0, 250.0 / 621,
                       125.0 / 594, 0.0, 512.0 / 1771};
  const double cs[6] = {2825.0 / 27648,  0.0,           18575.0 / 48384,
                        13525.0 / 55296, 277.0 / 14336, 0.25};

  // Computation of K1.
  double k1[NumEq];
  FuncF(t, y, g, h, k1);

  // Stores temprary K values.
  double yTemp[NumEq];

  // Computation of K2.
  double k2[NumEq];
  for (int i = 0; i < NumEq; ++i) {
    yTemp[i] = y[i] + b1[0] * k1[0];
  }
  FuncF(t + a[1] * h, yTemp, g, h, k2);

  // Computation of K3.
  double k3[NumEq];
  for (int i = 0; i < NumEq; ++i) {
    yTemp[i] = y[i] + b2[0] * k1[i] + b2[1] * k2[i];
  }
  FuncF(t + a[2] * h, yTemp, g, h, k3);

  // Computation of K4.
  double k4[NumEq];
  for (int i = 0; i < NumEq; ++i) {
    yTemp[i] = y[i] + b3[0] * k1[i] + b3[1] * k2[i] + b3[2] * k3[i];
  }
  FuncF(t + a[3] * h, yTemp, g, h, k4);

  // Computation of K5.
  double k5[NumEq];
  for (int i = 0; i < NumEq; ++i) {
    yTemp[i] =
        y[i] + b4[0] * k1[i] + b4[1] * k2[i] + b4[2] * k3[i] + b4[3] * k4[i];
  }
  FuncF(t + a[4] * h, yTemp, g, h, k5);

  // Computation of K6.
  double k6[NumEq];
  for (int i = 0; i < NumEq; ++i) {
    yTemp[i] = y[i] + b5[0] * k1[i] + b5[1] * k2[i] + b5[2] * k3[i] +
               b5[3] * k4[i] + b5[4] * k5[i];
  }
  FuncF(t + a[5] * h, yTemp, g, h, k6);

  // Computation of yEnd & yErr
  for (int i = 0; i < NumEq; ++i) {
    yEnd[i] = y[i] + c[0] * k1[i] + c[1] * k2[i] + c[2] * k3[i] + c[3] * k4[i] +
              c[4] * k5[i] + c[5] * k6[i];
    yErr[i] =
        fabs(yEnd[i] - (y[i] + cs[0] * k1[i] + cs[1] * k2[i] + cs[2] * k3[i] +
                        cs[3] * k4[i] + cs[4] * k5[i] + cs[5] * k6[i]));
  }
}

// Drives solverStep until y is found at t=t1.
// t0,t1 : Initial and final time to compute y at respectively.
// g : Parameter array required for computing f(t,y,g).
// y : Recieved y values at t=t0 and returned at t=t1
__device__ void solverDriver(const double t0, const double t1, const double *g,
                             OUT double *y) {
  // Define range of valid substep size.
  double hMax = fabs(t1 - t0);
  double hMin = 1.0e-10;

  // Define initial substep size.
  double h = 0.5 * hMax;
  double t = t0;

  // Repeat substep until end reached.
  while (t < t1) {
    // Make sure h is within bounds.
    h = fmin(fmax(hMin, fmin(hMax, h)), t1 - t);

    // Run a single substep.
    double yEnd[NumEq], yErr[NumEq];
    solverStep(t, y, g, h, yEnd, yErr);

    // Error calculation.
    double K1[NumEq];
    FuncF(t, y, g, h, K1);
    double maxErr = 0.0;
    bool isNAN = false;
    for (int i = 0; i < NumEq; ++i) {
      isNAN |= isnan(yErr[i]);
      maxErr = fmax(maxErr, yErr[i] / (fabs(y[i]) + fabs(K1[i]) + Tiny));
    }
    maxErr /= Eps;
    isNAN |= isnan(maxErr);

    // If error is unacceptable, recompute with smaller step size.
    if (maxErr >= 1.0 || isNAN) {
      if (isNAN) {
        h *= P1;
      } else {
        h = fmax(h * P1, Safety * h * pow(maxErr, Pshrnk));
      }
    }
    // Else accept the step and update step size & y values.
    else {
      t += h;
      if (maxErr > Errcon) {
        h = Safety * h * pow(maxErr, Pgrow);
      } else {
        h *= P2;
      }
      for (int i = 0; i < NumEq; ++i) {
        y[i] = yEnd[i];
      }
    }
  }
}

// Entry point from CPU program to the solver.
// t0 : Initial time.
// t1 : Final time.
// gGlobal : Global array containing parameters.
// yGlobal : Global array containing y values at t=t0
__global__ void solverMain(const double t0, const double t1,
                           const double *gGlobal, OUT double *yGlobal) {
  // Compute index of ODE system to be solved by this thread.
  const int currNumODE = threadIdx.x + (blockDim.x * blockIdx.x);
  if (currNumODE < NumODE) {
    // Allocate and populate local arrays for g & y.
    double gLocal[NumParam], yLocal[NumEq];
    for (int i = 0; i < NumParam; ++i) {
      gLocal[i] = gGlobal[currNumODE + NumODE * i];
    }
    for (int i = 0; i < NumEq; ++i) {
      yLocal[i] = yGlobal[currNumODE + NumODE * i];
    }

    // Perform a full step.
    solverDriver(t0, t1, gLocal, yLocal);

    // Move computed y values back to the global array.
    for (int i = 0; i < NumEq; ++i) {
      yGlobal[currNumODE + NumODE * i] = yLocal[i];
    }
  }
}