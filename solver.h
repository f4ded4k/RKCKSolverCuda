// source : https://arxiv.org/pdf/1611.02274.pdf

#ifndef _SOLVER_H_
#define _SOLVER_H_

#include <cuda_runtime.h>

#define OUT

// Solver constant parameters.
constexpr double Eps = 1.0e-10;
constexpr double Tiny = 1.0e-30;
constexpr double Safety = 0.9;
constexpr double Pgrow = -0.2;
constexpr double Pshrnk = -0.25;
constexpr double Errcon = 1.89e-4;
constexpr double P1 = 0.1;
constexpr double P2 = 5.0;

// F(t,y,g) = h*f(t,y,g) of Y' = f(t,y,g).
// t,y,g : Parameters required to compute f(t,y,g).
// h : Step size.
// F : Returns h*f(t,y,g)
__device__ void FuncF(const double t, const double *y, const double *g,
                      const double h, OUT double *F);

// Computes y values for a single timestep.
// t,y,g : Parameters required to compute f(t,y,g).
// h : Step size.
// yEnd : Returns y values at (t+h).
// yErr : Returns error estimated using 4 & 5-th order methods.
__device__ void solverStep(const double t, const double *y, const double *g,
                           const double h, OUT double *yEnd,
                           OUT double *yErr);

// Drives solverStep until y is found at t=t1.
// t0,t1 : Initial and final time to compute y at respectively.
// g : Parameter array required for computing f(t,y,g).
// y : Recieved y values at t=t0 and returned at t=t1
__device__ void solverDriver(const double t0, const double t1,
                             const double *g, OUT double *y);

// Entry point from CPU program to the solver.
// t0 : Initial time.
// t1 : Final time.
// gGlobal : Global array containing parameters.
// yGlobal : Global array containing y values at t=t0
__global__ void solverMain(const double t0, const double t1,
                           const double *gGlobal, OUT double *yGlobal);

#endif // !_SOLVER_H_