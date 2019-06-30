// source : https://arxiv.org/pdf/1611.02274.pdf

#ifndef _SYSTEM_PARAMS_H_
#define _SYSTEM_PARAMS_H_

// Number of ODE systems to run with different parameters.
constexpr int NumODE = 1;
// Number of equations in each ODE system.
constexpr int NumEq = 3;
// Number of parameters in each ODE system.
constexpr int NumParam = 8;
// Number of time steps.
constexpr int NumTimeSteps = 200;

#endif // !_SYSTEM_PARAMS_H_