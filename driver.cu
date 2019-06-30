// source : https://arxiv.org/pdf/1611.02274.pdf

#include <fstream>
#include <iostream>
#include <string>

#include <cuda.h>
#include <cuda_runtime.h>

#include <solver.h>
#include <system_params.h>

// Identifier for out parameters in a function signature.
#define OUT
// Logs all cuda runtime status messages.
#define PRINTErr(cudaExpr, OS) OS << cudaGetErrorString(cudaExpr) << std::endl;

// Reads initial values from filepath file.
// filepath : Path to the file.
void getInitialValues(const std::string &filepath, OUT double *y) {
  std::ifstream ifs(filepath, std::ios::in);
  int idx = 0;
  double val;
  while (ifs >> val) {
    y[idx++] = val;
  }
  ifs.close();
}

// Reads parameter values from filepath file.
// filepath : Path to the file.
void getParamValues(const std::string &filePath, OUT double *g) {
  std::ifstream ifs(filePath, std::ios::in);
  int idx = 0;
  double val;
  while (ifs >> val) {
    g[idx++] = val;
  }
  ifs.close();
}

// Entry point from the OS.
int main() {
  // Get parameters and initial values into memory.
  double y0[NumODE * NumEq], g[NumODE * NumParam];
  getInitialValues("params\\initial_vals.txt", y0);
  getParamValues("params\\param_vals.txt", g);

  // Define kernel launch parameters.
  int blockSize;
  if (NumODE < 4194304) {
    blockSize = 64;
  } else if (NumODE < 8388608) {
    blockSize = 128;
  } else if (NumODE < 16777216) {
    blockSize = 256;
  } else {
    blockSize = 512;
  }
  dim3 dimBlock(blockSize);
  dim3 dimGrid((NumODE + dimBlock.x - 1) / dimBlock.x);

  // Define time steps to save results at.
  double TimeSteps[NumTimeSteps];
  for (int i = 0; i < NumTimeSteps; ++i) {
    TimeSteps[i] = (double)i / 10.0;
  }

  // Storage for all Y values
  double y[NumODE][NumEq][NumTimeSteps];

  // Fill in the initial Y values known.
  for (int i = 0; i < NumODE; ++i) {
    for (int j = 0; j < NumEq; ++j) {
      y[i][j][0] = y0[i + NumODE * j];
    }
  }

  std::ofstream logOfs("log.txt", std::ios::out);

  // Allocate & move data to GPU memory.
  double *deviceY, *deviceG;
  PRINTErr(cudaMalloc(&deviceY, sizeof(double) * NumODE * NumEq), logOfs);
  PRINTErr(cudaMalloc(&deviceG, sizeof(double) * NumODE * NumParam), logOfs);
  PRINTErr(cudaMemcpy(deviceY, y0, NumODE * NumEq * sizeof(double),
                      cudaMemcpyHostToDevice),
           logOfs);
  PRINTErr(cudaMemcpy(deviceG, g, NumODE * NumParam * sizeof(double),
                      cudaMemcpyHostToDevice),
           logOfs);

  // Call gpu solver for each adjacent timestep pairs.
  for (int i = 0; i < NumTimeSteps - 1; ++i) {
    solverMain<<<dimGrid, dimBlock>>>(TimeSteps[i], TimeSteps[i + 1], deviceG,
                                      deviceY);
    PRINTErr(cudaDeviceSynchronize(), logOfs);
    PRINTErr(cudaMemcpy(y0, deviceY, NumODE * NumEq * sizeof(double),
                        cudaMemcpyDeviceToHost),
             logOfs);
    for (int j = 0; j < NumODE; ++j) {
      for (int k = 0; k < NumEq; ++k) {
        y[j][k][i + 1] = y0[j + NumODE * k];
      }
    }
  }

  // Deallocate gpu memory.
  PRINTErr(cudaFree(deviceY), logOfs);
  PRINTErr(cudaFree(deviceG), logOfs);

  for (int i = 0; i < NumTimeSteps; ++i)
    std::cout << y[0][0][i] << std::endl;

  logOfs.close();
  return 0;
}