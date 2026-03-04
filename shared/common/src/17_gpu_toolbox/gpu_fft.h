#ifndef ABINIT_SHARED_COMMON_SRC_17_GPU_FFT_H
#define ABINIT_SHARED_COMMON_SRC_17_GPU_FFT_H

#if defined HAVE_CONFIG_H
#include "config.h"
#endif

#ifdef HAVE_GPU_CUDA
#include <cufft.h>
#include <cuda_api_error_check.h>
extern cufftHandle plan_fft[2];

#elif HAVE_GPU_HIP
#include <hipfft/hipfft.h>
#include <hip_api_error_check.h>
extern hipfftHandle plan_fft[2];

#elif defined(HAVE_MKL) && defined(HAVE_OPENMP_OFFLOAD)
#include <fftw/fftw3.h>
#include <fftw/offload/fftw3_omp_offload.h>

#else
#error Unknown GPU FFT implementation to be used
#endif

#endif // ABINIT_SHARED_COMMON_SRC_17_GPU_FFT_H

