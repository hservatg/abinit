/* gpu_fft_mkl.cpp */

/*
 * Copyright (C) 2008-2026 ABINIT Group
 * this file is distributed under the terms of the
 * gnu general public license, see ~abinit/COPYING
 * or http://www.gnu.org/copyleft/gpl.txt .
 * for the initials of contributors, see ~abinit/doc/developers/contributors.txt.
 *
 * The main goal of this file is to contain MKL encapsulation files
 * that will be callable from fortran routines
 *
 */

#include<assert.h>
#include<gpu_fft.h>

// We keep two plans, and for each plan we create the fwd/bwd variants
fftw_plan fft_l_plan[2][2];
fftwf_plan fft_f_plan[2][2];

enum {
  FFTW_PLAN_FORWARD_ID = 0,
  FFTW_PLAN_INVERSE_ID = 1
};

/* --- extracted from m_gpu_toolbox.F90
  enum, bind(C)
    enumerator :: FFT_R2C = 42   !  z'2a'     ! Real to Complex (interleaved)
    enumerator :: FFT_C2R = 44   !  z'2c'     ! Complex (interleaved) to Real
    enumerator :: FFT_C2C = 41   !  z'29'     ! Complex to Complex, interleaved
    enumerator :: FFT_D2Z = 106  !  z'6a'     ! Double to Double-Complex
    enumerator :: FFT_Z2D = 108  !  z'6c'     ! Double-Complex to Double
    enumerator :: FFT_Z2Z = 105  !  z'69'     ! Double-Complex to Double-Complex
  end enum
  --- */

constexpr int FFT_R2C = 42;
constexpr int FFT_C2R = 44;
constexpr int FFT_C2C = 41;
constexpr int FFT_D2Z = 106;
constexpr int FFT_Z2D = 108;
constexpr int FFT_Z2Z = 105;

/*=========================================================================*/
/* NAME
 *  gpu_fft_plan_many
 *
 * FUNCTION
 *  Initialize a FFT plan with custom dimension, strided and batch size.
 *
 * INPUTS
 *   rank      Dimensionality of the transform (1, 2, or 3).
 *   n         Array of size rank, describing the size of each dimension,
 *             n[0] being the size of the outermost and n[rank-1] innermost
 *             (contiguous) dimension of a transform.
 *   inembed   Pointer of size rank that indicates the storage dimensions
 *             of the input data in memory.
 *             If set to NULL all other advanced data layout parameters are ignored.
 *   istride   Indicates the distance between two successive input elements
 *             in the least significant (i.e., innermost) dimension
 *   idist     Indicates the distance between the first element of two
 *             consecutive signals in a batch of the input data
 *   onembed   Pointer of size rank that indicates the storage dimensions of
 *             the output data in memory.
 *             If set to NULL all other advanced data layout parameters are ignored.
 *   ostride   Indicates the distance between two successive output elements in
 *             the output array in the least significant (i.e., innermost) dimension
 *   odist     Indicates the distance between the first element of two
 *             consecutive signals in a batch of the output data
 *   type      The transform data type
 *             (e.g., FFT_R2C for single precision real to complex)
 *   batch     Batch size for this transform
 */
/*=========================================================================*/

extern "C"
void gpu_fft_plan_many_cpp(int *fft_plan_id, int *rank, int **n, int **inembed,
                       int *istride, int *idist, int **onembed, int *ostride,
                       int *odist, int *fft_type, int *batch){
#warning HSG check if in and out can be nullptr
	// Temporal pointers -- assigned to nullptr because not working with these
	// directly
	fftw_complex *lcplx_tmp[2] = { nullptr, nullptr };
	fftwf_complex *fcplx_tmp[2] = { nullptr, nullptr };
	float *f_tmp = nullptr;
	double *d_tmp = nullptr;

	// real 32bit
	if (*fft_type == FFT_C2C) {
		fft_f_plan[*fft_plan_id][FFTW_PLAN_FORWARD_ID] = fftwf_plan_many_dft (*rank, *n,
		  *batch, fcplx_tmp[0], *inembed, *istride, *idist, fcplx_tmp[1], *onembed,
		  *ostride, *odist, FFTW_FORWARD, FFTW_ESTIMATE);
		assert (fft_f_plan[*fft_plan_id][FFTW_PLAN_FORWARD_ID] == nullptr && "Failed to create FFT C2C plan/forward.");
		fft_f_plan[*fft_plan_id][FFTW_PLAN_INVERSE_ID] = fftwf_plan_many_dft (*rank, *n,
		  *batch, fcplx_tmp[0], *inembed, *istride, *idist, fcplx_tmp[1], *onembed,
		  *ostride, *odist, FFTW_BACKWARD, FFTW_ESTIMATE);
		assert (fft_f_plan[*fft_plan_id][FFTW_PLAN_INVERSE_ID] == nullptr && "Failed to create FFT C2C plan/backward.");
	} else if (*fft_type == FFT_C2R) {
#warning HSG replicated?
#if 0
		fft_f_plan[*fft_plan_id][FFTW_PLAN_FORWARD_ID] = fftwf_plan_many_dft_c2r (*rank, *n,
		  *batch, fcplx_tmp[0], *inembed, *istride, *idist, f_tmp, *onembed, *ostride, *odist,
		  FFTW_ESTIMATE);
		assert (fft_f_plan[*fft_plan_id][FFTW_PLAN_FORWARD_ID] == nullptr && "Failed to create FFT C2R plan/forward.");
#else
		fft_f_plan[*fft_plan_id][FFTW_PLAN_FORWARD_ID] = nullptr;
#endif
		fft_f_plan[*fft_plan_id][FFTW_PLAN_INVERSE_ID] = fftwf_plan_many_dft_c2r (*rank, *n,
		  *batch, fcplx_tmp[0], *inembed, *istride, *idist, f_tmp, *onembed, *ostride, *odist,
		  FFTW_ESTIMATE);
		assert (fft_f_plan[*fft_plan_id][FFTW_PLAN_INVERSE_ID] == nullptr && "Failed to create FFT C2R plan/backward.");
	} else if (*fft_type == FFT_R2C) {
#warning HSG replicated?
		fft_f_plan[*fft_plan_id][FFTW_PLAN_FORWARD_ID] = fftwf_plan_many_dft_r2c (*rank, *n,
		  *batch, f_tmp, *inembed, *istride, *idist, fcplx_tmp[0], *onembed, *ostride, *odist,
		  FFTW_ESTIMATE);
		assert (fft_f_plan[*fft_plan_id][FFTW_PLAN_FORWARD_ID] == nullptr && "Failed to create FFT R2C plan/forward.");
#if 0
		fft_f_plan[*fft_plan_id][FFTW_PLAN_INVERSE_ID] = fftwf_plan_many_dft_r2c (*rank, *n,
		  *batch, f_tmp, *inembed, *istride, *idist, fcplx_tmp[0], *onembed, *ostride, *odist,
		  FFTW_ESTIMATE);
		assert (fft_f_plan[*fft_plan_id][FFTW_PLAN_INVERSE_ID] == nullptr && "Failed to create FFT R2C plan/backward.");
#else
		fft_f_plan[*fft_plan_id][FFTW_PLAN_INVERSE_ID] = nullptr;
#endif
	}
	// real 64bit
	else if (*fft_type == FFT_Z2Z) {
		fft_l_plan[*fft_plan_id][FFTW_PLAN_FORWARD_ID] = fftw_plan_many_dft (*rank, *n,
		  *batch, lcplx_tmp[0], *inembed, *istride, *idist, lcplx_tmp[1], *onembed,
		  *ostride, *odist, FFTW_FORWARD, FFTW_ESTIMATE);
		assert (fft_l_plan[*fft_plan_id][FFTW_PLAN_FORWARD_ID] == nullptr && "Failed to create FFT Z2Z plan/forward.");
		fft_l_plan[*fft_plan_id][FFTW_PLAN_INVERSE_ID] = fftw_plan_many_dft (*rank, *n,
		  *batch, lcplx_tmp[0], *inembed, *istride, *idist, lcplx_tmp[1], *onembed,
		  *ostride, *odist, FFTW_BACKWARD, FFTW_ESTIMATE);
		assert (fft_l_plan[*fft_plan_id][FFTW_PLAN_INVERSE_ID] == nullptr && "Failed to create FFT Z2Z plan/backward.");
	} else if (*fft_type == FFT_Z2D) {
#warning HSG replicated?
#if 0
		fft_l_plan[*fft_plan_id][FFTW_PLAN_FORWARD_ID] = fftw_plan_many_dft_c2r (*rank, *n,
		  *batch, lcplx_tmp[0], *inembed, *istride, *idist, d_tmp, *onembed, *ostride, *odist,
		  FFTW_ESTIMATE);
		assert (fft_l_plan[*fft_plan_id][FFTW_PLAN_FORWARD_ID] == nullptr && "Failed to create FFT Z2D plan/forward.");
#else
		fft_l_plan[*fft_plan_id][FFTW_PLAN_FORWARD_ID] = nullptr;
#endif
		fft_l_plan[*fft_plan_id][FFTW_PLAN_INVERSE_ID] = fftw_plan_many_dft_c2r (*rank, *n,
		  *batch, lcplx_tmp[0], *inembed, *istride, *idist, d_tmp, *onembed, *ostride, *odist,
		  FFTW_ESTIMATE);
		assert (fft_l_plan[*fft_plan_id][FFTW_PLAN_INVERSE_ID] == nullptr && "Failed to create FFT Z2D plan/backward.");
	} else if (*fft_type == FFT_D2Z) {
#warning HSG replicated?
		fft_l_plan[*fft_plan_id][FFTW_PLAN_FORWARD_ID] = fftw_plan_many_dft_r2c (*rank, *n,
		  *batch, d_tmp, *inembed, *istride, *idist, lcplx_tmp[0], *onembed, *ostride, *odist,
		  FFTW_ESTIMATE);
		assert (fft_l_plan[*fft_plan_id][FFTW_PLAN_FORWARD_ID] == nullptr && "Failed to create FFT D2Z plan/forward.");
#if 0
		fft_l_plan[*fft_plan_id][FFTW_PLAN_INVERSE_ID] = fftw_plan_many_dft_r2c (*rank, *n,
		  *batch, d_tmp, *inembed, *istride, *idist, lcplx_tmp[0], *onembed, *ostride, *odist,
		  FFTW_ESTIMATE);
		assert (fft_l_plan[*fft_plan_id][FFTW_PLAN_INVERSE_ID] == nullptr && "Failed to create FFT D2Z plan/backward.");
#else
		fft_l_plan[*fft_plan_id][FFTW_PLAN_INVERSE_ID] = nullptr;
#endif
	}
}

/*=========================================================================*/
/* NAME
 *  gpu_fft_stream_synchronize
 *
 * FUNCTION
 *  Wait for any FFT operations still running on stream
 */
/*=========================================================================*/

extern "C" void gpu_fft_stream_synchronize_cpp(int *fft_plan_id)
{
#if 0
  HIP_API_CHECK( hipStreamSynchronize(stream_compute[*fft_plan_id]) );
#else
# warning HSG - this may be unnecessary? or how to rewrite this?
#endif
}


/*=========================================================================*/
// NAME
//  gpu_fft_plan_destroy
//
// FUNCTION
//  Destroy FFT plan (both variants, FWD and BWD)
//
/*=========================================================================*/

extern "C"
void gpu_fft_plan_destroy_cpp(int *fft_plan_id){
#warning HSG -- this needs additional check -- which plan - C(f) or Z(l), which direction (both or only one?) ?
#if 0
	fftw_destroy_plan(plan_fft[*fft_plan_id][FFTW_PLAN_FORWARD_ID]);
	fftw_destroy_plan(plan_fft[*fft_plan_id][FFTW_PLAN_INVERSE_ID]);
	plan_fft[*fft_plan_id][FFTW_PLAN_FORWARD_ID] = plan_fft[*fft_plan_id][FFTW_PLAN_INVERSE_ID] = nullptr;
#endif
}


/*=========================================================================*/
/* NAME
 *  gpu_fft_exec_z2z
 *
 * FUNCTION
 *  Run a Fast Fourrier Transform on double-complex input and output
 *
 * INPUTS
 *   idata       Pointer to the complex input data (in GPU memory) to transform
 *   odata       Pointer to the complex output data (in GPU memory)
 *   direction   The transform direction: FFT_FORWARD or FFT_INVERSE
 *
 * OUTPUT
 *   odata       Contains the complex Fourier coefficients
 */
/*=========================================================================*/

extern "C"
void gpu_fft_exec_z2z_cpp(int *fft_plan_id, void **idata, void **odata, int *direction){
	assert (*direction == FFTW_PLAN_FORWARD_ID || *direction == FFTW_PLAN_INVERSE_ID && "Invalid FFT plan direction");
	assert (fft_l_plan[*fft_plan_id][*direction] != nullptr && "Selected L plan was not created");
	#pragma omp dispatch
	fftw_execute_dft(fft_l_plan[*fft_plan_id][*direction], (fftw_complex*)*idata,
	  (fftw_complex*)*odata);
}


/*=========================================================================*/
/* NAME
 *  gpu_fft_exec_c2c
 *
 * FUNCTION
 *  Run a Fast Fourrier Transform on float complex input and output
 *
 * INPUTS
 *   idata       Pointer to the complex input data (in GPU memory) to transform
 *   odata       Pointer to the complex output data (in GPU memory)
 *   direction   The transform direction: FFT_FORWARD or FFT_INVERSE
 *
 * OUTPUT
 *   odata       Contains the complex Fourier coefficients
 */
/*=========================================================================*/

extern "C"
void gpu_fft_exec_c2c_cpp(int *fft_plan_id, void **idata, void **odata, int *direction){
	assert (*direction == FFTW_PLAN_FORWARD_ID || *direction == FFTW_PLAN_INVERSE_ID && "Invalid FFT plan direction");
	assert (fft_f_plan[*fft_plan_id][*direction] != nullptr && "Selected F plan was not created");
	#pragma omp dispatch
	fftwf_execute_dft(fft_f_plan[*fft_plan_id][*direction], (fftwf_complex*)*idata,
	  (fftwf_complex*)*odata);
}

