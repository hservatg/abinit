/* gpu_linalg_openmp.cpp */

/*
 * Copyright (C) 2026 ABINIT Group (MMancini,FDahm)
 * this file is distributed under the terms of the
 * gnu general public license, see ~abinit/COPYING
 * or http://www.gnu.org/copyleft/gpl.txt .
 * for the initials of contributors, see ~abinit/doc/developers/contributors.txt.
 *
 * The main goal of this file is to contain cublas and magma encapsulation routines,
 * that will be callable from fortran routines
 *
 */

#include <gpu_linalg.h>

#include <mkl.h>
#include <mkl_omp_offload.h>

#include <stdio.h>

static bool linalg_multi_init = true;

/*=========================================================================*/
// NAME
//  gpu_linalg_init
//
// FUNCTION
//  Initialisation of linalg environnement on GPU
//
/*=========================================================================*/

extern "C" void gpu_linalg_init_()
{
  linalg_multi_init = false;
}

extern "C" void gpu_linalg_init_multi_()
{
  linalg_multi_init = true;
}

/*=========================================================================*/
// NAME
//  gpu_linalg_shutdown
//
// FUNCTION
//  Close linalg environnement on GPU
//
/*=========================================================================*/

extern "C" void gpu_linalg_shutdown_()
{
  if(linalg_multi_init) {
    linalg_multi_init = false;
  }
}

/*=========================================================================*/
// NAME
//  gpu_linalg_stream_synchronize
//
// FUNCTION
//  Wait for any linalg operations still running on stream
//
/*=========================================================================*/

extern "C" void gpu_linalg_stream_synchronize_()
{
	fprintf (stdout, "EMPTY CALL! %s:%d:%s\n", __FILE__, __LINE__, __func__);
}

/*=========================================================================*/
// NAME
//  gpu_xgemm
//
// FUNCTION
//  Compute a scalar-matrix-matrix product and return a scalar-matrix product on GPU
//  c = alpha * op(a) * op(b) + beta * c
//
// INPUTS
//  cplx  = 1 if real 2 if complex
//  transa= from of op(a) to be used in the matrix multiplication
//  transb= from of op(b) to be used in the matrix multiplication
//  m     = number of rows of the matrix op(a) and of the matrix c
//  n     = number of rows of the matrix op(b) and the number of columns of the matrix c
//  k     = number of columns of the matrix op(a) and the number of rows of the matrix op(b)
//  alpha = alpha scalar coefficient for matrix op(a)
//  a_gpu = pointer to gpu memory location of  matrix a
//  lda   = first dimension of a
//  b_gpu = pointer to gpu memory location of  matrix b
//  ldb   = first dimension of b
//  beta  = beta scalar coefficient for matrix c
//  c_gpu = pointer to gpu memory location of  matrix c
//  ldc   = first dimension of c
//
// OUTPUT
//  c_gpu     = c_gpu matrix
//
/*=========================================================================*/

extern "C" void gpu_xgemm_(int* cplx, char *transA, char *transB, int *N, int *M, int *K,
                           /* cuDoubleComplex */ void *alpha,
                           void **A_ptr, int *lda, void** B_ptr, int *ldb,
                           /* cuDoubleComplex */ void *beta, void** C_ptr, int *ldc)
{
#warning "Check routine signature"
	fprintf (stdout, "EMPTY CALL! %s:%d:%s\n", __FILE__, __LINE__, __func__);
} // gpu_xgemm

/*=========================================================================*/
// NAME
//  gpu_xgemm_strided_batched
//
// FUNCTION
//  Compute a batched scalar-matrix-matrix product and return a scalar-matrix product on GPU.
//  Meant to be used on non-contiguous matrixes with data is uniformly split in the same number of batches in each matrix.
//  c = alpha * op(a) * op(b) + beta * c
//
// INPUTS
//  cplx       = 1 if real 2 if complex
//  transa     = from of op(a) to be used in the matrix multiplication
//  transb     = from of op(b) to be used in the matrix multiplication
//  m          = number of rows of the matrix op(a) and of the matrix c
//  n          = number of rows of the matrix op(b) and the number of columns of the matrix c
//  k          = number of columns of the matrix op(a) and the number of rows of the matrix op(b)
//  alpha      = alpha scalar coefficient for matrix op(a)
//  a_gpu      = pointer to gpu memory location of  matrix a
//  lda        = first dimension of a
//  strideC    = stride between each batch in matrix a
//  b_gpu      = pointer to gpu memory location of  matrix b
//  ldb        = first dimension of b
//  strideC    = stride between each batch in matrix b
//  beta       = beta scalar coefficient for matrix c
//  c_gpu      = pointer to gpu memory location of  matrix c
//  ldc        = first dimension of c
//  strideC    = stride between each batch in matrix c
//  batchCount = number of batches in any matrix
//
// OUTPUT
//  c_gpu     = c_gpu matrix
//
/*=========================================================================*/

extern "C" void gpu_xgemm_strided_batched_(int* cplx, char *transA, char *transB,
                           int *N, int *M, int *K,
                           /* cuDoubleComplex */ void *alpha,
                           void **A_ptr, int *lda, int *strideA,
                           void** B_ptr, int *ldb, int *strideB,
                           /* cuDoubleComplex */ void *beta,
                           void** C_ptr, int *ldc, int *strideC, int *batchCount,
                           int *stream_id)
{
#warning "Check routine signature"
	fprintf (stdout, "EMPTY CALL! %s:%d:%s\n", __FILE__, __LINE__, __func__);
} // gpu_xgemm_strided_batched


/*=========================================================================*/
// NAME
//  gpu_xginv_strided_
//
// FUNCTION
//  Compute many matrixes inverses on GPU in batch on GPU.
//
// INPUTS
//  cplx       = 1 if real 2 if complex
//  n          = number of rows of the matrix op(b) and the number of columns of the matrix c
//  A_ptr      = pointer to gpu memory location of  matrixes a
//  lda        = first dimension of a
//  strideA    = stride between each matrix a
//  batchCount = number of batches in any matrix
//  W_ptr      = pointer to gpu memory locatiion for a work buffer, sized after A
//
// OUTPUT
//  A_ptr     = inverted matrixes pointers
//
/*=========================================================================*/

extern "C" void gpu_xginv_strided_(int* cplx,
                           int *N,
                           void **A_ptr, int *lda, int *strideA,
                           int *batchCount, void **W_ptr)
{
	fprintf (stdout, "EMPTY CALL! %s:%d:%s\n", __FILE__, __LINE__, __func__);
} // gpu_xginv_strided_

/*=========================================================================*/
// NAME
//  gpu_xsymm
//
// FUNCTION
//  Compute a symmetric scalar-matrix-matrix product and return a scalar-matrix product on GPU
//  c = alpha * op(a) * op(b) + beta * c    , if side == L
//  c = alpha * op(b) * op(a) + beta * c    , if side == R
//
// INPUTS
//  cplx  = 1 if real 2 if complex
//  side  = Specifies whether op(a) appears on the left or right of x for
//          the operation to be performed as follows:
//              L or l op(a)*x = alpha*b
//              R or r x*op(a) = alpha*b
//  uplo  = Specifies whether the matrix a is an upper or lower triangular
//          matrix as follows:
//              U or u Matrix a is an upper triangular matrix.
//              L or l Matrix a is a lower triangular matrix
//  m     = number of rows of the matrix op(a) and of the matrix c
//  n     = number of rows of the matrix op(b) and the number of columns of the matrix c
//  k     = number of columns of the matrix op(a) and the number of rows of the matrix op(b)
//  alpha = alpha scalar coefficient for matrix op(a)
//  a_gpu = pointer to gpu memory location of  matrix a
//  lda   = first dimension of a
//  b_gpu = pointer to gpu memory location of  matrix b
//  ldb   = first dimension of b
//  beta  = beta scalar coefficient for matrix c
//  c_gpu = pointer to gpu memory location of  matrix c
//  ldc   = first dimension of c
//
// OUTPUT
//  c_gpu     = c_gpu matrix
//
/*=========================================================================*/

extern "C" void gpu_xsymm_(int* cplx, char *side, char *uplo, int *N, int *M,
                           /* cuDoubleComplex */ void *alpha,
                           void **A_ptr, int *lda, void** B_ptr, int *ldb,
                           /* cuDoubleComplex */ void *beta, void** C_ptr, int *ldc)
{
#warning "Check routine signature"
	fprintf (stdout, "EMPTY CALL! %s:%d:%s\n", __FILE__, __LINE__, __func__);
} // gpu_xsymm_omp

/*=========================================================================*/
// NAME
//  gpu_xhemm
//
// FUNCTION
//  Compute a Hermitian scalar-matrix-matrix product and return a scalar-matrix product on GPU
//  c = alpha * op(a) * op(b) + beta * c    , if side == L
//  c = alpha * op(b) * op(a) + beta * c    , if side == R
//
// INPUTS
//  cplx  = 1 if real 2 if complex
//  side  = Specifies whether op(a) appears on the left or right of x for
//          the operation to be performed as follows:
//              L or l op(a)*x = alpha*b
//              R or r x*op(a) = alpha*b
//  uplo  = Specifies whether the matrix a is an upper or lower triangular
//          matrix as follows:
//              U or u Matrix a is an upper triangular matrix.
//              L or l Matrix a is a lower triangular matrix
//  m     = number of rows of the matrix op(a) and of the matrix c
//  n     = number of rows of the matrix op(b) and the number of columns of the matrix c
//  k     = number of columns of the matrix op(a) and the number of rows of the matrix op(b)
//  alpha = alpha scalar coefficient for matrix op(a)
//  a_gpu = pointer to gpu memory location of  matrix a
//  lda   = first dimension of a
//  b_gpu = pointer to gpu memory location of  matrix b
//  ldb   = first dimension of b
//  beta  = beta scalar coefficient for matrix c
//  c_gpu = pointer to gpu memory location of  matrix c
//  ldc   = first dimension of c
//
// OUTPUT
//  c_gpu     = c_gpu matrix
//
/*=========================================================================*/

extern "C" void gpu_zhemm_(char *side, char *uplo, int *N, int *M,
                           /* cuDoubleComplex */ void *alpha,
                           void **A_ptr, int *lda, void** B_ptr, int *ldb,
                           /* cuDoubleComplex */ void *beta, void** C_ptr, int *ldc)
{
#warning "Check routine signature"
	fprintf (stdout, "EMPTY CALL! %s:%d:%s\n", __FILE__, __LINE__, __func__);
} // gpu_zhemm_omp

/*=========================================================================*/
// NAME
//  gpu_xtrsm
//
// FUNCTION
// Solves a matrix equation (one matrix operand is triangular) on GPU.
// The xtrsm routines solve one of the following matrix equations
// op(a)*x = alpha*b
// or
// x*op(a) = alpha*b,
//
// INPUTS
// cplx= 1 if real 2 if complex
// side= Specifies whether op(a) appears on the left or right of x for
//      the operation to be performed as follows:
//      L or l op(a)*x = alpha*b
//      R or r x*op(a) = alpha*b
// uplo= Specifies whether the matrix a is an upper or lower triangular
//      matrix as follows:
//      U or u Matrix a is an upper triangular matrix.
//      L or l Matrix a is a lower triangular matrix
// transa= Specifies the form of op(a) to be used in the matrix
//      multiplication as follows:
//      N or n op(a) = a
//      T or t op(a) = a'
//      C or c op(a) = conjg(a')
// diag= Specifies whether or not a is unit triangular as follows:
//      U or u Matrix a is assumed to be unit triangular.
//      N or n Matrix a is not assumed to be unit triangular.
// m= Specifies the number of rows of b. The value of m must be at least zero
// n= Specifies the number of columns of b. The value of n must be at least zero
// alpha= Specifies the scalar alpha. When alpha is zero, then a is not referenced and b
//      need not be set before entry.
//  a_gpu = pointer to gpu memory location of  array a, DIMENSION (lda, k), where k is m when side = 'L' or 'l' and is n
//      when side = 'R' or 'r'.
// lda= Specifies the first dimension of a as declared in the calling
//     (sub)program. When side = 'L' or 'l', then lda must be at least max(1,
//      m), when side = 'R' or 'r', then lda must be at least max(1, n).
//  b_gpu = pointer to gpu memory location of  b Array, DIMENSION (ldb,n). Before entry, the leading m-by-n part of the array
//     b must contain the right-hand side matrix b.
// ldb= Specifies the first dimension of b as declared in the calling
//     (sub)program. The value of ldb must be at least max(1, m).
//
// OUTPUT
//  b_gpu
/*=========================================================================*/

extern "C" void gpu_xtrsm_(int* cplx, char *side, char *uplo, char *transA, char *diag,
                           int *N, int *M, /* cuDoubleComplex */ void *alpha,
                           void **A_ptr, int *ldA,
                           void** B_ptr, int *ldB)
{
#warning "Check routine signature"
	fprintf (stdout, "EMPTY CALL! %s:%d:%s\n", __FILE__, __LINE__, __func__);
} // gpu_xtrsm_

/*=========================================================================*/
// NAME
//  gpu_xaxpy
//
// FUNCTION
//  Compute blas-1 AXPY on GPU
//  y = alpha * x + y
//
// INPUTS
//  cplx  = 1 if real 2 if complex
//  alpha = scalar complex value (when doing computation with real value, only the real part is used)
//  x =     input vector
//  incrx = increment between two consecutive values of x
//  y =     in/out vector
//  incry = increment between two consecutive values of y
//
// OUTPUT
//  y
/*=========================================================================*/

extern "C" void gpu_xaxpy_(int* cplx, int *N,
                           /* cuDoubleComplex */ void *alpha,
                           void **X_ptr, int *incrx, void** Y_ptr, int *incry)
{
#warning "Check routine signature"
	fprintf (stdout, "EMPTY CALL! %s:%d:%s\n", __FILE__, __LINE__, __func__);
} // gpu_xaxpy_

/*=========================================================================*/
// NAME
//  gpu_xcopy
//
// FUNCTION
//  Compute blas-1 COPY on GPU
//  y = x (copy x into y)
//
// INPUTS / OUTPUTS
//  cplx  = 1 if real 2 if complex
//  N     = input vector size
//  x     = input vector
//  incrx = increment between two consecutive values of x
//  y     = output vector
//  incry = increment between two consecutive values of y
//
/*=========================================================================*/

extern "C" void gpu_xcopy_(int* cplx, int *N,
                           void **X_ptr, int *incrx,
                           void **Y_ptr, int *incry)
{
	if (*cplx == 1)
		#pragma omp dispatch
		cblas_dcopy (*N, (const double *)(*X_ptr), *incrx, (double *)(*Y_ptr), *incry);
	else
		#pragma omp dispatch
		cblas_zcopy (*N, *X_ptr, *incrx, *Y_ptr, *incry);
} // gpu_xcopy_

/*=========================================================================*/
// NAME
//  gpu_xscal
//
// FUNCTION
//  Compute blas-1 SCAL on GPU
//  x = alpha * x
//
// INPUTS
//  cplx  = 1 if real 2 if complex
//  N     = vector size
//  alpha = scalar complex value (when doing computation with real value, only the real part is used)
//  x     = input/output vector
//  incrx = increment between two consecutive values of x
//
// OUTPUT
//  x
/*=========================================================================*/

extern "C" void gpu_xscal_(int* cplx, int *N,
                           /* cuDoubleComplex */ void *alpha,
                           void **X_ptr, int *incrx)
{
#warning "Check routine signature"
	fprintf (stdout, "EMPTY CALL! %s:%d:%s\n", __FILE__, __LINE__, __func__);
} // gpu_xscal_

/*=========================================================================*/
// NAME
//  gpu_xpotrf
//
// FUNCTION
//  Compute a LAPACK SYGVD operation on GPU
//  computes the Cholesky factorization of a Hermitian positive-definite matrix
//
// See LAPACK doc in reference implementation:
// https://github.com/Reference-LAPACK/lapack/blob/master/SRC/dpotrf.f
//
// INPUTS
//  uplo  = character, 'u' or 'l'
//  n     = matrix size
//  A_ptr = pointer to gpu memory location of matrix A
//  lda   = leading dimension of matrix A
//  work_ptr =
//  lwork =
//  devInfo  =
//
/*=========================================================================*/

extern "C" void gpu_xpotrf_(const int* cplx,
                            const char* uplo,
                            const int* n,
                            void **A_ptr,
                            const int *lda,
                            void** work_ptr,
                            int* lwork,
                            int* info)
{
} // gpu_xpotrf_

extern "C" void gpu_xpotrf_buffersize_(const int* cplx,
                                       const char* uplo,
                                       const int* n,
                                       void **A_ptr,
                                       const int *lda,
                                       int* lwork)
{
	fprintf (stdout, "EMPTY CALL! %s:%d:%s\n", __FILE__, __LINE__, __func__);
} // gpu_xpotrf_bufferSize_

/*=========================================================================*/
// NAME
//  gpu_xsygvd
//
// FUNCTION
//  Compute a LAPACK SYGVD operation on GPU
//  compute eigen values/vectors of a real generalized
//  symmetric-definite eigenproblem
//
// See LAPACK doc in reference implementation:
// https://github.com/Reference-LAPACK/lapack/blob/master/SRC/dsygvd.f
//
// INPUTS
//  itype = integer, type of problem
//  jobz  = character, 'n'(eigenvalues only) or 'v' (eigenvalues + eigenvectors)
//  uplo  = character, 'u' or 'l'
//  A_nrows = matrix size
//  A_ptr = pointer to gpu memory location of matrix A
//  lda   = leading dimension of matrix A
//  B_ptr = pointer to gpu memory location of matrix B
//  ldb   = leading dimension of matrix B
//  W_ptr = pointer to gpu memory location of matrix W (output eigen values)
//  work_ptr =
//  lwork =
//  devInfo  =
//
/*=========================================================================*/

extern "C" void gpu_xsygvd_(const int* cplx,
                            const int* itype,
                            const char* jobz,
                            const char* uplo,
                            const int* A_nrows,
                            void **A_ptr,
                            const int *lda,
                            void** B_ptr,
                            const int* ldb,
                            void** W_ptr,
                            void** work_ptr,
                            int* lwork,
                            int* info)
{
	fprintf (stdout, "EMPTY CALL! %s:%d:%s\n", __FILE__, __LINE__, __func__);
} // gpu_xsygvd_


extern "C" void gpu_xsygvd_buffersize_(const int* cplx,
                                       const int* itype,
                                       const char* jobz,
                                       const char* uplo,
                                       const int* A_nrows,
                                       void **A_ptr,
                                       const int *lda,
                                       void** B_ptr,
                                       const int* ldb,
                                       void** W_ptr,
                                       int* lwork)
{
	fprintf (stdout, "EMPTY CALL! %s:%d:%s\n", __FILE__, __LINE__, __func__);
} // gpu_xsygvd_bufferSize_


/*=========================================================================*/
// NAME
//  gpu_xsyevd
//
// FUNCTION
//  Compute a LAPACK SYGVD operation on GPU
//  compute eigen values/vectors of a real generalized
//  symmetric-definite eigenproblem
//
// See LAPACK doc in reference implementation:
// https://github.com/Reference-LAPACK/lapack/blob/master/SRC/dsyevd.f
//
// INPUTS
//  itype = integer, type of problem
//  jobz  = character, 'n'(eigenvalues only) or 'v' (eigenvalues + eigenvectors)
//  uplo  = character, 'u' or 'l'
//  A_nrows = matrix size
//  A_ptr = pointer to gpu memory location of matrix A
//  lda   = leading dimension of matrix A
//  B_ptr = pointer to gpu memory location of matrix B
//  ldb   = leading dimension of matrix B
//  W_ptr = pointer to gpu memory location of matrix W (output eigen values)
//  work_ptr =
//  lwork =
//  devInfo  =
//
/*=========================================================================*/

extern "C" void gpu_xsyevd_(const int* cplx,
                            const char* jobz,
                            const char* uplo,
                            const int* A_nrows,
                            void **A_ptr,
                            const int *lda,
                            void** W_ptr,
                            void** work_ptr,
                            int* lwork,
                            int* info)
{
	fprintf (stdout, "EMPTY CALL! %s:%d:%s\n", __FILE__, __LINE__, __func__);
} // gpu_xsyevd_

extern "C" void gpu_xsyevd_buffersize_(const int* cplx,
                                       const char* jobz,
                                       const char* uplo,
                                       const int* A_nrows,
                                       void **A_ptr,
                                       const int *lda,
                                       void** W_ptr,
                                       int* lwork)
{
	fprintf (stdout, "EMPTY CALL! %s:%d:%s\n", __FILE__, __LINE__, __func__);
} // gpu_xsyevd_bufferSize_
