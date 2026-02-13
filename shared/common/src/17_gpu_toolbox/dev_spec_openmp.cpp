/* dev_spec_openmp.cpp */
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

void check_err(int line)
{
	fprintf (stdout, "EMPTY CALL! %s:%d:%s\n", __FILE__, __LINE__, __func__);
}

// Gives the number of GPU devices ---------
extern "C" void get_gpu_ndev_(int* ndevice)
{
	fprintf (stdout, "EMPTY CALL! %s:%d:%s\n", __FILE__, __LINE__, __func__);
}

// Gives the max memory available for a GPU device ---------
extern "C" void get_gpu_max_mem_(int* device, float* max_mem)
{
	fprintf (stdout, "EMPTY CALL! %s:%d:%s\n", __FILE__, __LINE__, __func__);
}

// Gives max available memory available for current GPU device ---------
extern "C" void gpu_get_max_mem_cpp(size_t* max_mem)
{
	fprintf (stdout, "EMPTY CALL! %s:%d:%s\n", __FILE__, __LINE__, __func__);
}

// Gives currently free memory available for current GPU device ---------
extern "C" void gpu_get_free_mem_cpp(size_t* free_mem)
{
	fprintf (stdout, "EMPTY CALL! %s:%d:%s\n", __FILE__, __LINE__, __func__);
}

// Set the device if it exists   -----------------
extern "C" void set_dev_(int* gpudevice)
{
	fprintf (stdout, "EMPTY CALL! %s:%d:%s\n", __FILE__, __LINE__, __func__);
}

// Unset the devices  -----------------
extern "C" void unset_dev_()
{
	fprintf (stdout, "EMPTY CALL! %s:%d:%s\n", __FILE__, __LINE__, __func__);
}

// Synchronize device (makes the CPU waits the GPU to finish all running kernels)
// this is required when using mamanged memory in order to reuse safely on CPU data
// that were processed / modified by the GPU
extern "C" void gpu_device_synchronize_cpp()
{
	fprintf (stdout, "EMPTY CALL! %s:%d:%s\n", __FILE__, __LINE__, __func__);
}

//
extern "C" void gpu_get_device_cpp(int *deviceId)
{
	fprintf (stdout, "EMPTY CALL! %s:%d:%s\n", __FILE__, __LINE__, __func__);
}

//
extern "C" void gpu_data_prefetch_async_cpp(const void* devPtr, size_t count, int deviceId)
{
	fprintf (stdout, "EMPTY CALL! %s:%d:%s\n", __FILE__, __LINE__, __func__);
}

//
#if 0
extern "C" void gpu_memory_advise_cpp(const void* devPtr, size_t count, cudaMemoryAdvise advice, int deviceId)
{
	fprintf (stdout, "EMPTY CALL! %s:%d:%s\n", __FILE__, __LINE__, __func__);
}
#endif

// Get context  -----------------------
extern "C" void check_context_(int *res,char *message)
{
	fprintf (stdout, "EMPTY CALL! %s:%d:%s\n", __FILE__, __LINE__, __func__);
}

// Get info from device  --------------
extern "C" void  get_dev_info_(int* device,
	char* name,
	int* lenname,
	int vers[2],
	float* globalmem,
	float* clockrate,
	int* gflops,
	int* constmem,
	int* sharemem,
	int* regist,
	int* nprocs,
	int* ncores
)
{
	fprintf (stdout, "EMPTY CALL! %s:%d:%s\n", __FILE__, __LINE__, __func__);
}

// Get number of devices  --------------
extern "C" void c_get_ndevice_(int* ndev)
{
	*ndev = omp_get_num_devices();
}

/***************************************************************/
/*******                                                ********/
/*******      GPU MEMORY MANAGEMENT ROUTINES            ********/
/*******                                                ********/
/***************************************************************/

/*============================================================================*/
/* Print memory information (total amount and free available)                 */
/*============================================================================*/

extern "C" void check_gpu_mem_(const char* str)
{
	fprintf (stdout, "EMPTY CALL! %s:%d:%s\n", __FILE__, __LINE__, __func__);
}

/*============================================================================*/
/* Allocate size byte in gpu memory and returns in gpu_ptr this location      */
/* INPUTS size= size in byte to allocate                                      */
/* OUTPUT gpu_ptr= C_PTR on gpu memory location that has been allocated       */
/*============================================================================*/

extern "C" void alloc_on_gpu_(void **gpu_ptr, const size_t* size)
{
#warning "Need to define which device to use!"
	*gpu_ptr = omp_target_alloc (*size, omp_get_default_device());
}

/*============================================================================*/
/* Free memory location pointed by gpu_ptr                                    */
/* OUTPUT gpu_ptr= C_PTR on gpu memory location that has been allocated       */
/* WARNING! : this routine is a dummy one when HAVE_GPU_CUDA is not enabled   */
/*            the correct one is in xx_gpu_toolbox/dev_spec.cu                */
/*============================================================================*/

extern "C" void dealloc_on_gpu_(void **gpu_ptr)
{
#warning "Need to define which device to use!"
	omp_target_free (*gpu_ptr, omp_get_default_device());
}

/*============================================================================*/
/* Copy size byte from cpu pointer to gpu pointer.                            */
/* INPUTS                                                                     */
/*  size = size in byte to copy                                               */
/*  cpu_ptr = host memory location (LOC)                                      */
/* OUTPUT                                                                     */
/*  gpu_ptr = C_PTR : gpu memory location                                     */
/* WARNING! : this routine is a dummy one when HAVE_GPU_CUDA is not enabled   */
/*            the correct one is in xx_gpu_toolbox/dev_spec.cu                */
/*============================================================================*/

extern "C" void copy_on_gpu_(void *cpu_ptr, void **gpu_ptr, const size_t* size)
{
#warning "Need to define which device to use!"
	assert (0 == omp_target_memcpy (*gpu_ptr, cpu_ptr, *size, 0, 0, omp_get_default_device(), omp_get_initial_device()));
}

/*============================================================================*/
/* Copy size byte from gpu pointer to cpu pointer.                            */
/* INPUTS                                                                     */
/*  size = size in byte to copy                                               */
/*  gpu_ptr = C_PTR : gpu memory location                                     */
/* OUTPUT                                                                     */
/*  cpu_ptr = host memory location (LOC of an allocated array)                */
/*============================================================================*/

extern "C" void copy_from_gpu_(void *cpu_ptr, void **gpu_ptr, const size_t* size)
{
#warning "Need to define which device to use!"
	assert (0 == omp_target_memcpy (cpu_ptr, *gpu_ptr, *size, 0, 0, omp_get_initial_device(), omp_get_default_device()));
}

/*============================================================================*/
/* Copy size byte from gpu to gpu memory.                                     */
/* INPUTS                                                                     */
/*  size = size in byte to copy                                               */
/*  src_gpu_ptr                                                               */
/* OUTPUT                                                                     */
/*  dest_gpu_ptr = C_PTR on gpu memory location                               */
/* WARNING! : this routine is a dummy one when HAVE_GPU_CUDA is not enabled   */
/*            the correct one is in xx_gpu_toolbox/dev_spec.cu                */
/*============================================================================*/

extern "C" void copy_gpu_to_gpu_cpp_(void **dest_gpu_ptr, void **src_gpu_ptr, const size_t* size)
{
#warning "Need to define which devices to use!"
	assert (0 == omp_target_memcpy (*dest_gpu_ptr, *src_gpu_ptr, *size, 0, 0, omp_get_default_device(), omp_get_default_device()));
}

/*============================================================================*/
/* Reset array (just wrapping cudaMemset)                                     */
/*                                                                            */
/* INPUTS                                                                     */
/*  gpu_ptr = C_PTR on gpu memory location                                    */
/*  value = integer used to initialize each bytes (should be in range [0,255])*/
/*  size = size in bytes of the region to be set                              */
/*                                                                            */
/* OUTPUT                                                                     */
/*  None                                                                      */
/*============================================================================*/

extern "C" void gpu_memset_cpp_(void **gpu_ptr, const int32_t* value, const size_t* size_in_bytes)
{
#warning "Need to define which device to use"
#warning "This needs OpenMP 6.0?"
	assert (0 == omp_target_memset (*gpu_ptr, *value, *size_in_bytes, omp_get_default_device()));
}

/*============================================================================*/
/* Kind of equivalent of fortran "allocated". Check if a gpu pointer          */
/* actually points to device allocated memory.                                */
/*                                                                            */
/* This is void function because I can't manage to bind it via iso_c_binding  */
/* as a fortran function; binding as a subroutine is ok though (?!)           */
/*                                                                            */
/* INPUTS                                                                     */
/*  gpu_ptr = C_PTR on gpu memory location                                    */
/*                                                                            */
/* OUTPUT                                                                     */
/*  boolean/logical (false = not allocated, true = allocated)                 */
/*============================================================================*/

extern "C" void gpu_allocated_impl_(void **gpu_ptr, bool* is_allocated)
{
#warning "Need to define which device to use"
	*is_allocated = omp_target_is_present (*gpu_ptr, omp_get_default_device()) != 0;
} // gpu_allocated_impl_

/*============================================================================*/
/* Utility routine to print memory location of a cuda managed pointer.        */
/*                                                                            */
/* We check that the pointer has actually been allocated with                 */
/* cudaMallocManaged and then prints device and host addresses.               */
/*                                                                            */
/* INPUTS                                                                     */
/*  gpu_ptr = C_PTR on gpu memory location                                    */
/*                                                                            */
/* OUTPUT                                                                     */
/*  None.                                                                     */
/*============================================================================*/

extern "C" void gpu_managed_ptr_status_(void **gpu_ptr, const char* str)
{
	fprintf (stdout, "EMPTY CALL! %s:%d:%s\n", __FILE__, __LINE__, __func__);
} // gpu_managed_ptr_status_
