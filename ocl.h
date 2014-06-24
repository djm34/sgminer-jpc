#ifndef __OCL_H__
#define __OCL_H__

#include "config.h"

#include <stdbool.h>
#ifdef __APPLE_CC__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include "miner.h"

typedef struct {
	cl_context context;
	cl_kernel kernel;
	cl_kernel kernel_blake;
	cl_kernel kernel_bmw;
	cl_kernel kernel_groestl;
	cl_kernel kernel_skein;
	cl_kernel kernel_jh;
	cl_kernel kernel_keccak;
	cl_kernel kernel_luffa;
	cl_kernel kernel_cubehash;
	cl_kernel kernel_shavite;
	cl_kernel kernel_simd;
	cl_kernel kernel_echo;
	cl_kernel kernel_hamsi;
	cl_kernel kernel_fugue;
	cl_kernel kernel_echo_hamsi_fugue;
    cl_kernel kernel_groestl1;
	cl_kernel kernel_skein1;
	cl_kernel kernel_blake1;
	cl_kernel kernel_jh1;
	cl_kernel kernel_groestl2;
	cl_kernel kernel_skein2;
	cl_kernel kernel_blake2;
	cl_kernel kernel_jh2;
	cl_kernel kernel_groestl3;
	cl_kernel kernel_skein3;
	cl_kernel kernel_blake3;
	cl_kernel kernel_jh3;
	cl_kernel kernel_finale;
	cl_command_queue commandQueue;
	cl_program program;
	cl_mem outputBuffer;
	cl_mem hash_buffer;
	cl_mem CLbuffer0;
	cl_mem padbuffer8;
	cl_mem flagR1;
	cl_mem flagR2;
	cl_mem flagR3;
	size_t padbufsize;
	unsigned char cldata[128];
	bool hasBitAlign;
	bool hasOpenCL11plus;
	bool hasOpenCL12plus;
	bool goffset;
	cl_uint vwidth;
	size_t max_work_size;
	size_t wsize;
	size_t compute_shaders;
	enum cl_kernels chosen_kernel;
} _clState;

extern char *file_contents(const char *filename, int *length);
extern int clDevicesNum(void);
extern _clState *initCl(unsigned int gpu, char *name, size_t nameSize);
#endif /* __OCL_H__ */
