/*
 * multi-kernel implementation of jackpotcoin algorithm.
 *
 * Copyright (c) 2014 djm34
 */

#ifndef ADVSHA3_CL
#define ADVSHA3_CL

#pragma OPENCL EXTENSION cl_amd_printf : enable

#if __ENDIAN_LITTLE__
#define SPH_LITTLE_ENDIAN 1
#else
#define SPH_BIG_ENDIAN 1
#endif

#define SPH_UPTR sph_u64

typedef unsigned int sph_u32;
typedef int sph_s32;
#ifndef __OPENCL_VERSION__
typedef unsigned long long sph_u64;
typedef long long sph_s64;
#else
typedef unsigned long sph_u64;
typedef long sph_s64;
#endif

#define SPH_64 1
#define SPH_64_TRUE 1



#define SPH_C32(x)    ((sph_u32)(x ## U))
#define SPH_T32(x) (as_uint(x))
#define SPH_ROTL32(x, n) rotate(as_uint(x), as_uint(n))
#define SPH_ROTR32(x, n)   SPH_ROTL32(x, (32 - (n)))

#define SPH_C64(x)    ((sph_u64)(x ## UL))
#define SPH_T64(x) (as_ulong(x))
#define SPH_ROTL64(x, n) rotate(as_ulong(x), (n) & 0xFFFFFFFFFFFFFFFFUL)
#define SPH_ROTR64(x, n)   SPH_ROTL64(x, (64 - (n)))

#define SPH_KECCAK_64 1
#define SPH_JH_64 1
#define SPH_KECCAK_NOCOPY 0
#define SPH_COMPACT_BLAKE_64 0
#define SPH_SMALL_FOOTPRINT_GROESTL 0
#define SPH_GROESTL_BIG_ENDIAN 0



#include "blake.cl"
#include "groestl.cl"
#include "jh.cl"
#include "skein.cl"
#include "__keccak.cl"

#define SWAP4(x) as_uint(as_uchar4(x).wzyx)
#define SWAP8(x) as_ulong(as_uchar8(x).s76543210)

#if SPH_BIG_ENDIAN
    #define DEC64E(x) (x)
    #define DEC64BE(x) (*(const __global sph_u64 *) (x));
#else
    #define DEC64E(x) SWAP8(x)
    #define DEC64BE(x) SWAP8(*(const __global sph_u64 *) (x));
#endif


typedef union {
        uint  U4[16];
        ulong U8[8];
     } hash_t;
 
__attribute__((reqd_work_group_size(WORKSIZE, 1, 1)))
__kernel void keccak(__global uint * input, __global hash_t* hashes) {

	
     uint  gid = get_global_id(0);
	 __global hash_t *hash = &(hashes[gid-get_global_offset(0)]);
     union {
        uint  U4[22];
        ulong U8[11];
     } HASH;

	 #pragma unroll 22
     for (uint i = 0; i < 22; i++) {
         HASH.U4[i] = input[i];
     }
     HASH.U4[19] = SWAP4(gid);

     ulong c0x, c1x, c2x, c3x, c4x;
     ulong a00 =  HASH.U8[0];
     ulong a10 = ~HASH.U8[1];
     ulong a20 = ~HASH.U8[2];
     ulong a30 =  HASH.U8[3];
     ulong a40 =  HASH.U8[4];
     ulong a01 =  HASH.U8[5];
     ulong a11 =  HASH.U8[6];
     ulong a21 =  HASH.U8[7];
     ulong a31 = ~HASH.U8[8];
     ulong a41 =  0;
     ulong a02 =  0;
     ulong a12 =  0;
     ulong a22 =  0xFFFFFFFFFFFFFFFFUL;
     ulong a32 =  0;
     ulong a42 =  0;
     ulong a03 =  0;
     ulong a13 =  0;
     ulong a23 =  0xFFFFFFFFFFFFFFFFUL;
     ulong a33 =  0;
     ulong a43 =  0;
     ulong a04 =  0xFFFFFFFFFFFFFFFFUL;
     ulong a14 =  0;
     ulong a24 =  0;
     ulong a34 =  0;
     ulong a44 =  0;
     KECCAK_F_1600;
	 a00 ^=  HASH.U8[9];
     a10 ^=  0x01;
     a31 ^=  0x8000000000000000UL;
     KECCAK_F_1600;
     a10     = ~a10;
     a20     = ~a20;
     HASH.U8[0x00] =  a00;
     HASH.U8[0x01] =  a10;
     HASH.U8[0x02] =  a20;
     HASH.U8[0x03] =  a30;
     HASH.U8[0x04] =  a40;
     HASH.U8[0x05] =  a01;
     HASH.U8[0x06] =  a11;
     HASH.U8[0x07] =  a21;
		
		#pragma unroll 8
		for (int i=0;i<8;i++) {hash->U8[i]=HASH.U8[i];}
		

}

__attribute__((reqd_work_group_size(WORKSIZE, 1, 1)))
__kernel void groestl1(__global hash_t* hashes, __global ushort* flag1,__global ushort* flag2) 
{

uint gid = get_global_id(0);
    __global hash_t *hash = &(hashes[gid-get_global_offset(0)]);
__global ushort *flagA = &(flag1[gid-get_global_offset(0)]);
__global ushort *flagB = &(flag2[gid-get_global_offset(0)]);
flagA[0]= 0x01;
flagB[0]= 0x01;

__local sph_u64 T0_C[256], T1_C[256], T2_C[256], T3_C[256], T4_C[256], T5_C[256], T6_C[256], T7_C[256];

    int init = get_local_id(0);
    int step = get_local_size(0);

    for (int i = init; i < 256; i += step)
    {
        T0_C[i] = T0[i];
        T1_C[i] = T1[i];
        T2_C[i] = T2[i];
        T3_C[i] = T3[i];
        T4_C[i] = T4[i];
        T5_C[i] = T5[i];
        T6_C[i] = T6[i];
        T7_C[i] = T7[i];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

#define T0 T0_C
#define T1 T1_C
#define T2 T2_C
#define T3 T3_C
#define T4 T4_C
#define T5 T5_C
#define T6 T6_C
#define T7 T7_C


            if (hash->U4[0] & 0x1) {
flagA[0]= 0xff;
//flagB[0]= 0xa1;

    sph_u64 H[16];
    for (unsigned int u = 0; u < 15; u ++)
        H[u] = 0;
#if USE_LE
    H[15] = ((sph_u64)(512 & 0xFF) << 56) | ((sph_u64)(512 & 0xFF00) << 40);
#else
    H[15] = (sph_u64)512;
#endif

    sph_u64 g[16], m[16];
    m[0] = hash->U8[0];
    m[1] = hash->U8[1];
    m[2] = hash->U8[2];
    m[3] = hash->U8[3];
    m[4] = hash->U8[4];
    m[5] = hash->U8[5];
    m[6] = hash->U8[6];
    m[7] = hash->U8[7];
    for (unsigned int u = 0; u < 16; u ++)
        g[u] = m[u] ^ H[u];
    m[8] = 0x80; g[8] = m[8] ^ H[8];
    m[9] = 0; g[9] = m[9] ^ H[9];
    m[10] = 0; g[10] = m[10] ^ H[10];
    m[11] = 0; g[11] = m[11] ^ H[11];
    m[12] = 0; g[12] = m[12] ^ H[12];
    m[13] = 0; g[13] = m[13] ^ H[13];
    m[14] = 0; g[14] = m[14] ^ H[14];
    m[15] = 0x100000000000000; g[15] = m[15] ^ H[15];
    PERM_BIG_P(g);
    PERM_BIG_Q(m);
    for (unsigned int u = 0; u < 16; u ++)
        H[u] ^= g[u] ^ m[u];
    sph_u64 xH[16];
    for (unsigned int u = 0; u < 16; u ++)
        xH[u] = H[u];
    PERM_BIG_P(xH);
    for (unsigned int u = 0; u < 16; u ++)
        H[u] ^= xH[u];
    for (unsigned int u = 0; u < 8; u ++)
        hash->U8[u] = H[u + 8];

    barrier(CLK_GLOBAL_MEM_FENCE);             

	
            }

#undef T0 
#undef T1 
#undef T2 
#undef T3 
#undef T4 
#undef T5 
#undef T6 
#undef T7 

}

__attribute__((reqd_work_group_size(WORKSIZE, 1, 1)))
__kernel void skein1(__global hash_t* hashes, __global ushort* flag1,__global ushort* flag2) 
{

uint gid = get_global_id(0);
    __global hash_t *hash = &(hashes[gid-get_global_offset(0)]);
	__global ushort *flagA = &(flag1[gid-get_global_offset(0)]);
	__global ushort *flagB = &(flag2[gid-get_global_offset(0)]);
	if (flagA[0]==0x01 && flagB[0]==0x01) {

    sph_u64 h0 = SPH_C64(0x4903ADFF749C51CE), h1 = SPH_C64(0x0D95DE399746DF03), h2 = SPH_C64(0x8FD1934127C79BCE), h3 = SPH_C64(0x9A255629FF352CB1), h4 = SPH_C64(0x5DB62599DF6CA7B0), h5 = SPH_C64(0xEABE394CA9D5C3F4), h6 = SPH_C64(0x991112C71A75B523), h7 = SPH_C64(0xAE18A40B660FCC33);
    sph_u64 m0, m1, m2, m3, m4, m5, m6, m7;
    sph_u64 bcount = 0;
/*
    m0 = SWAP8(DEC64E(hash->U8[0]));
    m1 = SWAP8(DEC64E(hash->U8[1]));
    m2 = SWAP8(DEC64E(hash->U8[2]));
    m3 = SWAP8(DEC64E(hash->U8[3]));
    m4 = SWAP8(DEC64E(hash->U8[4]));
    m5 = SWAP8(DEC64E(hash->U8[5]));
    m6 = SWAP8(DEC64E(hash->U8[6]));
    m7 = SWAP8(DEC64E(hash->U8[7]));
    UBI_BIG(480, 64);
    bcount = 0;
    m0 = m1 = m2 = m3 = m4 = m5 = m6 = m7 = 0;
    UBI_BIG(510, 8);
    hash->U8[0] = SWAP8(DEC64E(h0));
    hash->U8[1] = SWAP8(DEC64E(h1));
    hash->U8[2] = SWAP8(DEC64E(h2));
    hash->U8[3] = SWAP8(DEC64E(h3));
    hash->U8[4] = SWAP8(DEC64E(h4));
    hash->U8[5] = SWAP8(DEC64E(h5));
    hash->U8[6] = SWAP8(DEC64E(h6));
    hash->U8[7] = SWAP8(DEC64E(h7));
*/ 
    m0 = hash->U8[0];
    m1 = hash->U8[1];
    m2 = hash->U8[2];
    m3 = hash->U8[3];
    m4 = hash->U8[4];
    m5 = hash->U8[5];
    m6 = hash->U8[6];
    m7 = hash->U8[7];
    UBI_BIG(480, 64);
    bcount = 0;
    m0 = m1 = m2 = m3 = m4 = m5 = m6 = m7 = 0;
    UBI_BIG(510, 8);
    hash->U8[0] = h0;
    hash->U8[1] = h1;
    hash->U8[2] = h2;
    hash->U8[3] = h3;
    hash->U8[4] = h4;
    hash->U8[5] = h5;
    hash->U8[6] = h6;
    hash->U8[7] = h7;


    barrier(CLK_GLOBAL_MEM_FENCE);

	}
	
		
}

__attribute__((reqd_work_group_size(WORKSIZE, 1, 1)))
__kernel void blake1(__global hash_t* hashes, __global ushort* flag1, __global ushort* flag2) 
{
uint gid = get_global_id(0);
    __global hash_t *hash = &(hashes[gid-get_global_offset(0)]);

__global ushort *flagA = &(flag1[gid-get_global_offset(0)]);
__global ushort *flagB = &(flag2[gid-get_global_offset(0)]);
flagA[0]= 0x01;
            if ( (hash->U4[0] & 0x1) && flagB[0]==0x01 ) {
flagA[0]= 0xff;



     sph_u64 H0 = SPH_C64(0x6A09E667F3BCC908), H1 = SPH_C64(0xBB67AE8584CAA73B);
    sph_u64 H2 = SPH_C64(0x3C6EF372FE94F82B), H3 = SPH_C64(0xA54FF53A5F1D36F1);
    sph_u64 H4 = SPH_C64(0x510E527FADE682D1), H5 = SPH_C64(0x9B05688C2B3E6C1F);
    sph_u64 H6 = SPH_C64(0x1F83D9ABFB41BD6B), H7 = SPH_C64(0x5BE0CD19137E2179);
    sph_u64 S0 = 0, S1 = 0, S2 = 0, S3 = 0;
    sph_u64 T0 = SPH_C64(0xFFFFFFFFFFFFFC00) + (64 << 3), T1 = 0xFFFFFFFFFFFFFFFF;;

    if ((T0 = SPH_T64(T0 + 1024)) < 1024)
    {
        T1 = SPH_T64(T1 + 1);
    }
    sph_u64 M0, M1, M2, M3, M4, M5, M6, M7;
    sph_u64 M8, M9, MA, MB, MC, MD, ME, MF;
    sph_u64 V0, V1, V2, V3, V4, V5, V6, V7;
    sph_u64 V8, V9, VA, VB, VC, VD, VE, VF;
    M0 = DEC64E(hash->U8[0]);
    M1 = DEC64E(hash->U8[1]);
    M2 = DEC64E(hash->U8[2]);
    M3 = DEC64E(hash->U8[3]);
    M4 = DEC64E(hash->U8[4]);
    M5 = DEC64E(hash->U8[5]);
    M6 = DEC64E(hash->U8[6]);
    M7 = DEC64E(hash->U8[7]);
    M8 = 0x8000000000000000;
    M9 = 0;
    MA = 0;
    MB = 0;
    MC = 0;
    MD = 1;
    ME = 0;
    MF = 0x200;

    COMPRESS64;

    hash->U8[0] = DEC64E(H0);
    hash->U8[1] = DEC64E(H1);
    hash->U8[2] = DEC64E(H2);
    hash->U8[3] = DEC64E(H3);
    hash->U8[4] = DEC64E(H4);
    hash->U8[5] = DEC64E(H5);
    hash->U8[6] = DEC64E(H6);
    hash->U8[7] = DEC64E(H7);   
    barrier(CLK_GLOBAL_MEM_FENCE);
//blake512
			}
           
}

__attribute__((reqd_work_group_size(WORKSIZE, 1, 1)))
__kernel void jh1(__global hash_t* hashes, __global ushort* flag1, __global ushort* flag2) 
{

uint gid = get_global_id(0);
    __global hash_t *hash = &(hashes[gid-get_global_offset(0)]);
    __global ushort *flagA = &(flag1[gid-get_global_offset(0)]);
	__global ushort *flagB = &(flag2[gid-get_global_offset(0)]);
	if (flagA[0]==0x01 && flagB[0]==0x01) {
//jh512				
	sph_u64 h0h = C64e(0x6fd14b963e00aa17), h0l = C64e(0x636a2e057a15d543), h1h = C64e(0x8a225e8d0c97ef0b), h1l = C64e(0xe9341259f2b3c361), h2h = C64e(0x891da0c1536f801e), h2l = C64e(0x2aa9056bea2b6d80), h3h = C64e(0x588eccdb2075baa6), h3l = C64e(0xa90f3a76baf83bf7);
    sph_u64 h4h = C64e(0x0169e60541e34a69), h4l = C64e(0x46b58a8e2e6fe65a), h5h = C64e(0x1047a7d0c1843c24), h5l = C64e(0x3b6e71b12d5ac199), h6h = C64e(0xcf57f6ec9db1f856), h6l = C64e(0xa706887c5716b156), h7h = C64e(0xe3c2fcdfe68517fb), h7l = C64e(0x545a4678cc8cdd4b);
    sph_u64 tmp;

    for(int i = 0; i < 2; i++)
    {
        if (i == 0) {
            h0h ^= hash->U8[0];
            h0l ^= hash->U8[1];
            h1h ^= hash->U8[2];
            h1l ^= hash->U8[3];
            h2h ^= hash->U8[4];
            h2l ^= hash->U8[5];
            h3h ^= hash->U8[6];
            h3l ^= hash->U8[7];
        } else if(i == 1) {
            h4h ^= hash->U8[0];
            h4l ^= hash->U8[1];
            h5h ^= hash->U8[2];
            h5l ^= hash->U8[3];
            h6h ^= hash->U8[4];
            h6l ^= hash->U8[5];
            h7h ^= hash->U8[6];
            h7l ^= hash->U8[7];
        
            h0h ^= 0x80;
            h3l ^= 0x2000000000000;
        }
        E8;
    }
    h4h ^= 0x80;
    h7l ^= 0x2000000000000;

    hash->U8[0] = h4h;
    hash->U8[1] = h4l;
    hash->U8[2] = h5h;
    hash->U8[3] = h5l;
    hash->U8[4] = h6h;
    hash->U8[5] = h6l;
    hash->U8[6] = h7h;
    hash->U8[7] = h7l;
    barrier(CLK_LOCAL_MEM_FENCE);			
			}

}


__attribute__((reqd_work_group_size(WORKSIZE, 1, 1)))
__kernel void groestl2(__global hash_t* hashes, __global ushort* flag1,__global ushort* flag2) 
{
	

uint gid = get_global_id(0);
    __global hash_t *hash = &(hashes[gid-get_global_offset(0)]);

__global ushort *flagA = &(flag1[gid-get_global_offset(0)]);
__global ushort *flagB = &(flag2[gid-get_global_offset(0)]);
flagA[0]= 0x01;

 __local sph_u64 T0_C[256], T1_C[256], T2_C[256], T3_C[256], T4_C[256], T5_C[256], T6_C[256], T7_C[256];

    int init = get_local_id(0);
    int step = get_local_size(0);

    for (int i = init; i < 256; i += step)
    {
        T0_C[i] = T0[i];
        T1_C[i] = T1[i];
        T2_C[i] = T2[i];
        T3_C[i] = T3[i];
        T4_C[i] = T4[i];
        T5_C[i] = T5[i];
        T6_C[i] = T6[i];
        T7_C[i] = T7[i];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
#define T0 T0_C
#define T1 T1_C
#define T2 T2_C
#define T3 T3_C
#define T4 T4_C
#define T5 T5_C
#define T6 T6_C
#define T7 T7_C

            if ( (hash->U4[0] & 0x1) && flagB[0]==0x01 ) {
flagA[0]= 0xff;
//flagB[0]= 0xa1;            

    sph_u64 H[16];
    for (unsigned int u = 0; u < 15; u ++)
        H[u] = 0;
#if USE_LE
    H[15] = ((sph_u64)(512 & 0xFF) << 56) | ((sph_u64)(512 & 0xFF00) << 40);
#else
    H[15] = (sph_u64)512;
#endif

    sph_u64 g[16], m[16];
    m[0] = hash->U8[0];
    m[1] = hash->U8[1];
    m[2] = hash->U8[2];
    m[3] = hash->U8[3];
    m[4] = hash->U8[4];
    m[5] = hash->U8[5];
    m[6] = hash->U8[6];
    m[7] = hash->U8[7];
    for (unsigned int u = 0; u < 16; u ++)
        g[u] = m[u] ^ H[u];
    m[8] = 0x80; g[8] = m[8] ^ H[8];
    m[9] = 0; g[9] = m[9] ^ H[9];
    m[10] = 0; g[10] = m[10] ^ H[10];
    m[11] = 0; g[11] = m[11] ^ H[11];
    m[12] = 0; g[12] = m[12] ^ H[12];
    m[13] = 0; g[13] = m[13] ^ H[13];
    m[14] = 0; g[14] = m[14] ^ H[14];
    m[15] = 0x100000000000000; g[15] = m[15] ^ H[15];
    PERM_BIG_P(g);
    PERM_BIG_Q(m);
    for (unsigned int u = 0; u < 16; u ++)
        H[u] ^= g[u] ^ m[u];
    sph_u64 xH[16];
    for (unsigned int u = 0; u < 16; u ++)
        xH[u] = H[u];
    PERM_BIG_P(xH);
    for (unsigned int u = 0; u < 16; u ++)
        H[u] ^= xH[u];
    for (unsigned int u = 0; u < 8; u ++)
        hash->U8[u] = H[u + 8];

    barrier(CLK_GLOBAL_MEM_FENCE);    
	
            }

#undef T0 
#undef T1 
#undef T2 
#undef T3 
#undef T4 
#undef T5 
#undef T6 
#undef T7 

}

__attribute__((reqd_work_group_size(WORKSIZE, 1, 1)))
__kernel void skein2(__global hash_t* hashes, __global ushort* flag1,__global ushort* flag2) 
{
uint gid = get_global_id(0);
    __global hash_t *hash = &(hashes[gid-get_global_offset(0)]);
    __global ushort *flagA = &(flag1[gid-get_global_offset(0)]);
	__global ushort *flagB = &(flag2[gid-get_global_offset(0)]);
	if (flagA[0]==0x01 && flagB[0]==0x01) {

// skein

    sph_u64 h0 = SPH_C64(0x4903ADFF749C51CE), h1 = SPH_C64(0x0D95DE399746DF03), h2 = SPH_C64(0x8FD1934127C79BCE), h3 = SPH_C64(0x9A255629FF352CB1), h4 = SPH_C64(0x5DB62599DF6CA7B0), h5 = SPH_C64(0xEABE394CA9D5C3F4), h6 = SPH_C64(0x991112C71A75B523), h7 = SPH_C64(0xAE18A40B660FCC33);
    sph_u64 m0, m1, m2, m3, m4, m5, m6, m7;
    sph_u64 bcount = 0;

    m0 = hash->U8[0];
    m1 = hash->U8[1];
    m2 = hash->U8[2];
    m3 = hash->U8[3];
    m4 = hash->U8[4];
    m5 = hash->U8[5];
    m6 = hash->U8[6];
    m7 = hash->U8[7];
    UBI_BIG(480, 64);
    bcount = 0;
    m0 = m1 = m2 = m3 = m4 = m5 = m6 = m7 = 0;
    UBI_BIG(510, 8);
    hash->U8[0] = h0;
    hash->U8[1] = h1;
    hash->U8[2] = h2;
    hash->U8[3] = h3;
    hash->U8[4] = h4;
    hash->U8[5] = h5;
    hash->U8[6] = h6;
    hash->U8[7] = h7;
 
    barrier(CLK_GLOBAL_MEM_FENCE);

          }
		
}

__attribute__((reqd_work_group_size(WORKSIZE, 1, 1)))
__kernel void blake2(__global hash_t* hashes, __global ushort* flag1, __global ushort* flag2) 
{

uint gid = get_global_id(0);
    __global hash_t *hash = &(hashes[gid-get_global_offset(0)]);

__global ushort *flagA = &(flag1[gid-get_global_offset(0)]);
__global ushort *flagB = &(flag2[gid-get_global_offset(0)]);
flagA[0]= 0x01;
            if ( (hash->U4[0] & 0x1) && flagB[0]==0x01 ) {
flagA[0]= 0xff;
            

            
// blake512
     sph_u64 H0 = SPH_C64(0x6A09E667F3BCC908), H1 = SPH_C64(0xBB67AE8584CAA73B);
    sph_u64 H2 = SPH_C64(0x3C6EF372FE94F82B), H3 = SPH_C64(0xA54FF53A5F1D36F1);
    sph_u64 H4 = SPH_C64(0x510E527FADE682D1), H5 = SPH_C64(0x9B05688C2B3E6C1F);
    sph_u64 H6 = SPH_C64(0x1F83D9ABFB41BD6B), H7 = SPH_C64(0x5BE0CD19137E2179);
    sph_u64 S0 = 0, S1 = 0, S2 = 0, S3 = 0;
    sph_u64 T0 = SPH_C64(0xFFFFFFFFFFFFFC00) + (64 << 3), T1 = 0xFFFFFFFFFFFFFFFF;;

    if ((T0 = SPH_T64(T0 + 1024)) < 1024)
    {
        T1 = SPH_T64(T1 + 1);
    }
    sph_u64 M0, M1, M2, M3, M4, M5, M6, M7;
    sph_u64 M8, M9, MA, MB, MC, MD, ME, MF;
    sph_u64 V0, V1, V2, V3, V4, V5, V6, V7;
    sph_u64 V8, V9, VA, VB, VC, VD, VE, VF;
    M0 = DEC64E(hash->U8[0]);
    M1 = DEC64E(hash->U8[1]);
    M2 = DEC64E(hash->U8[2]);
    M3 = DEC64E(hash->U8[3]);
    M4 = DEC64E(hash->U8[4]);
    M5 = DEC64E(hash->U8[5]);
    M6 = DEC64E(hash->U8[6]);
    M7 = DEC64E(hash->U8[7]);
    M8 = 0x8000000000000000;
    M9 = 0;
    MA = 0;
    MB = 0;
    MC = 0;
    MD = 1;
    ME = 0;
    MF = 0x200;

    COMPRESS64;

    hash->U8[0] = DEC64E(H0);
    hash->U8[1] = DEC64E(H1);
    hash->U8[2] = DEC64E(H2);
    hash->U8[3] = DEC64E(H3);
    hash->U8[4] = DEC64E(H4);
    hash->U8[5] = DEC64E(H5);
    hash->U8[6] = DEC64E(H6);
    hash->U8[7] = DEC64E(H7);   
    barrier(CLK_GLOBAL_MEM_FENCE);
//blake512
            } 
}

__attribute__((reqd_work_group_size(WORKSIZE, 1, 1)))
__kernel void jh2(__global hash_t* hashes, __global ushort* flag1, __global ushort* flag2) 
{

uint gid = get_global_id(0);    
    __global hash_t *hash = &(hashes[gid-get_global_offset(0)]);
    __global ushort *flagA = &(flag1[gid-get_global_offset(0)]);
	__global ushort *flagB = &(flag2[gid-get_global_offset(0)]);
	if (flagA[0]==0x01 && flagB[0]==0x01) {


//jh512				
	sph_u64 h0h = C64e(0x6fd14b963e00aa17), h0l = C64e(0x636a2e057a15d543), h1h = C64e(0x8a225e8d0c97ef0b), h1l = C64e(0xe9341259f2b3c361), h2h = C64e(0x891da0c1536f801e), h2l = C64e(0x2aa9056bea2b6d80), h3h = C64e(0x588eccdb2075baa6), h3l = C64e(0xa90f3a76baf83bf7);
    sph_u64 h4h = C64e(0x0169e60541e34a69), h4l = C64e(0x46b58a8e2e6fe65a), h5h = C64e(0x1047a7d0c1843c24), h5l = C64e(0x3b6e71b12d5ac199), h6h = C64e(0xcf57f6ec9db1f856), h6l = C64e(0xa706887c5716b156), h7h = C64e(0xe3c2fcdfe68517fb), h7l = C64e(0x545a4678cc8cdd4b);
    sph_u64 tmp;

    for(int i = 0; i < 2; i++)
    {
        if (i == 0) {
            h0h ^= hash->U8[0];
            h0l ^= hash->U8[1];
            h1h ^= hash->U8[2];
            h1l ^= hash->U8[3];
            h2h ^= hash->U8[4];
            h2l ^= hash->U8[5];
            h3h ^= hash->U8[6];
            h3l ^= hash->U8[7];
        } else if(i == 1) {
            h4h ^= hash->U8[0];
            h4l ^= hash->U8[1];
            h5h ^= hash->U8[2];
            h5l ^= hash->U8[3];
            h6h ^= hash->U8[4];
            h6l ^= hash->U8[5];
            h7h ^= hash->U8[6];
            h7l ^= hash->U8[7];
        
            h0h ^= 0x80;
            h3l ^= 0x2000000000000;
        }
        E8;
    }
    h4h ^= 0x80;
    h7l ^= 0x2000000000000;

    hash->U8[0] = h4h;
    hash->U8[1] = h4l;
    hash->U8[2] = h5h;
    hash->U8[3] = h5l;
    hash->U8[4] = h6h;
    hash->U8[5] = h6l;
    hash->U8[6] = h7h;
    hash->U8[7] = h7l;
    barrier(CLK_LOCAL_MEM_FENCE);			
			}

}


__attribute__((reqd_work_group_size(WORKSIZE, 1, 1)))
__kernel void groestl3(__global hash_t* hashes, __global ushort* flag1,__global ushort* flag2) 
{
	

uint gid = get_global_id(0);
    __global hash_t *hash = &(hashes[gid-get_global_offset(0)]);

	 __local sph_u64 T0_C[256], T1_C[256], T2_C[256], T3_C[256], T4_C[256], T5_C[256], T6_C[256], T7_C[256];

    int init = get_local_id(0);
    int step = get_local_size(0);

    for (int i = init; i < 256; i += step)
    {
        T0_C[i] = T0[i];
        T1_C[i] = T1[i];
        T2_C[i] = T2[i];
        T3_C[i] = T3[i];
        T4_C[i] = T4[i];
        T5_C[i] = T5[i];
        T6_C[i] = T6[i];
        T7_C[i] = T7[i];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

#define T0 T0_C
#define T1 T1_C
#define T2 T2_C
#define T3 T3_C
#define T4 T4_C
#define T5 T5_C
#define T6 T6_C
#define T7 T7_C
	            

__global ushort *flagA = &(flag1[gid-get_global_offset(0)]);
__global ushort *flagB = &(flag2[gid-get_global_offset(0)]);
flagA[0]= 0x01;
            if ( (hash->U4[0] & 0x1) && flagB[0]==0x01 ) {
flagA[0]= 0xff;

    sph_u64 H[16];
    for (unsigned int u = 0; u < 15; u ++)
        H[u] = 0;
#if USE_LE
    H[15] = ((sph_u64)(512 & 0xFF) << 56) | ((sph_u64)(512 & 0xFF00) << 40);
#else
    H[15] = (sph_u64)512;
#endif

    sph_u64 g[16], m[16];
    m[0] = hash->U8[0];
    m[1] = hash->U8[1];
    m[2] = hash->U8[2];
    m[3] = hash->U8[3];
    m[4] = hash->U8[4];
    m[5] = hash->U8[5];
    m[6] = hash->U8[6];
    m[7] = hash->U8[7];
    for (unsigned int u = 0; u < 16; u ++)
        g[u] = m[u] ^ H[u];
    m[8] = 0x80; g[8] = m[8] ^ H[8];
    m[9] = 0; g[9] = m[9] ^ H[9];
    m[10] = 0; g[10] = m[10] ^ H[10];
    m[11] = 0; g[11] = m[11] ^ H[11];
    m[12] = 0; g[12] = m[12] ^ H[12];
    m[13] = 0; g[13] = m[13] ^ H[13];
    m[14] = 0; g[14] = m[14] ^ H[14];
    m[15] = 0x100000000000000; g[15] = m[15] ^ H[15];
    PERM_BIG_P(g);
    PERM_BIG_Q(m);
    for (unsigned int u = 0; u < 16; u ++)
        H[u] ^= g[u] ^ m[u];
    sph_u64 xH[16];
    for (unsigned int u = 0; u < 16; u ++)
        xH[u] = H[u];
    PERM_BIG_P(xH);
    for (unsigned int u = 0; u < 16; u ++)
        H[u] ^= xH[u];
    for (unsigned int u = 0; u < 8; u ++)
        hash->U8[u] = H[u + 8];

    barrier(CLK_GLOBAL_MEM_FENCE);             
            }

#undef T0 
#undef T1 
#undef T2 
#undef T3 
#undef T4 
#undef T5 
#undef T6 
#undef T7 

}

__attribute__((reqd_work_group_size(WORKSIZE, 1, 1)))
__kernel void skein3(__global hash_t* hashes,__global ushort* flag1,__global ushort* flag2) 
{
	
uint gid = get_global_id(0);
    __global hash_t *hash = &(hashes[gid-get_global_offset(0)]);
    __global ushort *flagA = &(flag1[gid-get_global_offset(0)]);
	__global ushort *flagB = &(flag2[gid-get_global_offset(0)]);
	if (flagA[0]==0x01 && flagB[0]==0x01) {

// skein

    sph_u64 h0 = SPH_C64(0x4903ADFF749C51CE), h1 = SPH_C64(0x0D95DE399746DF03), h2 = SPH_C64(0x8FD1934127C79BCE), h3 = SPH_C64(0x9A255629FF352CB1), h4 = SPH_C64(0x5DB62599DF6CA7B0), h5 = SPH_C64(0xEABE394CA9D5C3F4), h6 = SPH_C64(0x991112C71A75B523), h7 = SPH_C64(0xAE18A40B660FCC33);
    sph_u64 m0, m1, m2, m3, m4, m5, m6, m7;
    sph_u64 bcount = 0;

    m0 = hash->U8[0];
    m1 = hash->U8[1];
    m2 = hash->U8[2];
    m3 = hash->U8[3];
    m4 = hash->U8[4];
    m5 = hash->U8[5];
    m6 = hash->U8[6];
    m7 = hash->U8[7];
    UBI_BIG(480, 64);
    bcount = 0;
    m0 = m1 = m2 = m3 = m4 = m5 = m6 = m7 = 0;
    UBI_BIG(510, 8);
    hash->U8[0] = h0;
    hash->U8[1] = h1;
    hash->U8[2] = h2;
    hash->U8[3] = h3;
    hash->U8[4] = h4;
    hash->U8[5] = h5;
    hash->U8[6] = h6;
    hash->U8[7] = h7;
 
    barrier(CLK_GLOBAL_MEM_FENCE);

          }
	
		
}


__attribute__((reqd_work_group_size(WORKSIZE, 1, 1)))
__kernel void blake3(__global hash_t* hashes,__global ushort* flag1, __global ushort* flag2) 
{

uint gid = get_global_id(0);
    __global hash_t *hash = &(hashes[gid-get_global_offset(0)]);
__global ushort *flagA = &(flag1[gid-get_global_offset(0)]);
__global ushort *flagB = &(flag2[gid-get_global_offset(0)]);
flagA[0]= 0x01;
            if ( (hash->U4[0] & 0x1) && flagB[0]==0x01 ) {
flagA[0]= 0xff;

// blake512
     sph_u64 H0 = SPH_C64(0x6A09E667F3BCC908), H1 = SPH_C64(0xBB67AE8584CAA73B);
    sph_u64 H2 = SPH_C64(0x3C6EF372FE94F82B), H3 = SPH_C64(0xA54FF53A5F1D36F1);
    sph_u64 H4 = SPH_C64(0x510E527FADE682D1), H5 = SPH_C64(0x9B05688C2B3E6C1F);
    sph_u64 H6 = SPH_C64(0x1F83D9ABFB41BD6B), H7 = SPH_C64(0x5BE0CD19137E2179);
    sph_u64 S0 = 0, S1 = 0, S2 = 0, S3 = 0;
    sph_u64 T0 = SPH_C64(0xFFFFFFFFFFFFFC00) + (64 << 3), T1 = 0xFFFFFFFFFFFFFFFF;;

    if ((T0 = SPH_T64(T0 + 1024)) < 1024)
    {
        T1 = SPH_T64(T1 + 1);
    }
    sph_u64 M0, M1, M2, M3, M4, M5, M6, M7;
    sph_u64 M8, M9, MA, MB, MC, MD, ME, MF;
    sph_u64 V0, V1, V2, V3, V4, V5, V6, V7;
    sph_u64 V8, V9, VA, VB, VC, VD, VE, VF;
    M0 = DEC64E(hash->U8[0]);
    M1 = DEC64E(hash->U8[1]);
    M2 = DEC64E(hash->U8[2]);
    M3 = DEC64E(hash->U8[3]);
    M4 = DEC64E(hash->U8[4]);
    M5 = DEC64E(hash->U8[5]);
    M6 = DEC64E(hash->U8[6]);
    M7 = DEC64E(hash->U8[7]);
    M8 = 0x8000000000000000;
    M9 = 0;
    MA = 0;
    MB = 0;
    MC = 0;
    MD = 1;
    ME = 0;
    MF = 0x200;

    COMPRESS64;

    hash->U8[0] = DEC64E(H0);
    hash->U8[1] = DEC64E(H1);
    hash->U8[2] = DEC64E(H2);
    hash->U8[3] = DEC64E(H3);
    hash->U8[4] = DEC64E(H4);
    hash->U8[5] = DEC64E(H5);
    hash->U8[6] = DEC64E(H6);
    hash->U8[7] = DEC64E(H7);   
    barrier(CLK_GLOBAL_MEM_FENCE);
//blake512
			
            } 
}

__attribute__((reqd_work_group_size(WORKSIZE, 1, 1)))
__kernel void jh3(__global hash_t* hashes,__global ushort* flag1, __global ushort* flag2) 
{

uint gid = get_global_id(0);

    __global hash_t *hash = &(hashes[gid-get_global_offset(0)]);
    __global ushort *flagA = &(flag1[gid-get_global_offset(0)]);
	__global ushort *flagB = &(flag2[gid-get_global_offset(0)]);
	if (flagA[0]==0x01 && flagB[0]==0x01) {


    sph_u64 h0h = C64e(0x6fd14b963e00aa17), h0l = C64e(0x636a2e057a15d543), h1h = C64e(0x8a225e8d0c97ef0b), h1l = C64e(0xe9341259f2b3c361), h2h = C64e(0x891da0c1536f801e), h2l = C64e(0x2aa9056bea2b6d80), h3h = C64e(0x588eccdb2075baa6), h3l = C64e(0xa90f3a76baf83bf7);
    sph_u64 h4h = C64e(0x0169e60541e34a69), h4l = C64e(0x46b58a8e2e6fe65a), h5h = C64e(0x1047a7d0c1843c24), h5l = C64e(0x3b6e71b12d5ac199), h6h = C64e(0xcf57f6ec9db1f856), h6l = C64e(0xa706887c5716b156), h7h = C64e(0xe3c2fcdfe68517fb), h7l = C64e(0x545a4678cc8cdd4b);
    sph_u64 tmp;

    for(int i = 0; i < 2; i++)
    {
        if (i == 0) {
            h0h ^= hash->U8[0];
            h0l ^= hash->U8[1];
            h1h ^= hash->U8[2];
            h1l ^= hash->U8[3];
            h2h ^= hash->U8[4];
            h2l ^= hash->U8[5];
            h3h ^= hash->U8[6];
            h3l ^= hash->U8[7];
        } else if(i == 1) {
            h4h ^= hash->U8[0];
            h4l ^= hash->U8[1];
            h5h ^= hash->U8[2];
            h5l ^= hash->U8[3];
            h6h ^= hash->U8[4];
            h6l ^= hash->U8[5];
            h7h ^= hash->U8[6];
            h7l ^= hash->U8[7];
        
            h0h ^= 0x80;
            h3l ^= 0x2000000000000;
        }
        E8;
    }
    h4h ^= 0x80;
    h7l ^= 0x2000000000000;

    hash->U8[0] = h4h;
    hash->U8[1] = h4l;
    hash->U8[2] = h5h;
    hash->U8[3] = h5l;
    hash->U8[4] = h6h;
    hash->U8[5] = h6l;
    hash->U8[6] = h7h;
    hash->U8[7] = h7l;
    barrier(CLK_LOCAL_MEM_FENCE);			
		  }
}

__attribute__((reqd_work_group_size(WORKSIZE, 1, 1)))
__kernel void finale(__global hash_t* hashes, volatile __global uint * output, const ulong target,__global ushort* flag2) 
{

      uint gid = get_global_id(0);
    __global hash_t *hash = &(hashes[gid-get_global_offset(0)]);
   __global ushort *flagB = &(flag2[gid-get_global_offset(0)]);

    bool result = (flagB[0]!=0xa1 && (hash->U8[3] <= target));
    if (result)
	output[atomic_inc(output+0xFF)] = gid;
	  
}
#endif // ADVSHA3_CL
