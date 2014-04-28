/*
 * AdvSHA3 kernel implementation.
 *
 * @author  mkimid
 */

#ifndef ADVSHA3_CL
#define ADVSHA3_CL

#define SPH_ROTL64(x, n)                  rotate(x, (ulong)(n))
#define SPH_ROTR64(x, n)                  SPH_ROTL64(x, (64 - (n)))

#define SWAP4(x)                          as_uint(as_uchar4(x).wzyx)
#define SWAP8(x)                          as_ulong(as_uchar8(x).s76543210)

#include "__blake.cl"
#include "__groestl.cl"
#include "__jh.cl"
#include "__keccak.cl"
#include "__skein.cl"


void BLAKE512(ulong X[])	{
     ulong H0 = BLAKE_INI[0], H1 = BLAKE_INI[1];
     ulong H2 = BLAKE_INI[2], H3 = BLAKE_INI[3];
     ulong H4 = BLAKE_INI[4], H5 = BLAKE_INI[5];
     ulong H6 = BLAKE_INI[6], H7 = BLAKE_INI[7];
     ulong M0, M1, M2, M3, M4, M5, M6, M7;
     ulong M8, M9, MA, MB, MC, MD, ME, MF;
     ulong V0, V1, V2, V3, V4, V5, V6, V7;
     ulong V8, V9, VA, VB, VC, VD, VE, VF;
     M0 = SWAP8(X[0]);
     M1 = SWAP8(X[1]);
     M2 = SWAP8(X[2]);
     M3 = SWAP8(X[3]);
     M4 = SWAP8(X[4]);
     M5 = SWAP8(X[5]);
     M6 = SWAP8(X[6]);
     M7 = SWAP8(X[7]);
     M8 = 0x8000000000000000;
     M9 = 0;
     MA = 0;
     MB = 0;
     MC = 0;
     MD = 1;
     ME = 0;
     MF = 0x200;
     V0 = H0;
     V1 = H1;
     V2 = H2;
     V3 = H3;
     V4 = H4;
     V5 = H5;
     V6 = H6;
     V7 = H7;
     V8 = CB0;
     V9 = CB1;
     VA = CB2;
     VB = CB3;
     VC = CB4 ^ 0x200;
     VD = CB5 ^ 0x200;
     VE = CB6;
     VF = CB7;
     ROUND_B(0);
     ROUND_B(1);
     ROUND_B(2);
     ROUND_B(3);
     ROUND_B(4);
     ROUND_B(5);
     ROUND_B(6);
     ROUND_B(7);
     ROUND_B(8);
     ROUND_B(9);
     ROUND_B(0);
     ROUND_B(1);
     ROUND_B(2);
     ROUND_B(3);
     ROUND_B(4);
     ROUND_B(5);
     X[0] = SWAP8(H0 ^ V0 ^ V8);
     X[1] = SWAP8(H1 ^ V1 ^ V9);
     X[2] = SWAP8(H2 ^ V2 ^ VA);
     X[3] = SWAP8(H3 ^ V3 ^ VB);
     X[4] = SWAP8(H4 ^ V4 ^ VC);
     X[5] = SWAP8(H5 ^ V5 ^ VD);
     X[6] = SWAP8(H6 ^ V6 ^ VE);
     X[7] = SWAP8(H7 ^ V7 ^ VF);
}


void GROESTL512(ulong X[], __local const ulong LT0[256],
                           __local const ulong LT1[256],
                           __local const ulong LT2[256],
                           __local const ulong LT3[256],
                           __local const ulong LT4[256],
                           __local const ulong LT5[256],
                           __local const ulong LT6[256],
                           __local const ulong LT7[256]) {

     ulong g[16], m[16], x[16], t[16];
	 ulong H[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x2000000000000UL };
     for (uint u = 0; u < 8; u++) m[u] = X[u];
     for (uint u = 0; u < 8; u++) g[u] = m[u];
     m[0x08] = 0x80;              g[0x08] = 0x80;
     m[0x09] = 0;                 g[0x09] = 0;
     m[0x0A] = 0;                 g[0x0A] = 0;
     m[0x0B] = 0;                 g[0x0B] = 0;
     m[0x0C] = 0;                 g[0x0C] = 0;
     m[0x0D] = 0;                 g[0x0D] = 0;
     m[0x0E] = 0;                 g[0x0E] = 0;
     m[0x0F] = 0x100000000000000; g[0x0F] = m[0x0F] ^ H[0x0F];
     for (uint r = 0; r < 14; r++) {
         ROUND_BIG_P(g, r);
     }
     for (uint r = 0; r < 14; r++) {
          ROUND_BIG_Q(m, r);
     }
     for (uint u = 0; u < 16; u++) x[u]  = (H[u] ^= (g[u] ^ m[u]));
     for (uint r = 0; r < 14; r++) {
         ROUND_BIG_P(x, r);
     }
     for (uint u = 8; u < 16; u++) H[u] ^= x[u];
	 for (uint u = 0; u <  8; u++) X[u]  = H[u + 0x08];
}


void JH512(ulong X[]) {
     ulong h0h = JH_INI[0x00], h0l = JH_INI[0x01];
	 ulong h1h = JH_INI[0x02], h1l = JH_INI[0x03];
     ulong h2h = JH_INI[0x04], h2l = JH_INI[0x05];
     ulong h3h = JH_INI[0x06], h3l = JH_INI[0x07];
     ulong h4h = JH_INI[0x08], h4l = JH_INI[0x09];
	 ulong h5h = JH_INI[0x0A], h5l = JH_INI[0x0B];
	 ulong h6h = JH_INI[0x0C], h6l = JH_INI[0x0D];
     ulong h7h = JH_INI[0x0E], h7l = JH_INI[0x0F];
     ulong tmp, t;
     for(uint i = 0; i < 2; i++) {
        if (i == 0) {
            h0h ^= X[0x00];
            h0l ^= X[0x01];
            h1h ^= X[0x02];
            h1l ^= X[0x03];
            h2h ^= X[0x04];
            h2l ^= X[0x05];
            h3h ^= X[0x06];
            h3l ^= X[0x07];
        } else if (i == 1) {
            h4h ^= X[0x00];
            h4l ^= X[0x01];
            h5h ^= X[0x02];
            h5l ^= X[0x03];
            h6h ^= X[0x04];
            h6l ^= X[0x05];
            h7h ^= X[0x06];
            h7l ^= X[0x07];
            h0h ^= 0x80;
            h3l ^= 0x2000000000000;
        }
        SLu( 0, 0);
        SLu( 1, 1);
        SLu( 2, 2);
        SLu( 3, 3);
        SLu( 4, 4);
        SLu( 5, 5);
        SLu( 6, 6);
        SLu( 7, 0);
        SLu( 8, 1);
        SLu( 9, 2);
        SLu(10, 3);
        SLu(11, 4);
        SLu(12, 5);
        SLu(13, 6);
        SLu(14, 0);
        SLu(15, 1);
        SLu(16, 2);
        SLu(17, 3);
        SLu(18, 4);
        SLu(19, 5);
        SLu(20, 6);
        SLu(21, 0);
        SLu(22, 1);
        SLu(23, 2);
        SLu(24, 3);
        SLu(25, 4);
        SLu(26, 5);
        SLu(27, 6);
        SLu(28, 0);
        SLu(29, 1);
        SLu(30, 2);
        SLu(31, 3);
        SLu(32, 4);
        SLu(33, 5);
        SLu(34, 6);
        SLu(35, 0);
        SLu(36, 1);
        SLu(37, 2);
        SLu(38, 3);
        SLu(39, 4);
        SLu(40, 5);
        SLu(41, 6);
     }
     h4h ^= 0x80;
     h7l ^= 0x2000000000000;
     X[0x00] = h4h;
     X[0x01] = h4l;
     X[0x02] = h5h;
     X[0x03] = h5l;
     X[0x04] = h6h;
     X[0x05] = h6l;
     X[0x06] = h7h;
     X[0x07] = h7l;
}


void KECCAK512_80(ulong X[]) {
     ulong c0x, c1x, c2x, c3x, c4x;
     ulong a00 =  X[0];
     ulong a10 = ~X[1];
     ulong a20 = ~X[2];
     ulong a30 =  X[3];
     ulong a40 =  X[4];
     ulong a01 =  X[5];
     ulong a11 =  X[6];
     ulong a21 =  X[7];
     ulong a31 = ~X[8];
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
	 a00 ^=  X[9];
     a10 ^=  0x01;
     a31 ^=  0x8000000000000000UL;
     KECCAK_F_1600;
     a10     = ~a10;
     a20     = ~a20;
     X[0x00] =  a00;
     X[0x01] =  a10;
     X[0x02] =  a20;
     X[0x03] =  a30;
     X[0x04] =  a40;
     X[0x05] =  a01;
     X[0x06] =  a11;
     X[0x07] =  a21;
}


void KECCAK512_88(ulong X[]) {
     ulong c0x, c1x, c2x, c3x, c4x;
     ulong a00 =  X[0];
     ulong a10 = ~X[1];
     ulong a20 = ~X[2];
     ulong a30 =  X[3];
     ulong a40 =  X[4];
     ulong a01 =  X[5];
     ulong a11 =  X[6];
     ulong a21 =  X[7];
     ulong a31 = ~X[8];
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
	 a00 ^=  X[9];
     a10 ^=  X[10];
     a20 ^=  0x01;
     a31 ^=  0x8000000000000000UL;
     KECCAK_F_1600;
     a10     = ~a10;
     a20     = ~a20;
     X[0x00] =  a00;
     X[0x01] =  a10;
     X[0x02] =  a20;
     X[0x03] =  a30;
     X[0x04] =  a40;
     X[0x05] =  a01;
     X[0x06] =  a11;
     X[0x07] =  a21;
}

void SKEIN512(ulong X[]) {
     ulong h0 = SKEIN_INI[0x00], h1 = SKEIN_INI[0x01];
     ulong h2 = SKEIN_INI[0x02], h3 = SKEIN_INI[0x03];
     ulong h4 = SKEIN_INI[0x04], h5 = SKEIN_INI[0x05];
     ulong h6 = SKEIN_INI[0x06], h7 = SKEIN_INI[0x07];
     ulong m0 = X[0x00];
     ulong m1 = X[0x01];
     ulong m2 = X[0x02];
     ulong m3 = X[0x03];
     ulong m4 = X[0x04];
     ulong m5 = X[0x05];
     ulong m6 = X[0x06];
     ulong m7 = X[0x07];
     ulong bcount = 0;
     UBI_BIG(480, 64);
     bcount = 0;
     m0 = m1 = m2 = m3 = m4 = m5 = m6 = m7 = 0;
     UBI_BIG(510, 8);
     X[0x00] = h0;
     X[0x01] = h1;
     X[0x02] = h2;
     X[0x03] = h3;
     X[0x04] = h4;
     X[0x05] = h5;
     X[0x06] = h6;
     X[0x07] = h7;
}

__attribute__((reqd_work_group_size(WORKSIZE, 1, 1)))
__kernel void search(__global uint * input, volatile __global uint * output, const ulong target) {

   __local ulong LT0[256], LT1[256], LT2[256], LT3[256], LT4[256], LT5[256], LT6[256], LT7[256];
     uint init = get_local_id(0);
     uint step = get_local_size(0);
     for (uint i = init; i < 256; i += step) {
         LT0[i] = T0[i];
         LT1[i] = T1[i];
         LT2[i] = T2[i];
         LT3[i] = T3[i];
         LT4[i] = T4[i];
         LT5[i] = T5[i];
         LT6[i] = T6[i];
         LT7[i] = T7[i];
     }
     barrier(CLK_LOCAL_MEM_FENCE);

     uint  gid = get_global_id(0);

     union {
        uint  U4[22];
        ulong U8[11];
     } HASH;

     for (uint i = 0; i < 22; i++) {
         HASH.U4[i] = input[i];
     }
     HASH.U4[19] = SWAP4(gid);

     //
     // Dual Version Mode
     //

     if (HASH.U4[21] == 7) {
        KECCAK512_88(HASH.U8);
        if ((HASH.U8[3] <= target) && ((HASH.U4[0] & 7) == 0)) {
           output[output[0xFF]++] = gid;
        }
     }
     else {
        KECCAK512_80(HASH.U8);
        for (uint round = 0; round < 3; round++) {
            if (HASH.U4[0] & 0x1) {
               GROESTL512(HASH.U8, LT0, LT1, LT2, LT3, LT4, LT5, LT6, LT7);
            }
            else {
               SKEIN512(HASH.U8);
            }
            if (HASH.U4[0] & 0x01) {
               BLAKE512(HASH.U8);
            }
            else {
               JH512(HASH.U8);
            }
        }
        if (HASH.U8[3] <= target) {
           output[output[0xFF]++] = gid;
        }
     }

}

#endif // ADVSHA3_CL
