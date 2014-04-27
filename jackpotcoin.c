/*
 * Copyright 2014 mkimid
 *
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 *
 */

#include "config.h"
#include "miner.h"

#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#include "sph/sph_blake.h"
#include "sph/sph_groestl.h"
#include "sph/sph_skein.h"
#include "sph/sph_jh.h"
#include "sph/sph_keccak.h"

/*
 * Convert uint LE vector to BE format
 */
static inline void
be32enc_vect(uint32_t *dst, const uint32_t *src, uint32_t len) {
	uint32_t i;
	for (i = 0; i < len; i++) dst[i] = htobe32(src[i]);
}


static void advsha3_hash(void *state, const void *input) {
    sph_blake512_context     ctx_blake;
    sph_groestl512_context   ctx_groestl;
    sph_jh512_context        ctx_jh;
    sph_keccak512_context    ctx_keccak;
    sph_skein512_context     ctx_skein;

    uint32_t hash[16];

    if (opt_debughash) {
       unsigned char * ptr = (unsigned char *)input;
       int ii;
       printf("hash input : ");
       for (ii = 0; ii < 88; ii++) {
           printf("%.2x", (unsigned char)ptr[ii]);
       }
       printf("\n");
       ptr = (unsigned char *)(&hash[0]);
    }

    unsigned int round_mask = (
       (unsigned int)(((unsigned char *)input)[84]) <<  0 |
       (unsigned int)(((unsigned char *)input)[85]) <<  8 |
       (unsigned int)(((unsigned char *)input)[86]) << 16 |
       (unsigned int)(((unsigned char *)input)[87]) << 24 );

    sph_keccak512_init(&ctx_keccak);
    sph_keccak512 (&ctx_keccak, input, 80);
    sph_keccak512_close(&ctx_keccak, (&hash));

    unsigned int round;
    for (round = 0; round < 3; round++) {
        if (hash[0] & 0x01) {
           sph_groestl512_init(&ctx_groestl);
           sph_groestl512 (&ctx_groestl, (&hash), 64);
           sph_groestl512_close(&ctx_groestl, (&hash));
        }
        else {
           sph_skein512_init(&ctx_skein);
           sph_skein512 (&ctx_skein, (&hash), 64);
           sph_skein512_close(&ctx_skein, (&hash));
        }
        if (hash[0] & 0x01) {
           sph_blake512_init(&ctx_blake);
           sph_blake512 (&ctx_blake, (&hash), 64);
           sph_blake512_close(&ctx_blake, (&hash));
        }
        else {
           sph_jh512_init(&ctx_jh);
           sph_jh512 (&ctx_jh, (&hash), 64);
           sph_jh512_close(&ctx_jh, (&hash));
        }
        if (opt_debughash) {
           unsigned char * ptr = (unsigned char *)(&hash[0]);
           int ii;
           printf("round %d : ", round);
           for (ii = 0; ii < 64; ii++) {
               printf("%.2x", (unsigned char)ptr[ii]);
           }
           printf("\n");
        }
    }

	memcpy(state, hash, 32);
}

void advsha3_regenhash(struct work *work) {
     uint32_t data[22];
     uint32_t *nonce = (uint32_t *)(work->data + 76);
     uint32_t *ohash = (uint32_t *)(work->hash);
     be32enc_vect(data, (const uint32_t *)work->data, 22);
     data[19] = htobe32(*nonce);
     advsha3_hash(ohash, data);
}


static const uint32_t diff1targ = 0x0000ffff;


/* Used externally as confirmation of correct OCL code */
int advsha3_test(unsigned char *pdata, const unsigned char *ptarget, uint32_t nonce)
{
	uint32_t tmp_hash7, Htarg = le32toh(((const uint32_t *)ptarget)[7]);
	uint32_t data[20], ohash[8];

	be32enc_vect(data, (const uint32_t *)pdata, 19);
	data[19] = htobe32(nonce);

	advsha3_hash(ohash, data);
	tmp_hash7 = be32toh(ohash[7]);

	applog(LOG_DEBUG, "htarget %08lx diff1 %08lx hash %08lx",
				(long unsigned int)Htarg,
				(long unsigned int)diff1targ,
				(long unsigned int)tmp_hash7);
	if (tmp_hash7 > diff1targ)
		return -1;
	if (tmp_hash7 > Htarg)
		return 0;
	return 1;
}

bool advsha3_scanhash(struct thr_info *thr, const unsigned char __maybe_unused *pmidstate,
		     unsigned char *pdata, unsigned char __maybe_unused *phash1,
		     unsigned char __maybe_unused *phash, const unsigned char *ptarget,
		     uint32_t max_nonce, uint32_t *last_nonce, uint32_t n)
{
	uint32_t *nonce = (uint32_t *)(pdata + 76);
	char *scratchbuf;
	uint32_t data[20];
	uint32_t tmp_hash7;
	uint32_t Htarg = le32toh(((const uint32_t *)ptarget)[7]);
	bool ret = false;

	be32enc_vect(data, (const uint32_t *)pdata, 19);

	while(1) {
		uint32_t ostate[8];

		*nonce = ++n;
		data[19] = (n);
		advsha3_hash(ostate, data);
		tmp_hash7 = (ostate[7]);

		applog(LOG_INFO, "data7 %08lx", (long unsigned int)data[7]);

		if (unlikely(tmp_hash7 <= Htarg)) {
			((uint32_t *)pdata)[19] = htobe32(n);
			*last_nonce = n;
			ret = true;
			break;
		}

		if (unlikely((n >= max_nonce) || thr->work_restart)) {
			*last_nonce = n;
			break;
		}
	}

	return ret;
}
