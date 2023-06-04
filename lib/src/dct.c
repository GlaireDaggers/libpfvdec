#include <stdint.h>

#include "dct.h"

static size_t INV_ZIGZAG_TABLE[64] = {
	0, 1, 5, 6, 14, 15, 27, 28, 2, 4, 7, 13, 16, 26, 29, 42, 3, 8, 12, 17, 25, 30, 41, 43, 9, 11, 18, 24, 31, 40, 44, 53, 10, 19, 23, 32, 39, 45, 52, 54, 20, 22,
	33, 38, 46, 51, 55, 60, 21, 34, 37, 47, 50, 56, 59, 61, 35, 36, 48, 49, 57, 58, 62, 63
};

void idct(int32_t* vector, int32_t stride) {
    // extract rows (with input permutation)
    int32_t c0 = vector[0 * stride];
    int32_t d4 = vector[1 * stride];
    int32_t c2 = vector[2 * stride];
    int32_t d6 = vector[3 * stride];
    int32_t c1 = vector[4 * stride];
    int32_t d5 = vector[5 * stride];
    int32_t c3 = vector[6 * stride];
    int32_t d7 = vector[7 * stride];

    // odd stage 4
    int32_t c4 = d4;
    int32_t c5 = d5 + d6;
    int32_t c7 = d5 - d6;
    int32_t c6 = d7;

    // odd stage 3
    int32_t b4 = c4 + c5;
    int32_t b5 = c4 - c5;
    int32_t b6 = c6 + c7;
    int32_t b7 = c6 - c7;

    // even stage 3
    int32_t b0 = c0 + c1;
    int32_t b1 = c0 - c1;
    int32_t b2 = c2 + c2 / 4 + c3 / 2;
    int32_t b3 = c2 / 2 - c3 - c3 / 4;

    // odd stage 2
    int32_t a4 = b7 / 4 + b4 + b4 / 4 - b4 / 16;
    int32_t a7 = b4 / 4 - b7 - b7 / 4 + b7 / 16;
    int32_t a5 = b5 - b6 + b6 / 4 + b6 / 16;
    int32_t a6 = b6 + b5 - b5 / 4 - b5 / 16;

    // even stage 2
    int32_t a0 = b0 + b2;
    int32_t a1 = b1 + b3;
    int32_t a2 = b1 - b3;
    int32_t a3 = b0 - b2;

    // stage 1
    vector[0 * stride] = a0 + a4;
    vector[1 * stride] = a1 + a5;
    vector[2 * stride] = a2 + a6;
    vector[3 * stride] = a3 + a7;
    vector[4 * stride] = a3 - a7;
    vector[5 * stride] = a2 - a6;
    vector[6 * stride] = a1 - a5;
    vector[7 * stride] = a0 - a4;

    // total: 36A 12S
}

void dct_decode(int16_t* src, int32_t* q_table, int32_t* result) {
	// note that qtable already contains 24.8 scale factors, so we don't need to convert to 24.8 explicitly at this step
	for (int i = 0; i < 64; i++) {
		size_t idx = INV_ZIGZAG_TABLE[i];

		int32_t n = (int32_t)src[idx];
		int32_t d = q_table[idx];

		result[i] = n * d;
	}

	// inverse transform columns
	for (int i = 0; i < 8; i++) {
		idct(result + i, 8);
	}

	// inverse transform rows
	for (int i = 0; i < 8; i++) {
		idct(result + (i * 8), 1);
	}
}