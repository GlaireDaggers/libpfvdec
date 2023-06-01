#include <stdint.h>

#include "dct.h"

#define S_0 90
#define S_1 65
#define S_2 69
#define S_3 76
#define S_4 90
#define S_5 115
#define S_6 167
#define S_7 328

#define RS_0 724
#define RS_1 1004
#define RS_2 946
#define RS_3 851
#define RS_4 724
#define RS_5 568
#define RS_6 391
#define RS_7 199

#define A_1 181
#define A_2 138
#define A_3 181
#define A_4 334
#define A_5 97

#define RA_1 362
#define RA_2 473
#define RA_3 362
#define RA_4 195
#define RA_5 668

#define F -256

static size_t INV_ZIGZAG_TABLE[64] = {
	0, 1, 5, 6, 14, 15, 27, 28, 2, 4, 7, 13, 16, 26, 29, 42, 3, 8, 12, 17, 25, 30, 41, 43, 9, 11, 18, 24, 31, 40, 44, 53, 10, 19, 23, 32, 39, 45, 52, 54, 20, 22,
	33, 38, 46, 51, 55, 60, 21, 34, 37, 47, 50, 56, 59, 61, 35, 36, 48, 49, 57, 58, 62, 63
};

void fast_dct8_inverse_transform(int32_t* vector, int32_t stride) {
	int v15 = (vector[0 * stride] * RS_0) >> FP_BITS;
	int v26 = (vector[1 * stride] * RS_1) >> FP_BITS;
	int v21 = (vector[2 * stride] * RS_2) >> FP_BITS;
	int v28 = (vector[3 * stride] * RS_3) >> FP_BITS;
	int v16 = (vector[4 * stride] * RS_4) >> FP_BITS;
	int v25 = (vector[5 * stride] * RS_5) >> FP_BITS;
	int v22 = (vector[6 * stride] * RS_6) >> FP_BITS;
	int v27 = (vector[7 * stride] * RS_7) >> FP_BITS;

	int v19 = (v25 - v28) >> 1;
	int v20 = (v26 - v27) >> 1;
	int v23 = (v26 + v27) >> 1;
	int v24 = (v25 + v28) >> 1;

	int v7 = (v23 + v24) >> 1;
	int v11 = (v21 + v22) >> 1;
	int v13 = (v23 - v24) >> 1;
	int v17 = (v21 - v22) >> 1;

	int v8 = (v15 + v16) >> 1;
	int v9 = (v15 - v16) >> 1;

	int v18 = ((v19 - v20) * A_5) >> FP_BITS;  // Different from original
	int v12 = ((((v19 * A_4) >> FP_BITS) - v18) * F) >> FP_BITS;
	int v14 = ((v18 - ((v20 * A_2) >> FP_BITS)) * F) >> FP_BITS;

	int v6 = v14 - v7;
	int v5 = ((v13 * RA_3) >> FP_BITS) - v6;
	int v4 = -v5 - v12;
	int v10 = ((v17 * RA_1) >> FP_BITS) - v11;

	int v0 = (v8 + v11) >> 1;
	int v1 = (v9 + v10) >> 1;
	int v2 = (v9 - v10) >> 1;
	int v3 = (v8 - v11) >> 1;

	vector[0 * stride] = (v0 + v7) >> 1;
	vector[1 * stride] = (v1 + v6) >> 1;
	vector[2 * stride] = (v2 + v5) >> 1;
	vector[3 * stride] = (v3 + v4) >> 1;
	vector[4 * stride] = (v3 - v4) >> 1;
	vector[5 * stride] = (v2 - v5) >> 1;
	vector[6 * stride] = (v1 - v6) >> 1;
	vector[7 * stride] = (v0 - v7) >> 1;
}

void dct_decode(int16_t* src, int32_t* q_table, int32_t* result) {
	for (int i = 0; i < 64; i++) {
		int32_t n = (int32_t)src[INV_ZIGZAG_TABLE[i]] << FP_BITS;
		int32_t d = q_table[i];

		result[i] = n * d;
	}

	// inverse transform columns
	for (int i = 0; i < 8; i++) {
		fast_dct8_inverse_transform(result + i, 8);
	}

	// inverse transform rows
	for (int i = 0; i < 8; i++) {
		fast_dct8_inverse_transform(result + (i * 8), 1);
	}
}