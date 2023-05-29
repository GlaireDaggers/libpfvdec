#include <stdint.h>

#include "dct.h"

// adapted from https://www.nayuki.io/page/fast-discrete-cosine-transform-algorithms

// in the original these are arrays, but are only ever referred to with constant indices, so we can just unfold the whole array
// also storing reciprocals to replace divides with multiplies

#define S_1 0.254897789552079584470970f
#define S_2 0.270598050073098492199862f
#define S_3 0.300672443467522640271861f
#define S_0 0.353553390593273762200422f
#define S_4 0.353553390593273762200422f
#define S_5 0.449988111568207852319255f
#define S_6 0.653281482438188263928322f
#define S_7 1.281457723870753089398043f

#define RS_0 (1.0f / S_0)
#define RS_1 (1.0f / S_1)
#define RS_2 (1.0f / S_2)
#define RS_3 (1.0f / S_3)
#define RS_4 (1.0f / S_4)
#define RS_5 (1.0f / S_5)
#define RS_6 (1.0f / S_6)
#define RS_7 (1.0f / S_7)

#define A_1 0.707106781186547524400844f
#define A_2 0.541196100146196984399723f
#define A_3 0.707106781186547524400844f
#define A_4 1.306562964876376527856643f
#define A_5 0.382683432365089771728460f

#define RA_1 (1.0f / A_1)
#define RA_2 (1.0f / A_2)
#define RA_3 (1.0f / A_3)
#define RA_4 (1.0f / A_4)
#define RA_5 (1.0f / A_5)

#define F (1.0f / (A_2 * A_5 - A_2 * A_4 - A_4 * A_5))

static size_t INV_ZIGZAG_TABLE[64] = {
    0, 1, 5, 6, 14, 15, 27, 28, 2, 4, 7, 13, 16, 26, 29, 42, 3, 8, 12, 17, 25, 30, 41, 43, 9, 11, 18, 24, 31, 40, 44, 53, 10, 19, 23, 32, 39, 45, 52, 54, 20, 22,
    33, 38, 46, 51, 55, 60, 21, 34, 37, 47, 50, 56, 59, 61, 35, 36, 48, 49, 57, 58, 62, 63
};

void fast_dct8_inverse_transform(float *vector, int stride) {
    float v15 = vector[0 * stride] * RS_0;
    float v26 = vector[1 * stride] * RS_1;
    float v21 = vector[2 * stride] * RS_2;
    float v28 = vector[3 * stride] * RS_3;
    float v16 = vector[4 * stride] * RS_4;
    float v25 = vector[5 * stride] * RS_5;
    float v22 = vector[6 * stride] * RS_6;
    float v27 = vector[7 * stride] * RS_7;
    
    float v19 = (v25 - v28) * 0.5f;
    float v20 = (v26 - v27) * 0.5f;
    float v23 = (v26 + v27) * 0.5f;
    float v24 = (v25 + v28) * 0.5f;
    
    float v7  = (v23 + v24) * 0.5f;
    float v11 = (v21 + v22) * 0.5f;
    float v13 = (v23 - v24) * 0.5f;
    float v17 = (v21 - v22) * 0.5f;
    
    float v8 = (v15 + v16) * 0.5f;
    float v9 = (v15 - v16) * 0.5f;
    
    float v18 = (v19 - v20) * A_5;  // Different from original
    float v12 = (v19 * A_4 - v18) * F;
    float v14 = (v18 - v20 * A_2) * F;
    
    float v6 = v14 - v7;
    float v5 = v13 * RA_3 - v6;
    float v4 = -v5 - v12;
    float v10 = v17 * RA_1 - v11;
    
    float v0 = (v8 + v11) * 0.5f;
    float v1 = (v9 + v10) * 0.5f;
    float v2 = (v9 - v10) * 0.5f;
    float v3 = (v8 - v11) * 0.5f;
    
    vector[0 * stride] = (v0 + v7) * 0.5f;
    vector[1 * stride] = (v1 + v6) * 0.5f;
    vector[2 * stride] = (v2 + v5) * 0.5f;
    vector[3 * stride] = (v3 + v4) * 0.5f;
    vector[4 * stride] = (v3 - v4) * 0.5f;
    vector[5 * stride] = (v2 - v5) * 0.5f;
    vector[6 * stride] = (v1 - v6) * 0.5f;
    vector[7 * stride] = (v0 - v7) * 0.5f;
}

void dct_decode(int16_t *src, float *q_table, float *result) {
    for (int i = 0; i < 64; i++) {
        float n = (float)src[INV_ZIGZAG_TABLE[i]];
        float d = q_table[i];

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