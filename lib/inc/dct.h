#pragma once

// 24.8 fixed point
#define FP_BITS 8

void dct_decode(int16_t *src, int32_t *q_table, int32_t *result);