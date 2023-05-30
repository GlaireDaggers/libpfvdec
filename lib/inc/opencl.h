#pragma once

#define STRINGIFY(x) #x

const char *OPENCL_KERNEL =
    "// adapted from https://www.nayuki.io/page/fast-discrete-cosine-transform-algorithms\n"
    "\n"
    "// in the original these are arrays, but are only ever referred to with constant indices, so we can just unfold the whole array\n"
    "// also storing reciprocals to replace divides with multiplies\n"
    "\n"
    "#define S_1 0.254897789552079584470970\n"
    "#define S_2 0.270598050073098492199862\n"
    "#define S_3 0.300672443467522640271861\n"
    "#define S_0 0.353553390593273762200422\n"
    "#define S_4 0.353553390593273762200422\n"
    "#define S_5 0.449988111568207852319255\n"
    "#define S_6 0.653281482438188263928322\n"
    "#define S_7 1.281457723870753089398043\n"
    "\n"
    "#define RS_0 (1.0f / S_0)\n"
    "#define RS_1 (1.0f / S_1)\n"
    "#define RS_2 (1.0f / S_2)\n"
    "#define RS_3 (1.0f / S_3)\n"
    "#define RS_4 (1.0f / S_4)\n"
    "#define RS_5 (1.0f / S_5)\n"
    "#define RS_6 (1.0f / S_6)\n"
    "#define RS_7 (1.0f / S_7)\n"
    "\n"
    "#define A_1 0.707106781186547524400844\n"
    "#define A_2 0.541196100146196984399723\n"
    "#define A_3 0.707106781186547524400844\n"
    "#define A_4 1.306562964876376527856643\n"
    "#define A_5 0.382683432365089771728460\n"
    "\n"
    "#define RA_1 (1.0f / A_1)\n"
    "#define RA_2 (1.0f / A_2)\n"
    "#define RA_3 (1.0f / A_3)\n"
    "#define RA_4 (1.0f / A_4)\n"
    "#define RA_5 (1.0f / A_5)\n"
    "\n"
    "#define F (1.0f / (A_2 * A_5 - A_2 * A_4 - A_4 * A_5))\n"
    "\n"
    "const uint INV_ZIGZAG_TABLE[64] = {\n"
    "    0, 1, 5, 6, 14, 15, 27, 28, 2, 4, 7, 13, 16, 26, 29, 42, 3, 8, 12, 17, 25, 30, 41, 43, 9, 11, 18, 24, 31, 40, 44, 53, 10, 19, 23, 32, 39, 45, 52, 54, 20, 22,\n"
    "    33, 38, 46, 51, 55, 60, 21, 34, 37, 47, 50, 56, 59, 61, 35, 36, 48, 49, 57, 58, 62, 63\n"
    "};\n"
    "\n"
    "void fast_dct8_inverse_transform(local float* vector, int stride) {\n"
    "\tfloat v15 = vector[0 * stride] * RS_0;\n"
    "    float v26 = vector[1 * stride] * RS_1;\n"
    "    float v21 = vector[2 * stride] * RS_2;\n"
    "    float v28 = vector[3 * stride] * RS_3;\n"
    "    float v16 = vector[4 * stride] * RS_4;\n"
    "    float v25 = vector[5 * stride] * RS_5;\n"
    "    float v22 = vector[6 * stride] * RS_6;\n"
    "    float v27 = vector[7 * stride] * RS_7;\n"
    "    \n"
    "    float v19 = (v25 - v28) * 0.5;\n"
    "    float v20 = (v26 - v27) * 0.5;\n"
    "    float v23 = (v26 + v27) * 0.5;\n"
    "    float v24 = (v25 + v28) * 0.5;\n"
    "    \n"
    "    float v7  = (v23 + v24) * 0.5;\n"
    "    float v11 = (v21 + v22) * 0.5;\n"
    "    float v13 = (v23 - v24) * 0.5;\n"
    "    float v17 = (v21 - v22) * 0.5;\n"
    "    \n"
    "    float v8 = (v15 + v16) * 0.5;\n"
    "    float v9 = (v15 - v16) * 0.5;\n"
    "    \n"
    "    float v18 = (v19 - v20) * A_5;  // Different from original\n"
    "    float v12 = (v19 * A_4 - v18) * F;\n"
    "    float v14 = (v18 - v20 * A_2) * F;\n"
    "    \n"
    "    float v6 = v14 - v7;\n"
    "    float v5 = v13 * RA_3 - v6;\n"
    "    float v4 = -v5 - v12;\n"
    "    float v10 = v17 * RA_1 - v11;\n"
    "    \n"
    "    float v0 = (v8 + v11) * 0.5;\n"
    "    float v1 = (v9 + v10) * 0.5;\n"
    "    float v2 = (v9 - v10) * 0.5;\n"
    "    float v3 = (v8 - v11) * 0.5;\n"
    "    \n"
    "    vector[0 * stride] = (v0 + v7) * 0.5;\n"
    "    vector[1 * stride] = (v1 + v6) * 0.5;\n"
    "    vector[2 * stride] = (v2 + v5) * 0.5;\n"
    "    vector[3 * stride] = (v3 + v4) * 0.5;\n"
    "    vector[4 * stride] = (v3 - v4) * 0.5;\n"
    "    vector[5 * stride] = (v2 - v5) * 0.5;\n"
    "    vector[6 * stride] = (v1 - v6) * 0.5;\n"
    "    vector[7 * stride] = (v0 - v7) * 0.5;\n"
    "}\n"
    "\n"
    "void dct_decode(global short* src, uint src_offset, global float* qtable_buffer, uint qtable_index, local float* result) {\n"
    "    for (int i = 0; i < 64; i++) {\n"
    "\t\tfloat n = convert_float(src[src_offset + INV_ZIGZAG_TABLE[i]]);\n"
    "\t\tfloat d = qtable_buffer[i + (qtable_index * 64)];\n"
    "\t\t\n"
    "\t\tresult[i] = n * d;\n"
    "\t}\n"
    "\t\n"
    "\t// inverse DCT columns\n"
    "\tfor (int i = 0; i < 8; i++) {\n"
    "\t\tfast_dct8_inverse_transform(result + i, 8);\n"
    "\t}\n"
    "\t\n"
    "\t// inverse DCT rows\n"
    "\tfor (int i = 0; i < 8; i++) {\n"
    "\t\tfast_dct8_inverse_transform(result + (i * 8), 1);\n"
    "\t}\n"
    "}\n"
    "\n"
    "void blit_subblock(local float* dct, global uchar* dst, uint dst_w, uint dst_h, uint dx, uint dy) {\n"
    "\tfor (int row = 0; row < 8; row++) {\n"
    "\t\tint dest_row = row + dy;\n"
    "\t\tint src_offset = row * 8;\n"
    "\t\tint dst_offset = (dest_row * dst_w) + dx;\n"
    "\n"
    "\t\tfor (int column = 0; column < 8; column++) {\n"
    "\t\t\tfloat f = dct[src_offset + column] + 128.0;\n"
    "\t\t\tdst[dst_offset + column] = convert_uchar_sat(f);\n"
    "\t\t}\n"
    "\t}\n"
    "}\n"
    "\n"
    "void blit_subblock_delta(local float* dct, global uchar* prev, uint px, uint py, global uchar* dst, uint dst_w, uint dst_h, uint dx, uint dy) {\n"
    "\tfor (int row = 0; row < 8; row++) {\n"
    "\t\tint dest_row = row + dy;\n"
    "\t\tint prev_row = row + py;\n"
    "\t\tint src_offset = row * 8;\n"
    "\t\tint dst_offset = (dest_row * dst_w) + dx;\n"
    "\t\tint prev_offset = (prev_row * dst_w) + px;\n"
    "\n"
    "\t\tfor (int column = 0; column < 8; column++) {\n"
    "\t\t\tfloat f = prev[prev_offset + column] + (dct[src_offset + column] * 2.0);\n"
    "\t\t\tdst[dst_offset + column] = convert_uchar_sat(f);\n"
    "\t\t}\n"
    "\t}\n"
    "}\n"
    "\n"
    "kernel void idct(global short* in_coeff, global float* qtable_buffer, uint qtable_index, global uchar* out_plane) {\n"
    "    uint block_x = get_global_id(0);\n"
    "    uint block_y = get_global_id(1);\n"
    "\n"
    "    uint blocks_wide = get_global_size(0);\n"
    "    uint blocks_high = get_global_size(1);\n"
    "\n"
    "    uint block_idx = block_x + (block_y * blocks_wide);\n"
    "    uint block_offset = block_idx * 256;\n"
    "\n"
    "    local float dct0[64];\n"
    "    local float dct1[64];\n"
    "    local float dct2[64];\n"
    "    local float dct3[64];\n"
    "\n"
    "    dct_decode(in_coeff, block_offset, qtable_buffer, qtable_index, dct0);\n"
    "    dct_decode(in_coeff, block_offset + 64, qtable_buffer, qtable_index, dct1);\n"
    "    dct_decode(in_coeff, block_offset + 128, qtable_buffer, qtable_index, dct2);\n"
    "    dct_decode(in_coeff, block_offset + 192, qtable_buffer, qtable_index, dct3);\n"
    "\t\n"
    "\tuint bx = block_x * 16;\n"
    "\tuint by = block_y * 16;\n"
    "\t\n"
    "\tblit_subblock(dct0, out_plane, blocks_wide * 16, blocks_high * 16, bx, by);\n"
    "\tblit_subblock(dct1, out_plane, blocks_wide * 16, blocks_high * 16, bx + 8, by);\n"
    "\tblit_subblock(dct2, out_plane, blocks_wide * 16, blocks_high * 16, bx, by + 8);\n"
    "\tblit_subblock(dct3, out_plane, blocks_wide * 16, blocks_high * 16, bx + 8, by + 8);\n"
    "}\n"
    "\n"
    "kernel void idct_delta(global ushort* in_coeff, global char2* in_mvec, global float* qtable_buffer, uint qtable_index, global uchar* prev_plane, global uchar* out_plane) {\n"
    "    uint block_x = get_global_id(0);\n"
    "    uint block_y = get_global_id(1);\n"
    "\n"
    "    uint blocks_wide = get_global_size(0);\n"
    "    uint blocks_high = get_global_size(1);\n"
    "\n"
    "    uint block_idx = block_x + (block_y * blocks_wide);\n"
    "    uint block_offset = block_idx * 256;\n"
    "\t\n"
    "\tchar2 mvec = in_mvec[block_idx];\n"
    "\n"
    "    local float dct0[64];\n"
    "    local float dct1[64];\n"
    "    local float dct2[64];\n"
    "    local float dct3[64];\n"
    "\n"
    "    dct_decode(in_coeff, block_offset, qtable_buffer, qtable_index, dct0);\n"
    "    dct_decode(in_coeff, block_offset + 64, qtable_buffer, qtable_index, dct1);\n"
    "    dct_decode(in_coeff, block_offset + 128, qtable_buffer, qtable_index, dct2);\n"
    "    dct_decode(in_coeff, block_offset + 192, qtable_buffer, qtable_index, dct3);\n"
    "\t\n"
    "\tuint bx = block_x * 16;\n"
    "\tuint by = block_y * 16;\n"
    "\t\n"
    "\tblit_subblock_delta(dct0, prev_plane, bx + mvec.x, by + mvec.y, out_plane, blocks_wide * 16, blocks_high * 16, bx, by);\n"
    "\tblit_subblock_delta(dct1, prev_plane, bx + mvec.x + 8, by + mvec.y, out_plane, blocks_wide * 16, blocks_high * 16, bx + 8, by);\n"
    "\tblit_subblock_delta(dct2, prev_plane, bx + mvec.x, by + mvec.y + 8, out_plane, blocks_wide * 16, blocks_high * 16, bx, by + 8);\n"
    "\tblit_subblock_delta(dct3, prev_plane, bx + mvec.x + 8, by + mvec.y + 8, out_plane, blocks_wide * 16, blocks_high * 16, bx + 8, by + 8);\n"
    "}";