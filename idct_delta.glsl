layout(std140, binding = 0) uniform ParamUBO {
	uint qtable_index;
} ubo;

layout(std140, binding = 1) readonly buffer coeff_buffer {
	int coeff[];
};

layout(std140, binding = 2) readonly buffer mvec_buffer {
	ivec2 mvec[];
};

layout(std140, binding = 4) readonly buffer qtable_buffer {
	float qtable[];
};

uniform layout(binding = 5, r8) readonly image2d prev_plane;
uniform layout(binding = 6, r8) writeonly image2d out_plane;

#define S_1 0.254897789552079584470970
#define S_2 0.270598050073098492199862
#define S_3 0.300672443467522640271861
#define S_0 0.353553390593273762200422
#define S_4 0.353553390593273762200422
#define S_5 0.449988111568207852319255
#define S_6 0.653281482438188263928322
#define S_7 1.281457723870753089398043

#define RS_0 (1.0f / S_0)
#define RS_1 (1.0f / S_1)
#define RS_2 (1.0f / S_2)
#define RS_3 (1.0f / S_3)
#define RS_4 (1.0f / S_4)
#define RS_5 (1.0f / S_5)
#define RS_6 (1.0f / S_6)
#define RS_7 (1.0f / S_7)

#define A_1 0.707106781186547524400844
#define A_2 0.541196100146196984399723
#define A_3 0.707106781186547524400844
#define A_4 1.306562964876376527856643
#define A_5 0.382683432365089771728460

#define RA_1 (1.0f / A_1)
#define RA_2 (1.0f / A_2)
#define RA_3 (1.0f / A_3)
#define RA_4 (1.0f / A_4)
#define RA_5 (1.0f / A_5)

#define F (1.0f / (A_2 * A_5 - A_2 * A_4 - A_4 * A_5))

const uint INV_ZIGZAG_TABLE[64] = {
    0, 1, 5, 6, 14, 15, 27, 28, 2, 4, 7, 13, 16, 26, 29, 42, 3, 8, 12, 17, 25, 30, 41, 43, 9, 11, 18, 24, 31, 40, 44, 53, 10, 19, 23, 32, 39, 45, 52, 54, 20, 22,
    33, 38, 46, 51, 55, 60, 21, 34, 37, 47, 50, 56, 59, 61, 35, 36, 48, 49, 57, 58, 62, 63
};

void fast_dct8_inverse_transform(inout vector[64], int offset, int stride) {
	float v15 = vector[offset + (0 * stride)] * RS_0;
    float v26 = vector[offset + (1 * stride)] * RS_1;
    float v21 = vector[offset + (2 * stride)] * RS_2;
    float v28 = vector[offset + (3 * stride)] * RS_3;
    float v16 = vector[offset + (4 * stride)] * RS_4;
    float v25 = vector[offset + (5 * stride)] * RS_5;
    float v22 = vector[offset + (6 * stride)] * RS_6;
    float v27 = vector[offset + (7 * stride)] * RS_7;
    
    float v19 = (v25 - v28) * 0.5;
    float v20 = (v26 - v27) * 0.5;
    float v23 = (v26 + v27) * 0.5;
    float v24 = (v25 + v28) * 0.5;
    
    float v7  = (v23 + v24) * 0.5;
    float v11 = (v21 + v22) * 0.5;
    float v13 = (v23 - v24) * 0.5;
    float v17 = (v21 - v22) * 0.5;
    
    float v8 = (v15 + v16) * 0.5;
    float v9 = (v15 - v16) * 0.5;
    
    float v18 = (v19 - v20) * A_5;  // Different from original
    float v12 = (v19 * A_4 - v18) * F;
    float v14 = (v18 - v20 * A_2) * F;
    
    float v6 = v14 - v7;
    float v5 = v13 * RA_3 - v6;
    float v4 = -v5 - v12;
    float v10 = v17 * RA_1 - v11;
    
    float v0 = (v8 + v11) * 0.5;
    float v1 = (v9 + v10) * 0.5;
    float v2 = (v9 - v10) * 0.5;
    float v3 = (v8 - v11) * 0.5;
    
    vector[offset + (0 * stride)] = (v0 + v7) * 0.5;
    vector[offset + (1 * stride)] = (v1 + v6) * 0.5;
    vector[offset + (2 * stride)] = (v2 + v5) * 0.5;
    vector[offset + (3 * stride)] = (v3 + v4) * 0.5;
    vector[offset + (4 * stride)] = (v3 - v4) * 0.5;
    vector[offset + (5 * stride)] = (v2 - v5) * 0.5;
    vector[offset + (6 * stride)] = (v1 - v6) * 0.5;
    vector[offset + (7 * stride)] = (v0 - v7) * 0.5;
}

void dct_decode(uint src_offset, uint qtable_index, inout float result[64]) {
    for (int i = 0; i < 64; i++) {
		float n = convert_float(coeff[src_offset + INV_ZIGZAG_TABLE[i]]);
		float d = qtable[i + (qtable_index * 64)];
		
		result[i] = n * d;
	}
	
	// inverse DCT columns
	for (int i = 0; i < 8; i++) {
		fast_dct8_inverse_transform(result, i, 8);
	}
	
	// inverse DCT rows
	for (int i = 0; i < 8; i++) {
		fast_dct8_inverse_transform(result, i * 8, 1);
	}
}

void blit_subblock(float dct[64], uint sx, uint sy, uint dx, uint dy) {
	for (int row = 0; row < 8; row++) {
		int src_offset = row * 8;

		for (int column = 0; column < 8; column++) {
			float f = dct[src_offset + column] * 2.0;
            float prev = imageLoad(prev_plane, uint2(sx + column, sy + row));
			imageStore(out_plane, uint2(dx + column, dy + row), clamp(f + prev, 0.0, 255.0));
		}
	}
}

void main() {
	uint block_x = gl_GlobalInvocationID.x;
    uint block_y = gl_GlobalInvocationID.y;

    uint blocks_wide = gl_NumWorkGroups.x;
    uint blocks_high = gl_NumWorkGroups.y;

    uint block_idx = block_x + (block_y * blocks_wide);
    uint block_offset = block_idx * 256;

    float dct0[64];
    float dct1[64];
    float dct2[64];
    float dct3[64];

    dct_decode(block_offset, qtable_index, dct0);
    dct_decode(block_offset + 64, qtable_index, dct1);
    dct_decode(block_offset + 128, qtable_index, dct2);
    dct_decode(block_offset + 192, qtable_index, dct3);
	
	uint bx = block_x * 16;
	uint by = block_y * 16;
	
    ivec2 mv = mvec[block_idx];

	blit_subblock(dct0, bx + mv.x, by + mv.y, bx, by);
	blit_subblock(dct1, bx + mv.x + 8, by + mv.y, bx + 8, by);
	blit_subblock(dct2, bx + mv.x, by + mv.y + 8, bx, by + 8);
	blit_subblock(dct3, bx + mv.x + 8, by + mv.y + 8, bx + 8, by + 8);
}