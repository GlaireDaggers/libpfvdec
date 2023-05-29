#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include <assert.h>

#include "pfv.h"
#include "bitstream.h"
#include "huffman.h"
#include "dct.h"

#define RTASSERT(cond) if(!(cond)) { printf("ASSERTION FAILED:" #cond " (file: %s, line: %d)\n", __FILE__, __LINE__); abort(); }

typedef struct DeltaBlockHeader {
	int8_t mvec[2];
	uint8_t has_coeff;
} DeltaBlockHeader;

#define READ_U32(into, stream)                                                                         \
    {                                                                                                  \
        uint8_t buf[4];                                                                                \
        stream->read_fn(buf, 4, 1, stream->context);                                                   \
        into = buf[0] | ((uint32_t)buf[1] << 8) | ((uint32_t)buf[2] << 16) | ((uint32_t)buf[3] << 24); \
    }

#define READ_U16(into, stream)                       \
    {                                                \
        uint8_t buf[2];                              \
        stream->read_fn(buf, 2, 1, stream->context); \
        into = buf[0] | ((uint16_t)buf[1] << 8);     \
    }

#define READ_U8(into, stream)                         \
    {                                                 \
        stream->read_fn(&into, 1, 1, stream->context); \
    }

struct PFV_Decoder {
	PFV_Stream* stream;
	uint16_t width;
	uint16_t height;
	uint16_t framerate;
	uint8_t eof;
	float(*qtables)[64];
	uint16_t num_qtables;
	uint8_t* packet_buffer;
	size_t packet_buffer_len;
	PFV_HuffmanTree huffman_tree;
	size_t luma_pad_width;
	size_t luma_pad_height;
	size_t chroma_pad_width;
	size_t chroma_pad_height;
	size_t luma_blocks_wide;
	size_t luma_blocks_high;
	size_t chroma_blocks_wide;
	size_t chroma_blocks_high;
	int16_t* coeff_buffer_y;
	int16_t* coeff_buffer_u;
	int16_t* coeff_buffer_v;
	DeltaBlockHeader* header_buffer_y;
	DeltaBlockHeader* header_buffer_u;
	DeltaBlockHeader* header_buffer_v;
	uint8_t* plane_buffer_y;
	uint8_t* plane_buffer_u;
	uint8_t* plane_buffer_v;
	uint8_t* plane_buffer_y2;
	uint8_t* plane_buffer_u2;
	uint8_t* plane_buffer_v2;
	uint8_t* fb_y;
	uint8_t* fb_u;
	uint8_t* fb_v;
	uint64_t rewind_pos;
	int max_threads;
	double accum;
};

void read_block_headers(PFV_Decoder* decoder, PFV_BitStream* bitstream, DeltaBlockHeader* headers, int blocks_wide, int blocks_high) {
	int num_blocks = blocks_wide * blocks_high;

	// read headers from bitstream
	for (int i = 0; i < num_blocks; i++)
	{
		int has_mvec = pfv_bitstream_read(bitstream, 1);
		int has_coeff = pfv_bitstream_read(bitstream, 1);

		int8_t dx = 0, dy = 0;

		if (has_mvec) {
			dx = pfv_bitstream_read_signed(bitstream, 7);
			dy = pfv_bitstream_read_signed(bitstream, 7);
		}

		DeltaBlockHeader header;
		header.mvec[0] = dx;
		header.mvec[1] = dy;
		header.has_coeff = has_coeff;

		headers[i] = header;
	}
}

void read_plane_coefficients(PFV_Decoder* decoder, PFV_BitStream* bitstream, int16_t* target, int blocks_wide, int blocks_high) {
	int num_blocks = blocks_wide * blocks_high;
	int num_coeff = num_blocks * 256;

	int16_t block_coeff[256];

	// read coefficients from bitstream
	for (int i = 0; i < num_blocks; i++)
	{
		memset(block_coeff, 0, sizeof(int16_t) * 256);

		int out_idx = 0;
		while (out_idx < 256) {
			uint8_t num_zeroes = pfv_huffman_read(&decoder->huffman_tree, bitstream);
			uint8_t coeff_len = pfv_huffman_read(&decoder->huffman_tree, bitstream);

			out_idx += num_zeroes;

			if (coeff_len > 0) {
				RTASSERT(out_idx < 256);
				int16_t coeff = pfv_bitstream_read_signed(bitstream, coeff_len);
				block_coeff[out_idx++] = coeff;
			}
		}

		memcpy(&target[i * 256], block_coeff, sizeof(int16_t) * 256);
	}
}

void read_plane_delta_coefficients(PFV_Decoder* decoder, PFV_BitStream* bitstream, DeltaBlockHeader* headers, int16_t* target, int blocks_wide, int blocks_high) {
	int num_blocks = blocks_wide * blocks_high;
	int num_coeff = num_blocks * 256;

	int16_t block_coeff[256];

	// read coefficients from bitstream
	for (int i = 0; i < num_blocks; i++)
	{
		DeltaBlockHeader header = headers[i];

		if (header.has_coeff) {
			memset(block_coeff, 0, sizeof(int16_t) * 256);

			int out_idx = 0;
			while (out_idx < 256) {
				uint8_t num_zeroes = pfv_huffman_read(&decoder->huffman_tree, bitstream);
				uint8_t coeff_len = pfv_huffman_read(&decoder->huffman_tree, bitstream);

				out_idx += num_zeroes;

				if (coeff_len > 0) {
					RTASSERT(out_idx < 256);
					int16_t coeff = pfv_bitstream_read_signed(bitstream, coeff_len);
					block_coeff[out_idx++] = coeff;
				}
			}

			memcpy(&target[i * 256], block_coeff, sizeof(int16_t) * 256);
		}
		else {
			memset(&target[i * 256], 0, sizeof(int16_t) * 256);
		}
	}
}

void blit_subblock(float* src, uint8_t* dst, int dst_w, int dst_h, int dx, int dy) {
	for (int row = 0; row < 8; row++) {
		int dest_row = row + dy;
		int src_offset = row * 8;
		int dst_offset = (dest_row * dst_w) + dx;

		for (int column = 0; column < 8; column++) {
			float f = src[src_offset + column] + 128.0;

			if (f < 0.0) f = 0.0;
			else if (f > 255.0) f = 255.0;

			dst[dst_offset + column] = (uint8_t)f;
		}
	}
}

void blit_subblock_delta(float* src, uint8_t* prev, uint8_t* dst, int dst_w, int dst_h, int dx, int dy) {
	for (int row = 0; row < 8; row++) {
		int dest_row = row + dy;
		int src_offset = row * 8;
		int dst_offset = (dest_row * dst_w) + dx;

		for (int column = 0; column < 8; column++) {
			float f = (float)prev[src_offset + column] + (src[src_offset + column] * 2.f);

			if (f < 0.0) f = 0.0;
			else if (f > 255.0) f = 255.0;

			dst[dst_offset + column] = (uint8_t)f;
		}
	}
}

void copy_subblock(uint8_t* prev, uint8_t* dst, int dst_w, int dst_h, int dx, int dy) {
	//RTASSERT(dx >= 0 && dx <= dst_w - 16);
	//RTASSERT(dy >= 0 && dy <= dst_h - 16);

	for (int row = 0; row < 8; row++) {
		int dest_row = row + dy;
		int src_offset = row * 8;
		int dst_offset = (dest_row * dst_w) + dx;

		memcpy(dst + dst_offset, prev + src_offset, 8);
	}
}

void get_subblock(uint8_t* src, int src_w, int src_h, int sx, int sy, uint8_t* dst) {
	//RTASSERT(sx >= 0 && sx <= src_w - 16);
	//RTASSERT(sy >= 0 && sy <= src_h - 16);

	for (int row = 0; row < 8; row++) {
		int src_row = row + sy;
		int dst_offset = row * 8;
		int src_offset = (src_row * src_w) + sx;

		memcpy(dst + dst_offset, src + src_offset, 8);
	}
}

void decode_plane_dct(int max_threads, int16_t* src, float* qtable, uint8_t* dst, size_t blocks_wide, size_t blocks_high) {
	int num_blocks = blocks_wide * blocks_high;

	int dst_width = blocks_wide * 16;
	int dst_height = blocks_high * 16;

	int i;

	#pragma omp parallel for num_threads(max_threads)
	for (i = 0; i < num_blocks; i++) {
		float dct_buff[256];
		memset(dct_buff, 0, sizeof(float) * 256);

		float* dct0 = &dct_buff[0];
		float* dct1 = &dct_buff[64];
		float* dct2 = &dct_buff[128];
		float* dct3 = &dct_buff[192];

		int block_offset = i * 256;

		// grab each subblock
		int16_t* sb0 = &src[block_offset];
		int16_t* sb1 = &src[block_offset + 64];
		int16_t* sb2 = &src[block_offset + 128];
		int16_t* sb3 = &src[block_offset + 192];

		// decode each subblock & blit into destination
		dct_decode(sb0, qtable, dct0);
		dct_decode(sb1, qtable, dct1);
		dct_decode(sb2, qtable, dct2);
		dct_decode(sb3, qtable, dct3);

		int bx = (i % blocks_wide) * 16;
		int by = (i / blocks_wide) * 16;

		blit_subblock(dct0, dst, dst_width, dst_height, bx, by);
		blit_subblock(dct1, dst, dst_width, dst_height, bx + 8, by);
		blit_subblock(dct2, dst, dst_width, dst_height, bx, by + 8);
		blit_subblock(dct3, dst, dst_width, dst_height, bx + 8, by + 8);
	}
}

void copy_plane(uint8_t* src, uint8_t* dst, size_t src_width, size_t src_height, size_t dst_width, size_t dst_height) {
	for (int row = 0; row < dst_height; row++) {
		memcpy(dst + (row * dst_width), src + (row * src_width), dst_width);
	}
}

void decode_plane_delta_dct(int max_threads, int16_t* src, DeltaBlockHeader* headers, float* qtable, uint8_t* prev, uint8_t* dst, size_t blocks_wide, size_t blocks_high) {
	int num_blocks = blocks_wide * blocks_high;

	int dst_width = blocks_wide * 16;
	int dst_height = blocks_high * 16;

	int i;

	#pragma omp parallel for num_threads(max_threads)
	for (i = 0; i < num_blocks; i++) {
		float dct_buff[256];
		memset(dct_buff, 0, sizeof(float) * 256);

		uint8_t prev_buff[256];
		memset(prev_buff, 0, 256);

		float* dct0 = &dct_buff[0];
		float* dct1 = &dct_buff[64];
		float* dct2 = &dct_buff[128];
		float* dct3 = &dct_buff[192];

		uint8_t* prev0 = &prev_buff[0];
		uint8_t* prev1 = &prev_buff[64];
		uint8_t* prev2 = &prev_buff[128];
		uint8_t* prev3 = &prev_buff[192];

		DeltaBlockHeader header = headers[i];

		int bx = (i % blocks_wide) * 16;
		int by = (i / blocks_wide) * 16;

		get_subblock(prev, dst_width, dst_height, bx + (int)header.mvec[0], by + (int)header.mvec[1], prev0);
		get_subblock(prev, dst_width, dst_height, bx + (int)header.mvec[0] + 8, by + (int)header.mvec[1], prev1);
		get_subblock(prev, dst_width, dst_height, bx + (int)header.mvec[0], by + (int)header.mvec[1] + 8, prev2);
		get_subblock(prev, dst_width, dst_height, bx + (int)header.mvec[0] + 8, by + (int)header.mvec[1] + 8, prev3);

		if (header.has_coeff) {
			int block_offset = i * 256;

			// grab each subblock
			int16_t* sb0 = &src[block_offset];
			int16_t* sb1 = &src[block_offset + 64];
			int16_t* sb2 = &src[block_offset + 128];
			int16_t* sb3 = &src[block_offset + 192];

			// decode each subblock & blit into destination
			dct_decode(sb0, qtable, dct0);
			dct_decode(sb1, qtable, dct1);
			dct_decode(sb2, qtable, dct2);
			dct_decode(sb3, qtable, dct3);

			blit_subblock_delta(dct0, prev0, dst, dst_width, dst_height, bx, by);
			blit_subblock_delta(dct1, prev1, dst, dst_width, dst_height, bx + 8, by);
			blit_subblock_delta(dct2, prev2, dst, dst_width, dst_height, bx, by + 8);
			blit_subblock_delta(dct3, prev3, dst, dst_width, dst_height, bx + 8, by + 8);
		}
		else {
			copy_subblock(prev0, dst, dst_width, dst_height, bx, by);
			copy_subblock(prev1, dst, dst_width, dst_height, bx + 8, by);
			copy_subblock(prev2, dst, dst_width, dst_height, bx, by + 8);
			copy_subblock(prev3, dst, dst_width, dst_height, bx + 8, by + 8);
		}
	}
}

void decode_iframe(PFV_Decoder* decoder, uint8_t* payload, size_t payload_len) {
	PFV_BitStream bitstream = pfv_bitstream_new(payload, payload_len);
	size_t bitstream_length = payload_len * 8;

	// read symbol frequency table
	uint8_t symbol_table[16];

	for (int i = 0; i < 16; i++) {
		symbol_table[i] = (uint8_t)pfv_bitstream_read(&bitstream, 8);
	}

	// set up huffman table
	pfv_init_huffman_from_table(&decoder->huffman_tree, symbol_table);

	// fetch qtables
	uint8_t qtable_y = (uint8_t)pfv_bitstream_read(&bitstream, 8);
	uint8_t qtable_u = (uint8_t)pfv_bitstream_read(&bitstream, 8);
	uint8_t qtable_v = (uint8_t)pfv_bitstream_read(&bitstream, 8);

	// read plane coefficients from bitstream
	read_plane_coefficients(decoder, &bitstream, decoder->coeff_buffer_y, decoder->luma_blocks_wide, decoder->luma_blocks_high);
	read_plane_coefficients(decoder, &bitstream, decoder->coeff_buffer_u, decoder->chroma_blocks_wide, decoder->chroma_blocks_high);
	read_plane_coefficients(decoder, &bitstream, decoder->coeff_buffer_v, decoder->chroma_blocks_wide, decoder->chroma_blocks_high);

	// decode into YUV
	decode_plane_dct(decoder->max_threads, decoder->coeff_buffer_y, decoder->qtables[qtable_y], decoder->plane_buffer_y, decoder->luma_blocks_wide, decoder->luma_blocks_high);
	decode_plane_dct(decoder->max_threads, decoder->coeff_buffer_u, decoder->qtables[qtable_u], decoder->plane_buffer_u, decoder->chroma_blocks_wide, decoder->chroma_blocks_high);
	decode_plane_dct(decoder->max_threads, decoder->coeff_buffer_v, decoder->qtables[qtable_v], decoder->plane_buffer_v, decoder->chroma_blocks_wide, decoder->chroma_blocks_high);

	// copy into FB
	copy_plane(decoder->plane_buffer_y, decoder->fb_y, decoder->luma_pad_width, decoder->luma_pad_height, decoder->width, decoder->height);
	copy_plane(decoder->plane_buffer_u, decoder->fb_u, decoder->chroma_pad_width, decoder->chroma_pad_height, decoder->width / 2, decoder->height / 2);
	copy_plane(decoder->plane_buffer_v, decoder->fb_v, decoder->chroma_pad_width, decoder->chroma_pad_height, decoder->width / 2, decoder->height / 2);
}

void decode_pframe(PFV_Decoder* decoder, uint8_t* payload, size_t payload_len) {
	PFV_BitStream bitstream = pfv_bitstream_new(payload, payload_len);
	size_t bitstream_length = payload_len * 8;

	// read symbol frequency table
	uint8_t symbol_table[16];

	for (int i = 0; i < 16; i++) {
		symbol_table[i] = (uint8_t)pfv_bitstream_read(&bitstream, 8);
	}

	// set up huffman table
	pfv_init_huffman_from_table(&decoder->huffman_tree, symbol_table);

	// fetch qtables
	uint8_t qtable_y = (uint8_t)pfv_bitstream_read(&bitstream, 8);
	uint8_t qtable_u = (uint8_t)pfv_bitstream_read(&bitstream, 8);
	uint8_t qtable_v = (uint8_t)pfv_bitstream_read(&bitstream, 8);

	// read block headers
	read_block_headers(decoder, &bitstream, decoder->header_buffer_y, decoder->luma_blocks_wide, decoder->luma_blocks_high);
	read_block_headers(decoder, &bitstream, decoder->header_buffer_u, decoder->chroma_blocks_wide, decoder->chroma_blocks_high);
	read_block_headers(decoder, &bitstream, decoder->header_buffer_v, decoder->chroma_blocks_wide, decoder->chroma_blocks_high);

	// read plane coefficients from bitstream
	read_plane_delta_coefficients(decoder, &bitstream, decoder->header_buffer_y, decoder->coeff_buffer_y, decoder->luma_blocks_wide, decoder->luma_blocks_high);
	read_plane_delta_coefficients(decoder, &bitstream, decoder->header_buffer_u, decoder->coeff_buffer_u, decoder->chroma_blocks_wide, decoder->chroma_blocks_high);
	read_plane_delta_coefficients(decoder, &bitstream, decoder->header_buffer_v, decoder->coeff_buffer_v, decoder->chroma_blocks_wide, decoder->chroma_blocks_high);

	// decode into YUV
	decode_plane_delta_dct(decoder->max_threads, decoder->coeff_buffer_y, decoder->header_buffer_y, decoder->qtables[qtable_y], decoder->plane_buffer_y, decoder->plane_buffer_y2, decoder->luma_blocks_wide, decoder->luma_blocks_high);
	decode_plane_delta_dct(decoder->max_threads, decoder->coeff_buffer_u, decoder->header_buffer_u, decoder->qtables[qtable_u], decoder->plane_buffer_u, decoder->plane_buffer_u2, decoder->chroma_blocks_wide, decoder->chroma_blocks_high);
	decode_plane_delta_dct(decoder->max_threads, decoder->coeff_buffer_v, decoder->header_buffer_v, decoder->qtables[qtable_v], decoder->plane_buffer_v, decoder->plane_buffer_v2, decoder->chroma_blocks_wide, decoder->chroma_blocks_high);

	memcpy(decoder->plane_buffer_y, decoder->plane_buffer_y2, decoder->luma_blocks_wide * decoder->luma_blocks_high * 256);
	memcpy(decoder->plane_buffer_u, decoder->plane_buffer_u2, decoder->chroma_blocks_wide * decoder->chroma_blocks_high * 256);
	memcpy(decoder->plane_buffer_v, decoder->plane_buffer_v2, decoder->chroma_blocks_wide * decoder->chroma_blocks_high * 256);

	// copy into FB
	copy_plane(decoder->plane_buffer_y, decoder->fb_y, decoder->luma_pad_width, decoder->luma_pad_height, decoder->width, decoder->height);
	copy_plane(decoder->plane_buffer_u, decoder->fb_u, decoder->chroma_pad_width, decoder->chroma_pad_height, decoder->width / 2, decoder->height / 2);
	copy_plane(decoder->plane_buffer_v, decoder->fb_v, decoder->chroma_pad_width, decoder->chroma_pad_height, decoder->width / 2, decoder->height / 2);
}

void read_payload(PFV_Decoder* decoder, size_t payload_len) {
	if (decoder->packet_buffer_len < payload_len) {
		uint8_t* new_buffer = (uint8_t*)realloc(decoder->packet_buffer, payload_len);
		RTASSERT(new_buffer != NULL);

		decoder->packet_buffer = new_buffer;
		decoder->packet_buffer_len = payload_len;
	}

	decoder->stream->read_fn(decoder->packet_buffer, 1, payload_len, decoder->stream->context);
}

PFV_Decoder* pfv_decoder_new(PFV_Stream* stream, int max_threads) {
	// read header
	char magic[8];
	if (stream->read_fn(magic, 1, 8, stream->context) != 8) {
		printf("Failed reading PFV header\n");
		return NULL;
	}

	if (strncmp(magic, "PFVIDEO\0", 8)) {
		printf("Invalid PFV file\n");
		return NULL;
	}

	uint32_t version;
	READ_U32(version, stream);

	if (version != 200) {
		printf("Invalid PFV version\n");
		return NULL;
	}

	uint16_t width, height, framerate, num_qtable;

	READ_U16(width, stream);
	READ_U16(height, stream);
	READ_U16(framerate, stream);
	READ_U16(num_qtable, stream);

	float(*qtables)[64] = malloc(sizeof(*qtables) * num_qtable);

	if (qtables == NULL) {
		return NULL;
	}

	// read q-tables
	for (int i = 0; i < num_qtable; i++)
	{
		for (int x = 0; x < 64; x++)
		{
			uint16_t q;
			READ_U16(q, stream);
			qtables[i][x] = (float)q;
		}
	}

	PFV_Decoder* decoder = (PFV_Decoder*)malloc(sizeof(PFV_Decoder));

	if (decoder == NULL) {
		return NULL;
	}

	size_t chroma_width = width / 2;
	size_t chroma_height = height / 2;

	decoder->stream = stream;
	decoder->width = width;
	decoder->height = height;
	decoder->framerate = framerate;
	decoder->qtables = qtables;
	decoder->num_qtables = num_qtable;
	decoder->eof = false;
	decoder->packet_buffer = NULL;
	decoder->packet_buffer_len = 0;
	decoder->luma_pad_width = width + (16 - (width % 16)) % 16;
	decoder->luma_pad_height = height + (16 - (height % 16)) % 16;
	decoder->chroma_pad_width = chroma_width + (16 - (chroma_width % 16)) % 16;
	decoder->chroma_pad_height = chroma_height + (16 - (chroma_height % 16)) % 16;
	decoder->luma_blocks_wide = decoder->luma_pad_width / 16;
	decoder->luma_blocks_high = decoder->luma_pad_height / 16;
	decoder->chroma_blocks_wide = decoder->chroma_pad_width / 16;
	decoder->chroma_blocks_high = decoder->chroma_pad_height / 16;
	decoder->coeff_buffer_y = malloc(decoder->luma_blocks_wide * decoder->luma_blocks_high * 256 * sizeof(int16_t));
	decoder->coeff_buffer_u = malloc(decoder->chroma_blocks_wide * decoder->chroma_blocks_high * 256 * sizeof(int16_t));
	decoder->coeff_buffer_v = malloc(decoder->chroma_blocks_wide * decoder->chroma_blocks_high * 256 * sizeof(int16_t));
	decoder->header_buffer_y = malloc(decoder->luma_blocks_wide * decoder->luma_blocks_high * sizeof(DeltaBlockHeader));
	decoder->header_buffer_u = malloc(decoder->chroma_blocks_wide * decoder->chroma_blocks_high * sizeof(DeltaBlockHeader));
	decoder->header_buffer_v = malloc(decoder->chroma_blocks_wide * decoder->chroma_blocks_high * sizeof(DeltaBlockHeader));
	decoder->plane_buffer_y = malloc(decoder->luma_pad_width * decoder->luma_pad_height);
	decoder->plane_buffer_u = malloc(decoder->chroma_pad_width * decoder->chroma_pad_height);
	decoder->plane_buffer_v = malloc(decoder->chroma_pad_width * decoder->chroma_pad_height);
	decoder->plane_buffer_y2 = malloc(decoder->luma_pad_width * decoder->luma_pad_height);
	decoder->plane_buffer_u2 = malloc(decoder->chroma_pad_width * decoder->chroma_pad_height);
	decoder->plane_buffer_v2 = malloc(decoder->chroma_pad_width * decoder->chroma_pad_height);
	decoder->fb_y = malloc(width * height);
	decoder->fb_u = malloc((width / 2) * (height / 2));
	decoder->fb_v = malloc((width / 2) * (height / 2));
	decoder->rewind_pos = stream->tell_fn(stream->context);
	decoder->max_threads = max_threads;
	decoder->accum = 0.0;

	RTASSERT(decoder->coeff_buffer_y != NULL);
	RTASSERT(decoder->coeff_buffer_u != NULL);
	RTASSERT(decoder->coeff_buffer_v != NULL);
	RTASSERT(decoder->header_buffer_y != NULL);
	RTASSERT(decoder->header_buffer_u != NULL);
	RTASSERT(decoder->header_buffer_v != NULL);
	RTASSERT(decoder->plane_buffer_y != NULL);
	RTASSERT(decoder->plane_buffer_u != NULL);
	RTASSERT(decoder->plane_buffer_v != NULL);
	RTASSERT(decoder->plane_buffer_y2 != NULL);
	RTASSERT(decoder->plane_buffer_u2 != NULL);
	RTASSERT(decoder->plane_buffer_v2 != NULL);
	RTASSERT(decoder->fb_y != NULL);
	RTASSERT(decoder->fb_u != NULL);
	RTASSERT(decoder->fb_v != NULL);

	memset(&decoder->huffman_tree, 0, sizeof(PFV_HuffmanTree));
	memset(decoder->fb_y, 0, width * height);
	memset(decoder->fb_u, 128, (width / 2) * (height / 2));
	memset(decoder->fb_v, 128, (width / 2) * (height / 2));

	return decoder;
}

void pfv_decoder_destroy(PFV_Decoder* decoder) {
	decoder->stream->close_fn(decoder->stream->context);

	free(decoder->coeff_buffer_y);
	free(decoder->coeff_buffer_u);
	free(decoder->coeff_buffer_v);
	free(decoder->header_buffer_y);
	free(decoder->header_buffer_u);
	free(decoder->header_buffer_v);
	free(decoder->plane_buffer_y);
	free(decoder->plane_buffer_u);
	free(decoder->plane_buffer_v);
	free(decoder->plane_buffer_y2);
	free(decoder->plane_buffer_u2);
	free(decoder->plane_buffer_v2);
	free(decoder->fb_y);
	free(decoder->fb_u);
	free(decoder->fb_v);
	free(decoder->packet_buffer);
	free(decoder->qtables);
	free(decoder);
}

void pfv_decoder_get_video_params(PFV_Decoder* decoder, uint16_t* width, uint16_t* height, uint16_t* framerate) {
	*width = decoder->width;
	*height = decoder->height;
	*framerate = decoder->framerate;
}

uint8_t pfv_decoder_advance_delta(PFV_Decoder* decoder, double delta, void (*onvideo)(void*, uint8_t*, uint8_t*, uint8_t*), void* userdata) {
	double frame_delta = 1.0 / decoder->framerate;
	decoder->accum += delta;

	while (decoder->accum >= frame_delta) {
		if (!pfv_decoder_next_frame(decoder)) {
			return false;
		}

		decoder->accum -= frame_delta;
	}

	if (onvideo != NULL) {
		onvideo(userdata, decoder->fb_y, decoder->fb_u, decoder->fb_v);
	}

	return true;
}

uint8_t pfv_decoder_next_frame(PFV_Decoder* decoder) {
	uint8_t found_frame = false;

	while (!found_frame && !decoder->eof) {
		uint8_t packet_type = 0;
		READ_U8(packet_type, decoder->stream)

		uint32_t packet_len = 0;
		READ_U32(packet_len, decoder->stream);

		switch (packet_type) {
		case 0:
			// EOF
			decoder->eof = true;
			break;
		case 1:
			// iframe or drop frame
			found_frame = true;
			if (packet_len > 0) {
				read_payload(decoder, (size_t)packet_len);
				decode_iframe(decoder, decoder->packet_buffer, decoder->packet_buffer_len);
			}
			break;
		case 2:
			// pframe
			found_frame = true;
			if (packet_len > 0) {
				read_payload(decoder, (size_t)packet_len);
				decode_pframe(decoder, decoder->packet_buffer, decoder->packet_buffer_len);
			}
			break;
		default:
			// unknown
			decoder->stream->seek_fn(decoder->stream->context, (uint64_t)packet_len, PFV_SEEK_CUR);
			break;
		}
	}

	return !decoder->eof;
}

void pfv_decoder_get_framebuffer(PFV_Decoder* decoder, uint8_t** fb_y, uint8_t** fb_u, uint8_t** fb_v) {
	*fb_y = decoder->fb_y;
	*fb_u = decoder->fb_u;
	*fb_v = decoder->fb_v;
}

void pfv_decoder_reset(PFV_Decoder* decoder) {
	size_t width = decoder->width;
	size_t height = decoder->height;

	memset(decoder->fb_y, 0, width * height);
	memset(decoder->fb_u, 128, (width / 2) * (height / 2));
	memset(decoder->fb_v, 128, (width / 2) * (height / 2));

	decoder->stream->seek_fn(decoder->stream->context, decoder->rewind_pos, PFV_SEEK_SET);
	decoder->eof = false;
	decoder->accum = 0.0;
}
