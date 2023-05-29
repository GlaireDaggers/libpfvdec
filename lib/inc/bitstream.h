#pragma once

#include "pfv.h"

typedef struct {
    uint8_t *blob;
    size_t blob_len;
    size_t read_pos;
    uint64_t buffer;
    uint32_t num_bits;
    uint32_t read_bits;
} PFV_BitStream;

PFV_BitStream pfv_bitstream_new(uint8_t *blob, size_t blob_len);

uint32_t pfv_bitstream_read(PFV_BitStream *bitstream, uint32_t num_bits);
int32_t pfv_bitstream_read_signed(PFV_BitStream *bitstream, uint32_t num_bits);
void pfv_bitstream_put_back(PFV_BitStream* bitstream, uint32_t val, uint32_t num_bits);