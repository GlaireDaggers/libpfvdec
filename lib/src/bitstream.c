#include <assert.h>
#include <memory.h>

#include "bitstream.h"

PFV_BitStream pfv_bitstream_new(uint8_t* blob, size_t blob_len) {
	PFV_BitStream bitstream;
	bitstream.blob = blob;
	bitstream.blob_len = blob_len;
	bitstream.read_pos = 0;
	bitstream.buffer = 0;
	bitstream.num_bits = 0;
	bitstream.read_bits = 0;

	return bitstream;
}

uint32_t pfv_bitstream_read(PFV_BitStream* bitstream, uint32_t num_bits) {
	assert(num_bits <= 32 && num_bits > 0);

	if (bitstream->num_bits < num_bits) {
		size_t num_bytes = bitstream->blob_len - bitstream->read_pos;
		if (num_bytes > 4) {
			num_bytes = 4;
		}

		uint8_t buf[4] = { 0, 0, 0, 0 };
		memcpy(buf, bitstream->blob + bitstream->read_pos, num_bytes);
		bitstream->read_pos += num_bytes;

		for (int i = 0; i < num_bytes; i++) {
			bitstream->buffer |= (uint64_t)buf[i] << (bitstream->num_bits + (i * 8));
		}

		bitstream->num_bits += num_bytes * 8;
	}

	int actual_bits_read = bitstream->num_bits;
	if (actual_bits_read > num_bits) {
		actual_bits_read = num_bits;
	}

	assert(actual_bits_read > 0);

	uint64_t mask = (1 << actual_bits_read) - 1;
	uint64_t val = bitstream->buffer & mask;
	bitstream->buffer >>= actual_bits_read;
	bitstream->num_bits -= actual_bits_read;

	bitstream->read_bits += actual_bits_read;

	return (uint32_t)val;
}

int32_t pfv_bitstream_read_signed(PFV_BitStream* bitstream, uint32_t num_bits) {
	uint32_t shift = 32 - num_bits;
	return ((int32_t)pfv_bitstream_read(bitstream, num_bits) << shift) >> shift;
}

void pfv_bitstream_put_back(PFV_BitStream* bitstream, uint32_t val, uint32_t num_bits) {
	assert(bitstream->num_bits <= (64 - num_bits));

	uint32_t mask = (1 << num_bits) - 1;

	bitstream->buffer <<= num_bits;
	bitstream->buffer |= val & mask;
	bitstream->num_bits += num_bits;
	bitstream->read_bits -= num_bits;
}
