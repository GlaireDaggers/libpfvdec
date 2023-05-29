#pragma once

#include <stdint.h>

#include "bitstream.h"

typedef struct PFV_HuffmanNode PFV_HuffmanNode;

struct PFV_HuffmanNode {
    uint32_t freq;
    uint8_t ch;
    uint8_t has_ch;
    PFV_HuffmanNode *left;
    PFV_HuffmanNode *right;
};

typedef struct {
    uint32_t val;
    uint32_t len;
    uint8_t symbol;
} PFV_HuffmanCode;

typedef struct {
    PFV_HuffmanCode codes[16];
    uint8_t table[16];
    PFV_HuffmanCode dec_table[256];
    PFV_HuffmanNode tree_root;
} PFV_HuffmanTree;

void pfv_init_huffman_from_table(PFV_HuffmanTree *tree, uint8_t *table);
uint8_t pfv_huffman_read(PFV_HuffmanTree *tree, PFV_BitStream *bitstream);