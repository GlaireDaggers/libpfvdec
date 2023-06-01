#include <stdlib.h>
#include <memory.h>
#include <assert.h>
#include <stdbool.h>

#include "huffman.h"

#define RTASSERT(cond) if(!(cond)) { printf("ASSERTION FAILED:" #cond " (file: %s, line: %d)\n", __FILE__, __LINE__); abort(); }

#define CODEMASK(code) ((1 << code.len) - 1)
#define APPEND_BIT(code, bit) { code.val |= bit << code.len; code.len++; }

void insert_node_at(PFV_HuffmanNode node, PFV_HuffmanNode* table, size_t* table_len, int index) {
	assert(*table_len < 16);

	for (size_t i = *table_len; i > index; i--) {
		table[i] = table[i - 1];
	}

	table[index] = node;
	*table_len += 1;
}

int get_insert_index(PFV_HuffmanNode* node, PFV_HuffmanNode* table, size_t table_len) {
	for (int i = 0; i < table_len; i++) {
		if (node->freq > table[i].freq) {
			return i;
		}
	}

	return table_len;
}

PFV_HuffmanNode pop_node(PFV_HuffmanNode* table, size_t* table_len) {
	assert(*table_len > 0);

	*table_len -= 1;
	return table[*table_len];
}

void assign_codes(PFV_HuffmanNode* node, PFV_HuffmanCode* codes, PFV_HuffmanCode code) {
	if (node->has_ch) {
		code.symbol = node->ch;
		codes[node->ch] = code;
	}
	else {
		if (node->left != NULL) {
			PFV_HuffmanCode code2 = code;
			APPEND_BIT(code2, 0);
			assign_codes(node->left, codes, code2);
		}

		if (node->right != NULL) {
			PFV_HuffmanCode code2 = code;
			APPEND_BIT(code2, 1);
			assign_codes(node->right, codes, code2);
		}
	}
}

void destroy_node(PFV_HuffmanNode* node) {
	if (node->left != NULL) {
		destroy_node(node->left);
	}

	if (node->right != NULL) {
		destroy_node(node->right);
	}

	free(node);
}

void pfv_init_huffman_from_table(PFV_HuffmanTree* tree, uint8_t* table) {
	if (tree->tree_root.left != NULL) {
		destroy_node(tree->tree_root.left);
	}

	if (tree->tree_root.right != NULL) {
		destroy_node(tree->tree_root.right);
	}

	PFV_HuffmanNode p[16];
	memset(p, 0, sizeof(p));

	size_t p_count = 0;


	for (int i = 0; i < 16; i++) {
		if (table[i] > 0) {
			// insertion sort new node
			PFV_HuffmanNode node;
			node.freq = table[i];
			node.ch = i;
			node.has_ch = true;
			node.left = NULL;
			node.right = NULL;

			int index = get_insert_index(&node, p, p_count);
			insert_node_at(node, p, &p_count, index);
		}
	}

	// tree has no codes, just exit
	if (p_count == 0) {
		memset(tree, 0, sizeof(PFV_HuffmanTree));
		return;
	}

	while (p_count > 1) {
		PFV_HuffmanNode a = pop_node(p, &p_count);
		PFV_HuffmanNode b = pop_node(p, &p_count);

		PFV_HuffmanNode c;
		c.freq = a.freq + b.freq;
		c.left = malloc(sizeof(PFV_HuffmanNode));
		c.right = malloc(sizeof(PFV_HuffmanNode));
		c.ch = 0;
		c.has_ch = false;

		memcpy(c.left, &a, sizeof(PFV_HuffmanNode));
		memcpy(c.right, &b, sizeof(PFV_HuffmanNode));

		int index = get_insert_index(&c, p, p_count);
		insert_node_at(c, p, &p_count, index);
	}

	RTASSERT(p_count > 0);

	tree->tree_root = p[0];

	PFV_HuffmanCode code;
	memset(&code, 0, sizeof(PFV_HuffmanCode));
	memset(&tree->codes, 0, sizeof(PFV_HuffmanCode) * 16);

	assign_codes(&tree->tree_root, tree->codes, code);

	// generate pre-masked decoder table for codes of 8 bits or less - allows us to just read in a whole u8 and index into this table to get a code
	// if a code is longer than 8 bits, it falls back to the slow tree traversal path

	memset(tree->dec_table, 0, sizeof(PFV_HuffmanCode) * 256);

	for (int val = 0; val < 256; val++) {
		for (int c = 0; c < 16; c++) {
			PFV_HuffmanCode code = tree->codes[c];

			if (code.len > 0 && code.len <= 8 &&
				(val & CODEMASK(code)) == code.val) {
				tree->dec_table[val] = code;
				break;
			}
		}
	}
}

uint8_t pfv_huffman_read_slow(PFV_HuffmanTree* tree, PFV_BitStream* bitstream) {
	PFV_HuffmanNode* node = &tree->tree_root;

	while (!node->has_ch) {
		uint32_t bit = pfv_bitstream_read(bitstream, 1);

		if (bit) {
			RTASSERT(node->right != NULL);
			node = node->right;
		}
		else {
			RTASSERT(node->left != NULL);
			node = node->left;
		}
	}

	return node->ch;
}

uint8_t pfv_huffman_read(PFV_HuffmanTree* tree, PFV_BitStream* bitstream) {
	uint8_t cur = pfv_bitstream_read(bitstream, 8);
	PFV_HuffmanCode code = tree->dec_table[cur];

	if (code.len == 0) {
		// code wasn't found in lookup table, put back bits & fall back to slow path
		pfv_bitstream_put_back(bitstream, cur, 8);
		return pfv_huffman_read_slow(tree, bitstream);
	}
	else {
		// put back the bits we didn't use & return value
		uint8_t put_back = cur >> code.len;
		pfv_bitstream_put_back(bitstream, put_back, 8 - code.len);

		return code.symbol;
	}
}
