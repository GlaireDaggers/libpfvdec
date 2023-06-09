#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <memory.h>
#include <stdbool.h>

#include "pfv.h"

#include "SDL.h"

// #define SPEED_TEST

size_t fread_fn(void* ptr, size_t elem_size, size_t elem_count, void* context) {
	return fread(ptr, elem_size, elem_count, (FILE*)context);
}

uint64_t ftell_fn(void* context) {
	return (uint64_t)ftell((FILE*)context);
}

int32_t fseek_fn(void* context, uint64_t offset, int32_t whence) {
	return (int32_t)fseek((FILE*)context, (long)offset, (int)whence);
}

void fclose_fn(void* context) {
	fclose((FILE*)context);
}

PFV_Stream open_stream(const char* filename) {
	FILE* handle = fopen(filename, "rb");
	setvbuf(handle, NULL, _IOFBF, 65536);

	PFV_Stream stream;
	stream.context = handle;
	stream.read_fn = fread_fn;
	stream.tell_fn = ftell_fn;
	stream.seek_fn = fseek_fn;
	stream.close_fn = fclose_fn;

	return stream;
}

int decoded_frame = false;
int framecount = 0;

void onvideo(void* userdata, uint8_t* fb_y, uint8_t* fb_u, uint8_t* fb_v) {
	SDL_Texture* tex = (SDL_Texture*)userdata;
	uint32_t format;
	int access, width, height;
	SDL_QueryTexture(tex, &format, &access, &width, &height);
	SDL_UpdateYUVTexture((SDL_Texture*)userdata, NULL, fb_y, width, fb_u, width / 2, fb_v, width / 2);
	decoded_frame = true;
	framecount++;
}

int main() {
	// open PFV file

	PFV_Stream stream = open_stream("balcony2.pfv");

	if (stream.context == NULL) {
		printf("Failed opening input file\n");
		return -1;
	}

	PFV_Decoder* decoder = pfv_decoder_new(&stream, 8);

	if (decoder == NULL) {
		printf("Invalid or corrupt PFV file\n");
		return -1;
	}

	uint16_t width, height, framerate;
	pfv_decoder_get_video_params(decoder, &width, &height, &framerate);

	printf("Opened decoder: %d x %d (%d FPS)\n", width, height, framerate);

#ifdef SPEED_TEST
	for (int run = 0; run < 50; run++) {
		printf("RUN %d\n", run);
		pfv_decoder_reset(decoder);

		clock_t start = clock();

		uint64_t prev_time = SDL_GetPerformanceCounter();

		double sum = 0.0;

		int frame_count = 0;
		while (pfv_decoder_next_frame(decoder)) {
			uint64_t cur_time = SDL_GetPerformanceCounter();
			sum += (double)(cur_time - prev_time) / SDL_GetPerformanceFrequency();
			prev_time = cur_time;

			frame_count++;
		}

		clock_t end = clock();

		double duration = ((end - start) / (double)CLOCKS_PER_SEC) * 1000.0;

		printf("Decoded %d frames in %d ms\n", frame_count, (int)duration);
		printf("Avg time per frame: %f ms\n", (sum / frame_count) * 1000.0);
	}
#else
	// init SDL2

	if (SDL_Init(SDL_INIT_VIDEO) < 0) {
		printf("Failed initializing SDL\n");
		pfv_decoder_destroy(decoder);
		return -1;
	}

	SDL_Window* window = SDL_CreateWindow("libpfvdec test", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, width, height, SDL_WINDOW_OPENGL);

	if (window == NULL) {
		printf("Failed creating SDL window\n");
		pfv_decoder_destroy(decoder);
		return -1;
	}

	SDL_Renderer* renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);

	if (renderer == NULL) {
		printf("Failed creating renderer\n");
		pfv_decoder_destroy(decoder);
		return -1;
	}

	// create YUV surface
	SDL_Texture* vid_texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_YV12, SDL_TEXTUREACCESS_STREAMING, width, height);

	uint64_t prev_time = SDL_GetPerformanceCounter();

	double frame_timer_sum = 0.0;

	uint8_t window_open = true;
	while (window_open) {
		SDL_Event e;
		while (SDL_PollEvent(&e)) {
			switch (e.type) {
			case SDL_QUIT:
				window_open = false;
				break;
			}
		}

		uint64_t cur_time = SDL_GetPerformanceCounter();
		uint64_t delta = cur_time - prev_time;
		double dt = (double)delta / SDL_GetPerformanceFrequency();

		uint64_t frame_start = SDL_GetPerformanceCounter();
		decoded_frame = false;

		if (!pfv_decoder_advance_delta(decoder, dt, onvideo, vid_texture)) {
			pfv_decoder_reset(decoder);
		}

		uint64_t frame_end = SDL_GetPerformanceCounter();

		if (decoded_frame) {
			double fd = (double)(frame_end - frame_start) / SDL_GetPerformanceFrequency();
			frame_timer_sum += fd;

			if (framecount == 60) {
				printf("Avg decoding time: %f ms\n", (frame_timer_sum * 1000.0) / framecount);
				framecount = 0;
				frame_timer_sum = 0.0;
			}
		}

		prev_time = cur_time;

		SDL_RenderClear(renderer);
		SDL_RenderCopy(renderer, vid_texture, NULL, NULL);
		SDL_RenderPresent(renderer);
	}
#endif

	pfv_decoder_destroy(decoder);
	return 0;
}