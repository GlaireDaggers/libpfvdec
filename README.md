# libpfvdec

A fast & portable C library for decoding [PFV](https://github.com/GlaireDaggers/Pretty-Fast-Video) video streams.
Written against version 2.0.0 of the PFV codec.

## Dependencies

libpfvdec depends solely on the C runtime. It can be compiled with OpenMP support, but is not necessary (it will simply fall back to single-threaded mode)

The test executable depends on SDL2 (on Windows, use `-DSDL2_PATH` with cmake to set path to your SDL2 development libs)

## Usage

libpfvdec's API is inspired by the simplicity of [Theorafile](https://github.com/FNA-XNA/Theorafile).

To decode a PFV stream, construct an instance of PFV_Stream to wrap the underlying stream, create an instance of PFV_Decoder, and call
pfv_decoder_advance_delta to advance the decoder by delta time:

```c
void my_video_callback(void* userdata, uint8_t* buffer_y, uint8_t* buffer_u, uint8_t* buffer_v) {
    // handle video frames.
    // buffer_y, buffer_u, and buffer_v are pointers to buffers containing YUV plane data
}

PFV_Stream stream = open_stream("video.pfv");
PFV_Decoder* decoder = pfv_decoder_new(&stream, 8);

if (decoder == NULL) {
    printf("Invalid or corrupt PFV file\n");
    return -1;
}

uint16_t width, height, framerate;
pfv_decoder_get_video_params(decoder, &width, &height, &framerate);

printf("Opened decoder: %d x %d (%d FPS)\n", width, height, framerate);

// in your game loop:
pfv_decoder_advance_delta(decoder, deltatime, my_video_callback, NULL);
```

You can alternatively call pfv_decoder_next_frame to advance directly to the next frame regardless of delta time, and use pfv_decoder_get_framebuffer to retrieve pointers to the internal framebuffer (to update textures, etc)


## Building

libpfvdec uses CMake for building. To build the library by itself, enter the `lib/` directory, create & enter a new `build` folder, and invoke `cmake ..`
to generate the project files