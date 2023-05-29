#pragma once

#include <stdint.h>

#define PFV_SEEK_SET 0
#define PFV_SEEK_CUR 1
#define PFV_SEEK_END 2

/// <summary>
/// Represents an abstract stream which a PFV_Decoder can read from
/// </summary>
typedef struct
{
    void *context;
    size_t (*read_fn)(void *ptr, size_t elem_size, size_t elem_count, void *context);
    uint64_t (*tell_fn)(void *context);
    int32_t (*seek_fn)(void *context, uint64_t offset, int32_t whence);
    void (*close_fn)(void *context);
} PFV_Stream;

/// <summary>
/// Instance of a decoder which can read PFV streams
/// </summary>
typedef struct PFV_Decoder PFV_Decoder;

/// <summary>
/// Creates a new PFV decoder
/// </summary>
/// <param name="stream">The stream to decode from. The decoder will maintain ownership of the stream</param>
/// <param name="max_threads">The maximum number of threads to use (ignored if OpenMP is not available)</param>
/// <returns>A new decoder, or NULL if the decoder could not be created (stream contains invalid data, etc)</returns>
PFV_Decoder *pfv_decoder_new(PFV_Stream *stream, int max_threads);

/// <summary>
/// Destroy a decoder previously returned from pfv_decoder_new() and close the underlying stream
/// </summary>
/// <param name="decoder">The decoder to destroy</param>
void pfv_decoder_destroy(PFV_Decoder *decoder);

/// <summary>
/// Retrieve the video params from a PFV decoder
/// </summary>
/// <param name="decoder">The decoder instance</param>
/// <param name="width">The width of each frame in pixels</param>
/// <param name="height">The height of each frame in pixels</param>
/// <param name="framerate">The video framerate</param>
void pfv_decoder_get_video_params(PFV_Decoder *decoder, uint16_t *width, uint16_t *height, uint16_t *framerate);

/// <summary>
/// Read the next video frame from the stream
/// </summary>
/// <param name="decoder">The decoder instance</param>
/// <returns>1 if there is more data in the stream, 0 if the decoder has reached an end of stream marker</returns>
uint8_t pfv_decoder_next_frame(PFV_Decoder *decoder);

/// <summary>
/// Advance the decoder by delta time, triggering a callback for each decoded frame
/// </summary>
/// <param name="decoder">The decoder instance</param>
/// <param name="delta">The delta time in seconds to advance by</param>
/// <param name="onvideo">Callback to be invoked for each decoded frame</param>
/// <param name="userdata">Userdata to pass as first argument to callback</param>
/// <returns>1 if there is more data in the stream, 0 if the decoder has reached an end of stream marker</returns>
uint8_t pfv_decoder_advance_delta(PFV_Decoder* decoder, double delta, void (*onvideo)(void*, uint8_t*, uint8_t*, uint8_t*), void* userdata);

/// <summary>
/// Retrieve pointers to the PFV decoder's framebuffers
/// </summary>
/// <param name="decoder">The decoder instance</param>
/// <param name="fb_y">Output Y plane framebuffer pointer</param>
/// <param name="fb_u">Output U plane framebuffer pointer</param>
/// <param name="fb_v">Output V plane framebuffer pointer</param>
void pfv_decoder_get_framebuffer(PFV_Decoder* decoder, uint8_t** fb_y, uint8_t** fb_u, uint8_t** fb_v);

/// <summary>
/// Rewind a decoder back to the start of the PFV stream
/// </summary>
/// <param name="decoder">The decoder instance</param>
void pfv_decoder_reset(PFV_Decoder* decoder);
