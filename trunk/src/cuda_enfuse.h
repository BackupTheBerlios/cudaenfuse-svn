#ifndef CUDA_ENFUSE_H
#define CUDA_ENFUSE_H





// device emulation flag
#ifdef __DEVICE_EMULATION__
    const bool EMU = true;
#else
    const bool EMU = false;
#endif




// color management
typedef unsigned int TColor;
__device__ TColor make_color(float r, float g, float b, float a) {
    return
        ((int)(a * 255.0f) << 24) |
        ((int)(b * 255.0f) << 16) |
        ((int)(g * 255.0f) <<  8) |
        ((int)(r * 255.0f) <<  0);
}


extern "C++" float4 *cuda_enfuse(uchar4 **imageInMems, int imageInNum, int width, int height);
extern "C++" float4 *cuda_enfuse(uchar4 **imageInMems,  int imageInNum, int width, int height, int scaleDepth);

#endif

