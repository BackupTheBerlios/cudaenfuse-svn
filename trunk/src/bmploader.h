#ifndef BMPLOADER_H
#define BMPLOADER_H


#include <cuda_runtime.h>



extern "C" void LoadBMPFile(uchar4 **dst, int *width, int *height, const char *name);

extern "C" void SaveBMPFile(float4 **src, int width, int height, const char *name);

extern "C" void SaveGrayBMPFile(float **src, int width, int height, const char *name);

#endif

