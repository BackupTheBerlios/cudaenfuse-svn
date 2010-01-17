#ifndef FILTER_WRAPPERS_H
#define FILTER_WRAPPERS_H


extern "C++" cudaArray* uploadToArray(const uchar4 *source, int width, int height);
extern "C++" float* genGrayscale(const cudaArray *source, int width, int height);
extern "C++" float *genWeightmap(const cudaArray *image_source, const float *gray_source, int width, int height);
extern "C++" float4 *genGaussDown(float4 *source, int *width, int *height);
extern "C++" float4 *toFloat4(cudaArray *source, int width, int height);
extern "C++" float *genGaussDown_f(float *source, int *width, int *height);
extern "C++" float *genLinearDown_f(float *source, int *width, int *height);
extern "C++" float4 *genGaussUp(float4 *source, int *width, int *height);
extern "C++" float4 *genLaplace(float4 *source_down, float4 *source_up, int *width, int *height);
extern "C++" float4 *genWeightedLaplace(float4 *source_laplace, float *source_weight, int *width, int *height);
extern "C++" void normWeightmaps(float *map_1, float *map_2, int width, int height);
extern "C++" void normWeightmaps(float **maps, int numMaps, int width, int height);
extern "C++" float4 *genAdded(float4* source_1, float4 *source_2, int *width, int *height);
extern "C++" float *genAdded(float* source_1, float *source_2, int *width, int *height);
extern "C++" float4 *genCollapsed(float4 *src_big, float4 *src_small, int *width, int *height);

extern "C++" float4 *genSimpleEnfuse(float4 *target, cudaArray* source, float *weight, int width, int height);
extern "C++" float4 *genAppliedWeightmap(cudaArray* source, float *weight, int width, int height);

#endif
