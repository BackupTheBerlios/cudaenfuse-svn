#include <stdio.h>
#include <stdlib.h>

#include <cutil.h>

#include "bmploader.h"

#include "cuda_wrappers.h"







void cuda_init(int argc, char** argv) {
	CUT_DEVICE_INIT(argc, argv);
}

void cuda_exit(int argc, char** argv) {
    CUT_EXIT(argc, argv); 
}


void printCudaError(FILE* target, cudaError_t cudaErrno, char* msg) {
	if (cudaErrno > 0) {
		fprintf(target, "ERROR (%i) %s: %s\n",cudaErrno, msg, cudaGetErrorString(cudaErrno));
	}
}

void printCudaError(FILE* target, cudaError_t cudaErrno) {
    printCudaError(target, cudaErrno, "");
}





cudaArray *cuda_mallocArray(cudaChannelFormatDesc *channelDesc, int width, int height) {
    cudaArray* target;
    CUDA_SAFE_CALL(cudaMallocArray(&target, channelDesc, width, height));
    printCudaError(stderr, cudaGetLastError(), "cudaMallocArray");
    return target;
}

template<typename dataType> dataType *cuda_malloc(int width, int height) {
    dataType *target;
    CUDA_SAFE_CALL(cudaMalloc((void**) &target, width * height * sizeof(dataType)));
    printCudaError(stderr, cudaGetLastError(), "cudaMalloc");
    return target;
}
template float *cuda_malloc<float>(int width, int height); // pre-init <float>

void cuda_bindTexture(cudaChannelFormatDesc *channelDesc, const cudaArray* source, const char* texRefName) {
    const struct textureReference *texRef;
    cudaGetTextureReference(&texRef, texRefName);
    printCudaError(stderr, cudaGetLastError(), "cudaGetTextureReference");    
    
    CUDA_SAFE_CALL(cudaBindTextureToArray(texRef, source, channelDesc));
    printCudaError(stderr, cudaGetLastError(), "cudaBindTextureToArray");    
}


void cuda_unbindTexture(const char *texRefName) {
    const struct textureReference *texRef;
    cudaGetTextureReference(&texRef, texRefName);
    
    CUDA_SAFE_CALL(cudaUnbindTexture(texRef));
    printCudaError(stderr, cudaGetLastError(), "cudaUnbindTexture");
}

template<typename dataType> dataType *cuda_memcpy(dataType* source, int width, int height) {
    float *target = cuda_malloc<float>(width, height);

    int memSize = width * height * sizeof(dataType);
    CUDA_SAFE_CALL(cudaMemcpy(target, (void*)source, memSize, cudaMemcpyDeviceToDevice));
    printCudaError(stderr, cudaGetLastError(), "cudaMemcpy");
    
    return target;
}
template float *cuda_memcpy<float>(float *source, int width, int height); // pre-init <float>




cudaArray *cuda_uploadToArray(cudaArray* target, const void* source, int memSize) {
    CUDA_SAFE_CALL(cudaMemcpyToArray(target, 0, 0, (void*)source, memSize, cudaMemcpyHostToDevice));
    printCudaError(stderr, cudaGetLastError(), "cudaMemcpyToArray");
    return target;
}

cudaArray *cuda_uploadToArray(cudaChannelFormatDesc *channelDesc, int width, int height, const void *source) {
    cudaArray* target = cuda_mallocArray(channelDesc, width, height);
    
    int memSize = width * height * sizeof(uchar4);
    cuda_uploadToArray(target, source, memSize);
    return target;
}


template<typename T> T *cuda_download(const T* source, int width, int height) {
    int memSize = width * height * sizeof(T);

    T* target = (T*)malloc(memSize);
    
    CUDA_SAFE_CALL(cudaMemcpy(target, (void*)source, memSize, cudaMemcpyDeviceToHost));
    printCudaError(stderr, cudaGetLastError(), "cudaMemcpy");
    
    return target;    
}
template float4 *cuda_download(const float4* source, int width, int height); // pre-init <float4>
template float3 *cuda_download(const float3* source, int width, int height); // pre-init <float3>




template<typename dataType> void cuda_bindTexture(const dataType* source, int width, int height, const char* texRefName) {
    const struct textureReference *texRef;
    cudaGetTextureReference(&texRef, texRefName);
    printCudaError(stderr, cudaGetLastError(), "cudaGetTextureReference");    
    
    int memSize = width * height * sizeof(dataType);
    const cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<dataType>();
    
    CUDA_SAFE_CALL(cudaBindTexture(0, texRef, (void*)source, &channelDesc, (size_t)memSize));
    printCudaError(stderr, cudaGetLastError(), "cudaBindTexture");
}
template void cuda_bindTexture<float>(const float *source, int width, int height, const char* texRefName); // pre-init <float>



/*
 * Creates two events - start and stop - for timing purposes and 
 * records the time of the start event.
 */
void cuda_startTimer(cudaEvent_t *start, cudaEvent_t *end) {
    // create start and end events
    CUDA_SAFE_CALL(cudaEventCreate(start));
    printCudaError(stderr, cudaGetLastError(), "cudaEventCreate");
    CUDA_SAFE_CALL(cudaEventCreate(end));
    printCudaError(stderr, cudaGetLastError(), "cudaEventCreate");
    
    // register time of the start event
    CUDA_SAFE_CALL(cudaEventRecord(*start, 0));
    printCudaError(stderr, cudaGetLastError(), "cudaEventRecord");
}


/*
 * Records the time of the end event and writes the time elapsed between
 * the start and end event to the time variable.
 * The start and end event are both destroyed afterwards. 
 */
void cuda_stopTimer(cudaEvent_t *start, cudaEvent_t *end, float *time) {
    // wait for all threads to finish
    CUDA_SAFE_CALL(cudaThreadSynchronize());
    printCudaError(stderr, cudaGetLastError(), "cudaThreadSynchronize");
    
    // record time of the end event
    CUDA_SAFE_CALL(cudaEventRecord(*end, 0));
    printCudaError(stderr, cudaGetLastError(), "cudaEventRecord");
    CUDA_SAFE_CALL(cudaEventSynchronize(*end));
    printCudaError(stderr, cudaGetLastError(), "cudaEventSynchronize");
    
    // calculate elapsed time and write it to the time variable
    CUDA_SAFE_CALL(cudaEventElapsedTime(time, *start, *end));
    printCudaError(stderr, cudaGetLastError(), "cudaEventElapsedTime");
    
    // destroy start and end event
    CUDA_SAFE_CALL(cudaEventDestroy(*start));
    printCudaError(stderr, cudaGetLastError(), "cudaEventDestroy");
    CUDA_SAFE_CALL(cudaEventDestroy(*end));
    printCudaError(stderr, cudaGetLastError(), "cudaEventDestroy");
}




// TODO This needs -lcuda which i could not manage to add to the compile line _without_ modifying common.mk :(
/*
void printMemInfo() {
    unsigned int memTotal;
    unsigned int memFree;
    cuMemGetInfo(&memFree, &memTotal);
    fprintf(stdout, "\t\t -> Mem total: %i Kbytes\n", (unsigned int)ceil(memTotal/1024));
    fprintf(stdout, "\t\t -> Mem free: %i Kbytes\n", (unsigned int)ceil(memFree/1024));
}
*/

void printCudaDeviceQuery() {
    int deviceCount;
    CUDA_SAFE_CALL(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0)
        printf("There is no device supporting CUDA\n");
    int dev;
    for (dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp deviceProp;
        CUDA_SAFE_CALL(cudaGetDeviceProperties(&deviceProp, dev));
        if (dev == 0) {
            if (deviceProp.major == 9999 && deviceProp.minor == 9999)
                printf("There is no device supporting CUDA.\n");
            else if (deviceCount == 1)
                printf("There is 1 device supporting CUDA\n");
            else
                printf("There are %d devices supporting CUDA\n", deviceCount);
        }
        printf("\nDevice %d: \"%s\"\n", dev, deviceProp.name);
        printf("  Major revision number:                         %d\n",
               deviceProp.major);
        printf("  Minor revision number:                         %d\n",
               deviceProp.minor);
        printf("  Total amount of global memory:                 %u bytes\n",
               deviceProp.totalGlobalMem);
    #if CUDART_VERSION >= 2000
        printf("  Number of multiprocessors:                     %d\n",
               deviceProp.multiProcessorCount);
        printf("  Number of cores:                               %d\n",
               8 * deviceProp.multiProcessorCount);
    #endif
        printf("  Total amount of constant memory:               %u bytes\n",
               deviceProp.totalConstMem); 
        printf("  Total amount of shared memory per block:       %u bytes\n",
               deviceProp.sharedMemPerBlock);
        printf("  Total number of registers available per block: %d\n",
               deviceProp.regsPerBlock);
        printf("  Warp size:                                     %d\n",
               deviceProp.warpSize);
        printf("  Maximum number of threads per block:           %d\n",
               deviceProp.maxThreadsPerBlock);
        printf("  Maximum sizes of each dimension of a block:    %d x %d x %d\n",
               deviceProp.maxThreadsDim[0],
               deviceProp.maxThreadsDim[1],
               deviceProp.maxThreadsDim[2]);
        printf("  Maximum sizes of each dimension of a grid:     %d x %d x %d\n",
               deviceProp.maxGridSize[0],
               deviceProp.maxGridSize[1],
               deviceProp.maxGridSize[2]);
        printf("  Maximum memory pitch:                          %u bytes\n",
               deviceProp.memPitch);
        printf("  Texture alignment:                             %u bytes\n",
               deviceProp.textureAlignment);
        printf("  Clock rate:                                    %.2f GHz\n",
               deviceProp.clockRate * 1e-6f);
    #if CUDART_VERSION >= 2000
        printf("  Concurrent copy and execution:                 %s\n",
               deviceProp.deviceOverlap ? "Yes" : "No");
    #endif
    }
}










void writeImage(float4 *image, int width, int height, char* filename, int i, int j) {
    int memSize = width*height*sizeof(float4);
    
    float4 *h_imageOutMem;
    h_imageOutMem = (float4*)malloc(memSize);
    
    char *st = (char*)malloc(30);
    sprintf(st, filename, i, j);

    CUDA_SAFE_CALL(cudaMemcpy(h_imageOutMem, image, memSize, cudaMemcpyDeviceToHost));
    printCudaError(stderr, cudaGetLastError(), "cudaMemcpy");

    SaveBMPFile(&h_imageOutMem, width, height, st);

    free(st);
    free(h_imageOutMem);
}

void writeGrayImage(float *image, int width, int height, char* filename, int i, int j) {
    int memSize = width*height*sizeof(float);
    
    float *h_imageOutMem;
    h_imageOutMem = (float*)malloc(memSize);
    
    char *st = (char*)malloc(30);
    sprintf(st, filename, i, j);

    CUDA_SAFE_CALL(cudaMemcpy(h_imageOutMem, image, memSize, cudaMemcpyDeviceToHost));
    printCudaError(stderr, cudaGetLastError(), "cudaMemcpy");

    SaveGrayBMPFile(&h_imageOutMem, width, height, st);

    free(st);
    free(h_imageOutMem);
}



