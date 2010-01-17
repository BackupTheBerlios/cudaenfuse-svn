#ifndef CUDA_WRAPPERS_H
#define CUDA_WRAPPERS_H



/*
 * Error handling functions
 */
extern "C++" void printCudaError(FILE* target, cudaError_t cudaErrno, char* msg);
extern "C++" void printCudaError(FILE* target, cudaError_t cudaErrno);

/*
 * Device management
 */
extern "C++" void cuda_init(int argc, char** argv);
extern "C++" void cuda_exit(int argc, char** argv);
 
extern "C++" cudaArray *cuda_mallocArray(cudaChannelFormatDesc *channelDesc, int width, int height);
extern "C++" template<typename dataType> dataType *cuda_malloc(int width, int height);

extern "C++" cudaArray *cuda_uploadToArray(cudaArray* target, const void* source, int memSize);
extern "C++" cudaArray *cuda_uploadToArray(cudaChannelFormatDesc *channelDesc, int width, int height, const void *source);

extern "C++" template<typename T> T *cuda_download(const T* source, int width, int height);

extern "C++" template<typename dataType> dataType *cuda_memcpy(dataType* source, int width, int height);


extern "C++" void cuda_bindTexture(cudaChannelFormatDesc *channelDesc, const cudaArray* source, const char* texRefName);
extern "C++" template<typename dataType> void cuda_bindTexture(const dataType* source, int width, int height, const char* texRefName);
extern "C++" void cuda_unbindTexture(const char *texRefName);




/*
 * Device information
 */
extern "C++" void printCudaDeviceQuery();
//extern "C++" void printMemInfo();


/*
 * Timers
 */
extern "C++" void cuda_startTimer(cudaEvent_t *start, cudaEvent_t *end);
extern "C++" void cuda_stopTimer(cudaEvent_t *start, cudaEvent_t *end, float *time);


/*
 * Output (to harddisk) functions
 */
extern "C++" void writeImage(float4 *image, int width, int height, char* filename, int i, int j);
extern "C++" void writeGrayImage(float *image, int width, int height, char* filename, int i, int j);

#endif
