

#include <stdio.h>

// CUDA includes
#include <cutil.h>
#include <cutil_math.h>

// own stuff
#include "cuda_wrappers.h"

// kernel implementations
#include "k_grayscale.cu"
#include "k_weightmap.cu"
#include "k_gaussian.cu"
#include "k_laplace.cu"
#include "k_combine.cu"
#include "k_collapse.cu"
#include "k_copy.cu"
#include "k_linear.cu"
#include "k_simpleEnfuse.cu"


#include "cuda_wrappers.h"
#include "filter_wrappers.h"


/*
 * GLOBAL VARIABLES
 */

// configure the grid layout
const int blocksize_x = 8;
const int blocksize_y = 8;

// cuda channel descriptor 
cudaChannelFormatDesc uchar4tex = cudaCreateChannelDesc<uchar4>(); 



/*
 * Upload the given RGB structure given by a uchar4 pointer into a cudaArray in
 * the GPU's memory and return a pointer to the array on the GPU's memory.
 */
cudaArray* uploadToArray(const uchar4 *source, int width, int height) {
    return cuda_uploadToArray(&uchar4tex, width, height, source);
}


float4 *genSimpleEnfuse(float4 *target, cudaArray* source, float *weight, int width, int height) {

    int memSize_weight = width * height * sizeof(float);

    
    // make sources available as texture
    cuda_bindTexture(&uchar4tex, source, "en_tex_image");
    CUDA_SAFE_CALL(cudaBindTexture(0, en_tex_image_weight, weight, memSize_weight));
    printCudaError(stderr, cudaGetLastError(), "cudaBindTexture");


    // prepare kernel execution
    dim3 grid(floor(width/blocksize_x) + 1,floor(height/blocksize_y) + 1);
    dim3 threads(blocksize_x, blocksize_y);

    // execute kernel
    simpleEnfuse <<< grid, threads >>>(target, width, height);
    printCudaError(stderr, cudaGetLastError(), "simpleEnfuse");



    // cleanup
    CUDA_SAFE_CALL(cudaUnbindTexture(en_tex_image));
    printCudaError(stderr, cudaGetLastError(), "cudaUnbindTexture");
    CUDA_SAFE_CALL(cudaUnbindTexture(en_tex_image_weight));
    printCudaError(stderr, cudaGetLastError(), "cudaUnbindTexture");

    return target;
}

float4 *genAppliedWeightmap(cudaArray* source, float *weight, int width, int height) {

    float4 *target;

    int memSize_target = width * height * sizeof(float4);
    int memSize_weight = width * height * sizeof(float);

    // allocate graphics memory for target
    CUDA_SAFE_CALL(cudaMalloc((void**) &target, memSize_target));
    printCudaError(stderr, cudaGetLastError(), "cudaMalloc");
    
    // make sources available as texture
    cuda_bindTexture(&uchar4tex, source, "en_tex_image");
    CUDA_SAFE_CALL(cudaBindTexture(0, en_tex_image_weight, weight, memSize_weight));
    printCudaError(stderr, cudaGetLastError(), "cudaBindTexture");


    // prepare kernel execution
    dim3 grid(floor(width/blocksize_x) + 1,floor(height/blocksize_y) + 1);
    dim3 threads(blocksize_x, blocksize_y);

    // execute kernel
    applyWeightmap <<< grid, threads >>>(target, width, height);
    printCudaError(stderr, cudaGetLastError(), "applyWeightmap");



    // cleanup
    CUDA_SAFE_CALL(cudaUnbindTexture(en_tex_image));
    printCudaError(stderr, cudaGetLastError(), "cudaUnbindTexture");
    CUDA_SAFE_CALL(cudaUnbindTexture(en_tex_image_weight));
    printCudaError(stderr, cudaGetLastError(), "cudaUnbindTexture");

    return target;
}


float* genGrayscale(const cudaArray *source, int width, int height) {
    float *target = cuda_malloc<float>(width, height);
    cuda_bindTexture(&uchar4tex, source, "g_tex_image");

    // kernel execution
    dim3 grid(floor(width/blocksize_x) + 1,floor(height/blocksize_y) + 1);
    dim3 threads(blocksize_x, blocksize_y);
    grayscale <<< grid, threads >>>(target, width, height);
    printCudaError(stderr, cudaGetLastError(), "grayscale");

    cuda_unbindTexture("g_tex_image");    
    return target;
}

float *genWeightmap(const cudaArray *image_source, const float *gray_source, int width, int height) {

    float *target = cuda_malloc<float>(width, height);
    cuda_bindTexture(&uchar4tex, image_source, "w_tex_image");
    cuda_bindTexture<float>(gray_source, width, height, "w_tex_gray");

    // kernel execution
    dim3 grid(floor(width/blocksize_x) + 1,floor(height/blocksize_y) + 1);
    dim3 threads(blocksize_x, blocksize_y);
    weightmap <<< grid, threads >>>(target, width, height);
    printCudaError(stderr, cudaGetLastError(), "weightmap");

    cuda_unbindTexture("w_tex_image");    
    cuda_unbindTexture("w_tex_gray");    
    return target;
}


float4 *genGaussDown(float4 *source, int *width, int *height) {
    
    float4 *target;

    int target_width = floor(*width / 2); // TODO
    int target_height = floor(*height / 2); // TODO
    int target_memSize = target_width * target_height * sizeof(float4);
    
    int source_memSize = (*width) * (*height) * sizeof(float4);
//    printf("w: %i, h: %i\n", *width, *height);

    // allocate graphics memory for target
    CUDA_SAFE_CALL(cudaMalloc((void**) &target, target_memSize));
    printCudaError(stderr, cudaGetLastError(), "cudaMalloc");
    
    // make sources available as texture
    CUDA_SAFE_CALL(cudaBindTexture(0, ga_tex_image, source, source_memSize));
    printCudaError(stderr, cudaGetLastError(), "cudaBindTexture");


    // prepare kernel execution
    dim3 grid(floor(target_width/blocksize_x) + 1,floor(target_height/blocksize_y) + 1);
    dim3 threads(blocksize_x, blocksize_y);

    // execute kernel
    gaussDown <<< grid, threads >>>(target, target_width, target_height);
    printCudaError(stderr, cudaGetLastError(), "gauss");

    // update new image dimensions
    *width = target_width;
    *height = target_height;

    // cleanup
    CUDA_SAFE_CALL(cudaUnbindTexture(ga_tex_image));
    printCudaError(stderr, cudaGetLastError(), "cudaUnbindTexture");


    return target;
}

float4 *toFloat4(cudaArray *source, int width, int height) {
    
    float4 *target;

    // allocate graphics memory for target
    int memSize_target = width * height * sizeof(float4);
    CUDA_SAFE_CALL(cudaMalloc((void**) &target, memSize_target));
    printCudaError(stderr, cudaGetLastError(), "cudaMalloc");
    
    // make sources available as texture
    CUDA_SAFE_CALL(cudaBindTextureToArray(tf_tex_source, source));
    printCudaError(stderr, cudaGetLastError(), "cudaBindTextureToArray");

    // prepare kernel execution
    dim3 grid(floor(width/blocksize_x) + 1,floor(height/blocksize_y) + 1);
    dim3 threads(blocksize_x, blocksize_y);

    // execute kernel
    copyUcharToFloat <<< grid, threads >>>(target, width, height);
    printCudaError(stderr, cudaGetLastError(), "copyUcharToFloat");

    // cleanup
    CUDA_SAFE_CALL(cudaUnbindTexture(tf_tex_source));
    printCudaError(stderr, cudaGetLastError(), "cudaUnbindTexture");

    return target;
}


float *genGaussDown_f(float *source, int *width, int *height) {
    
    float *target;

    int target_width = floor(*width / 2); // TODO
    int target_height = floor(*height / 2); // TODO
    int target_memSize = target_width * target_height * sizeof(float);
    
    int source_memSize = (*width) * (*height) * sizeof(float);
//    printf("w: %i, h: %i\n", *width, *height);

    // allocate graphics memory for target
    CUDA_SAFE_CALL(cudaMalloc((void**) &target, target_memSize));
    printCudaError(stderr, cudaGetLastError(), "cudaMalloc");
    
    // make sources available as texture
    CUDA_SAFE_CALL(cudaBindTexture(0, ga_tex_image_f, source, source_memSize));
    printCudaError(stderr, cudaGetLastError(), "cudaBindTexture");


    // prepare kernel execution
    dim3 grid(floor(target_width/blocksize_x) + 1,floor(target_height/blocksize_y) + 1);
    dim3 threads(blocksize_x, blocksize_y);

    // execute kernel
    gaussDown_f <<< grid, threads >>>(target, target_width, target_height);
    printCudaError(stderr, cudaGetLastError(), "gaussDown_f");

    // update new image dimensions
    *width = target_width;
    *height = target_height;

    // cleanup
    CUDA_SAFE_CALL(cudaUnbindTexture(ga_tex_image));
    printCudaError(stderr, cudaGetLastError(), "cudaUnbindTexture");


    return target;
}

float *genLinearDown_f(float *source, int *width, int *height) {
    
    float *target;

    int target_width = floor(*width / 2); // TODO
    int target_height = floor(*height / 2); // TODO
    int target_memSize = target_width * target_height * sizeof(float);
    
    int source_memSize = (*width) * (*height) * sizeof(float);
//    printf("w: %i, h: %i\n", *width, *height);

    // allocate graphics memory for target
    CUDA_SAFE_CALL(cudaMalloc((void**) &target, target_memSize));
    printCudaError(stderr, cudaGetLastError(), "cudaMalloc");
    
    // make sources available as texture
    CUDA_SAFE_CALL(cudaBindTexture(0, dummy, source, source_memSize));
    printCudaError(stderr, cudaGetLastError(), "cudaBindTexture");


    // prepare kernel execution
    dim3 grid(floor(target_width/blocksize_x) + 1,floor(target_height/blocksize_y) + 1);
    dim3 threads(blocksize_x, blocksize_y);

    // execute kernel
    linearDown_f <<< grid, threads >>>(target, target_width, target_height);
    printCudaError(stderr, cudaGetLastError(), "linearDown_f");

    // update new image dimensions
    *width = target_width;
    *height = target_height;

    // cleanup
    CUDA_SAFE_CALL(cudaUnbindTexture(dummy));
    printCudaError(stderr, cudaGetLastError(), "cudaUnbindTexture");


    return target;
}




float4 *genGaussUp(float4 *source, int *width, int *height) {
    
    float4 *target;

    int target_width = floor(*width * 2); // TODO
    int target_height = floor(*height * 2); // TODO
    int target_memSize = target_width * target_height * sizeof(float4);
    
    int source_memSize = (*width) * (*height) * sizeof(float4);
//    printf("w: %i, h: %i\n", *width, *height);

    // allocate graphics memory for target
    CUDA_SAFE_CALL(cudaMalloc((void**) &target, target_memSize));
    printCudaError(stderr, cudaGetLastError(), "cudaMalloc");
    
    // make sources available as texture
    CUDA_SAFE_CALL(cudaBindTexture(0, ga_tex_image, source, source_memSize));
    printCudaError(stderr, cudaGetLastError(), "cudaBindTexture");


    // prepare kernel execution
    dim3 grid(floor(*width/blocksize_x) + 1,floor(*height/blocksize_y) + 1);
    dim3 threads(blocksize_x, blocksize_y);

    // execute kernel
    gaussUp <<< grid, threads >>>(target, *width, *height);
    printCudaError(stderr, cudaGetLastError(), "gauss");

    // update new image dimensions
    *width = target_width;
    *height = target_height;

    // cleanup
    CUDA_SAFE_CALL(cudaUnbindTexture(ga_tex_image));
    printCudaError(stderr, cudaGetLastError(), "cudaUnbindTexture");


    return target;
}



float4 *genLaplace(float4 *source_down, float4 *source_up, int *width, int *height) {
    
    float4 *target;

    
    int memSize = (*width) * (*height) * sizeof(float4);
//    printf("w: %i, h: %i\n", *width, *height);

    // allocate graphics memory for target
    CUDA_SAFE_CALL(cudaMalloc((void**) &target, memSize));
    printCudaError(stderr, cudaGetLastError(), "cudaMalloc");
    
    // make sources available as texture
    CUDA_SAFE_CALL(cudaBindTexture(0, la_tex_image_down, source_down, memSize));
    printCudaError(stderr, cudaGetLastError(), "cudaBindTexture");
    CUDA_SAFE_CALL(cudaBindTexture(0, la_tex_image_up, source_up, memSize));
    printCudaError(stderr, cudaGetLastError(), "cudaBindTexture");


    // prepare kernel execution
    dim3 grid(floor(*width/blocksize_x) + 1,floor(*height/blocksize_y) + 1);
    dim3 threads(blocksize_x, blocksize_y);

    // execute kernel
    laplace <<< grid, threads >>>(target, *width, *height);
    printCudaError(stderr, cudaGetLastError(), "laplace");



    // FIXME
    // update new image dimensions
    int target_width = floor(*width / 2); // TODO
    int target_height = floor(*height / 2); // TODO
    *width = target_width;
    *height = target_height;


    // cleanup
    CUDA_SAFE_CALL(cudaUnbindTexture(la_tex_image_up));
    printCudaError(stderr, cudaGetLastError(), "cudaUnbindTexture");
    CUDA_SAFE_CALL(cudaUnbindTexture(la_tex_image_down));
    printCudaError(stderr, cudaGetLastError(), "cudaUnbindTexture");


    return target;
}


float4 *genWeightedLaplace(float4 *source_laplace, float *source_weight, int *width, int *height) {

    float4 *target;

    int memSize_laplace = (*width) * (*height) * sizeof(float4);
    int memSize_weight = (*width) * (*height) * sizeof(float);
//    printf("w: %i, h: %i\n", *width, *height);

    // allocate graphics memory for target
    CUDA_SAFE_CALL(cudaMalloc((void**) &target, memSize_laplace));
    printCudaError(stderr, cudaGetLastError(), "cudaMalloc");
    
    // make sources available as texture
    CUDA_SAFE_CALL(cudaBindTexture(0, la_tex_image_laplace, source_laplace, memSize_laplace));
    printCudaError(stderr, cudaGetLastError(), "cudaBindTexture");
    CUDA_SAFE_CALL(cudaBindTexture(0, la_tex_image_weight, source_weight, memSize_weight));
    printCudaError(stderr, cudaGetLastError(), "cudaBindTexture");


    // prepare kernel execution
    dim3 grid(floor(*width/blocksize_x) + 1,floor(*height/blocksize_y) + 1);
    dim3 threads(blocksize_x, blocksize_y);

    // execute kernel
    weightedLaplace <<< grid, threads >>>(target, *width, *height);
    printCudaError(stderr, cudaGetLastError(), "weightedLaplace");



    // FIXME
    // update new image dimensions
    int target_width = floor(*width / 2); // TODO
    int target_height = floor(*height / 2); // TODO
    *width = target_width;
    *height = target_height;


    // cleanup
    CUDA_SAFE_CALL(cudaUnbindTexture(la_tex_image_up));
    printCudaError(stderr, cudaGetLastError(), "cudaUnbindTexture");
    CUDA_SAFE_CALL(cudaUnbindTexture(la_tex_image_down));
    printCudaError(stderr, cudaGetLastError(), "cudaUnbindTexture");

    return target;
}


void normWeightmaps(float *map_1, float *map_2, int width, int height) {
    // prepare kernel execution
    dim3 grid(floor(width/blocksize_x) + 1,floor(height/blocksize_y) + 1);
    dim3 threads(blocksize_x, blocksize_y);

    // execute kernel
    normalizeWeightmaps <<< grid, threads >>>(map_1, map_2, width, height);
    printCudaError(stderr, cudaGetLastError(), "normalizeWeightmaps");
}


void normWeightmaps(float **maps, int numMaps, int width, int height) {

    // sum up all given weightmaps to the target
    float *target = cuda_memcpy<float>(maps[0], width, height);
    for (int i = 1; i < numMaps; i++) {
        // prepare kernel execution
        dim3 grid(floor(width/blocksize_x) + 1,floor(height/blocksize_y) + 1);
        dim3 threads(blocksize_x, blocksize_y);

        // execute kernel
        addToTarget <<< grid, threads >>>(target, maps[i], width, height);
        printCudaError(stderr, cudaGetLastError(), "addToTarget");
    }


    // normalize all given weightmaps with the calculated divisors
    cuda_bindTexture<float>(target, width, height, "w_tex_divisor");    
    for (int i = 0; i < numMaps; i++) {
        // prepare kernel execution
        dim3 grid(floor(width/blocksize_x) + 1,floor(height/blocksize_y) + 1);
        dim3 threads(blocksize_x, blocksize_y);

        // execute kernel
        normalizeWeightmap <<< grid, threads >>>(maps[i], width, height);
        printCudaError(stderr, cudaGetLastError(), "normalizeWeightmap");
    }    
    
   cuda_unbindTexture("w_tex_divisor");
   cudaFree(target);
}





float4 *genAdded(float4* source_1, float4 *source_2, int *width, int *height) {

    float4 *target;

    int memSize = (*width) * (*height) * sizeof(float4);
//    printf("w: %i, h: %i\n", *width, *height);

    // allocate graphics memory for target
    CUDA_SAFE_CALL(cudaMalloc((void**) &target, memSize));
    printCudaError(stderr, cudaGetLastError(), "cudaMalloc");


    // make sources available as texture
    CUDA_SAFE_CALL(cudaBindTexture(0, a_tex_image_1, source_1, memSize));
    printCudaError(stderr, cudaGetLastError(), "cudaBindTexture");
    CUDA_SAFE_CALL(cudaBindTexture(0, a_tex_image_2, source_2, memSize));
    printCudaError(stderr, cudaGetLastError(), "cudaBindTexture");


    // prepare kernel execution
    dim3 grid(floor(*width/blocksize_x) + 1,floor(*height/blocksize_y) + 1);
    dim3 threads(blocksize_x, blocksize_y);

    // execute kernel
    add <<< grid, threads >>>(target, *width, *height);
    printCudaError(stderr, cudaGetLastError(), "laplace");




    // FIXME
    // update new image dimensions
    int target_width = floor(*width / 2); // TODO
    int target_height = floor(*height / 2); // TODO
    *width = target_width;
    *height = target_height;


    // cleanup
    CUDA_SAFE_CALL(cudaUnbindTexture(a_tex_image_1));
    printCudaError(stderr, cudaGetLastError(), "cudaUnbindTexture");
    CUDA_SAFE_CALL(cudaUnbindTexture(a_tex_image_2));
    printCudaError(stderr, cudaGetLastError(), "cudaUnbindTexture");

    return target;
}


float *genAdded(float* source_1, float *source_2, int *width, int *height) {

    float *target;

    int memSize = (*width) * (*height) * sizeof(float);
//    printf("w: %i, h: %i\n", *width, *height);

    // allocate graphics memory for target
    CUDA_SAFE_CALL(cudaMalloc((void**) &target, memSize));
    printCudaError(stderr, cudaGetLastError(), "cudaMalloc");


    // make sources available as texture
    CUDA_SAFE_CALL(cudaBindTexture(0, af_tex_image_1, source_1, memSize));
    printCudaError(stderr, cudaGetLastError(), "cudaBindTexture");
    CUDA_SAFE_CALL(cudaBindTexture(0, af_tex_image_2, source_2, memSize));
    printCudaError(stderr, cudaGetLastError(), "cudaBindTexture");


    // prepare kernel execution
    dim3 grid(floor(*width/blocksize_x) + 1,floor(*height/blocksize_y) + 1);
    dim3 threads(blocksize_x, blocksize_y);

    // execute kernel
    add_f <<< grid, threads >>>(target, *width, *height);
    printCudaError(stderr, cudaGetLastError(), "laplace");

    // cleanup
    CUDA_SAFE_CALL(cudaUnbindTexture(af_tex_image_1));
    printCudaError(stderr, cudaGetLastError(), "cudaUnbindTexture");
    CUDA_SAFE_CALL(cudaUnbindTexture(af_tex_image_2));
    printCudaError(stderr, cudaGetLastError(), "cudaUnbindTexture");

    return target;
}





float4 *genCollapsed(float4 *src_big, float4 *src_small, int *width, int *height) {
    
    int memSize = (*width) * (*height) * sizeof(float4);

    // make sources available as texture
    CUDA_SAFE_CALL(cudaBindTexture(0, c_tex_image, src_small, memSize));
    printCudaError(stderr, cudaGetLastError(), "cudaBindTexture");


    // prepare kernel execution
    dim3 grid(floor(*width/blocksize_x) + 1,floor(*height/blocksize_y) + 1);
    dim3 threads(blocksize_x, blocksize_y);

    // execute kernel
    collapse <<< grid, threads >>>(src_big, *width, *height);
    printCudaError(stderr, cudaGetLastError(), "laplace");




    *width = floor(*width * 2); // TODO
    *height = floor(*height * 2); // TODO

    return src_big;
}

