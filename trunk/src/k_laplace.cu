

texture<float4, 1, cudaReadModeElementType> la_tex_image_up; 
texture<float4, 1, cudaReadModeElementType> la_tex_image_down; 

__global__ void laplace(float4 *target, int width, int height) {
    const int ix = blockDim.x * blockIdx.x + threadIdx.x;
    const int iy = blockDim.y * blockIdx.y + threadIdx.y;

    if (ix < width && iy < height) {

        float4 p1 = tex1Dfetch(la_tex_image_up, ix + iy * width);
        float4 p2 = tex1Dfetch(la_tex_image_down, ix + iy * width);
        
        float4 t = p2 - p1;

        target[ix + iy * width] = t;
    }
}






texture<float4, 1, cudaReadModeElementType> la_tex_image_laplace;
texture<float, 1, cudaReadModeElementType> la_tex_image_weight; 

__global__ void weightedLaplace(float4 *target, int width, int height) {
    const int ix = blockDim.x * blockIdx.x + threadIdx.x;
    const int iy = blockDim.y * blockIdx.y + threadIdx.y;

    if (ix < width && iy < height) {

        float4 p1 = tex1Dfetch(la_tex_image_laplace, ix + iy * width);
        float p2 = tex1Dfetch(la_tex_image_weight, ix + iy * width);
        

        float4 t = p1 * p2;

        target[ix + iy * width] = t;
    }
}


