

texture<float4, 1, cudaReadModeElementType> a_tex_image_1;
texture<float4, 1, cudaReadModeElementType> a_tex_image_2;

__global__ void add(float4 *target, int width, int height) {
    const int ix = blockDim.x * blockIdx.x + threadIdx.x;
    const int iy = blockDim.y * blockIdx.y + threadIdx.y;

    if (ix < width && iy < height) {

        float4 p1 = tex1Dfetch(a_tex_image_1, ix + iy * width);
        float4 p2 = tex1Dfetch(a_tex_image_2, ix + iy * width);
        
        float4 t = (p2 + p1);

        target[ix + iy * width] = t;
    }
}






texture<float, 1, cudaReadModeElementType> af_tex_image_1;
texture<float, 1, cudaReadModeElementType> af_tex_image_2;

__global__ void add_f(float *target, int width, int height) {
    const int ix = blockDim.x * blockIdx.x + threadIdx.x;
    const int iy = blockDim.y * blockIdx.y + threadIdx.y;

    if (ix < width && iy < height) {

        float p1 = tex1Dfetch(af_tex_image_1, ix + iy * width);
        float p2 = tex1Dfetch(af_tex_image_2, ix + iy * width);
        
        float t = (p2 + p1);

        target[ix + iy * width] = t;
    }
}


