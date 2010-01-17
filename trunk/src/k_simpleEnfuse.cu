
texture<uchar4, 2, cudaReadModeNormalizedFloat> en_tex_image; 
texture<float, 1, cudaReadModeElementType> en_tex_image_weight; 

__global__ void simpleEnfuse(float4 *target, int width, int height) {
    const int ix = blockDim.x * blockIdx.x + threadIdx.x;
    const int iy = blockDim.y * blockIdx.y + threadIdx.y;

    if (ix < width && iy < height) {

        float4 p = tex2D(en_tex_image, ix, iy);
        float w = tex1Dfetch(en_tex_image_weight, ix + iy * width);
        
        float4 e = p * w;
        float4 t = target[ix + iy * width];

        target[ix + iy * width] = t + e;
        
    }
}


__global__ void applyWeightmap(float4 *target, int width, int height) {
    const int ix = blockDim.x * blockIdx.x + threadIdx.x;
    const int iy = blockDim.y * blockIdx.y + threadIdx.y;

    if (ix < width && iy < height) {

        float4 p = tex2D(en_tex_image, ix, iy);
        float w = tex1Dfetch(en_tex_image_weight, ix + iy * width);
        
        float4 e = p * w;

        target[ix + iy * width] = e;
        
    }
}

