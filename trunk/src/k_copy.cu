



texture<uchar4, 2, cudaReadModeNormalizedFloat> tf_tex_source; 

__global__ void copyUcharToFloat(float4 *target, int width, int height) {

    const int ix = blockDim.x * blockIdx.x + threadIdx.x;
    const int iy = blockDim.y * blockIdx.y + threadIdx.y;

    if (ix < width && iy < height) {
        float4 pixel = tex2D(tf_tex_source, ix, iy);
        target[width * iy + ix] = pixel;
    }
    
}
