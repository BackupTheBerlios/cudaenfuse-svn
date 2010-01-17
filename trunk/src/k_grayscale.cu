

// 2D texture, normalized 4D-float
texture<uchar4, 2, cudaReadModeNormalizedFloat> g_tex_image; 



__global__ void grayscale(float *gray, int width, int height) {

    const int ix = blockDim.x * blockIdx.x + threadIdx.x;
    const int iy = blockDim.y * blockIdx.y + threadIdx.y;

    if (ix < width && iy < height) {
        float4 pixel = tex2D(g_tex_image, ix, iy);
        //gray[width * iy + ix] = 0.313524 * pixel.x + 0.615514 * pixel.y + 0.119537 * pixel.z + 0.524288;
        //gray[width * iy + ix] = 0.313524 * pixel.x + 0.615514 * pixel.y + 0.119537 * pixel.z;
        gray[width * iy + ix] = (pixel.x + pixel.y + pixel.z) / 3;
    }
    
}
