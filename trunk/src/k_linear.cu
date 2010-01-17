texture<float, 1, cudaReadModeElementType> dummy;

__global__ void linearDown_f(float *target, int width, int height) {
    const int ix = blockDim.x * blockIdx.x + threadIdx.x;
    const int iy = blockDim.y * blockIdx.y + threadIdx.y;

    if (ix < width && iy < height) {

        int tx = 2 * ix;
        int ty = 2 * iy;
        int w = 2*width;

        float p1 = tex1Dfetch(dummy, tx - 2 + (ty -2) * w );


        target[width * iy + ix] = p1;
    }
}

