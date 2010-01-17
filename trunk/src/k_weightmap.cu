

texture<float, 1, cudaReadModeElementType> w_tex_gray;
texture<uchar4, 2, cudaReadModeNormalizedFloat> w_tex_image; 


__global__ void weightmap(float* weight, int width, int height) {
    const int ix = blockDim.x * blockIdx.x + threadIdx.x;
    const int iy = blockDim.y * blockIdx.y + threadIdx.y;

    if (ix < width && iy < height) {

        float4 pixel = tex2D(w_tex_image, ix, iy);


        // well-exposedness (based on source image pixel)
        float well_x = exp( (-1) * pow(pixel.x - 0.5, 2) / 0.08 );
        float well_y = exp( (-1) * pow(pixel.y - 0.5, 2) / 0.08 );
        float well_z = exp( (-1) * pow(pixel.z - 0.5, 2) / 0.08 );
        float w_exposedness = well_x * well_y * well_z+0.005;


        // contrast (based on grayscale pixel)
        float gv_1 = tex1Dfetch(w_tex_gray, (ix-1) +  (iy-1) * width);
        float gv_2 = tex1Dfetch(w_tex_gray, ix + (iy-1) * width);
        float gv_3 = tex1Dfetch(w_tex_gray, (ix+1) + (iy-1) * width);
        
        float gv_4 = tex1Dfetch(w_tex_gray, (ix-1) + iy * width);
        float gv_5 = tex1Dfetch(w_tex_gray, ix + iy * width);
        float gv_6 = tex1Dfetch(w_tex_gray, (ix+1) + iy * width);

        float gv_7 = tex1Dfetch(w_tex_gray, (ix-1) + (iy+1) * width);
        float gv_8 = tex1Dfetch(w_tex_gray, ix + (iy+1) * width);
        float gv_9 = tex1Dfetch(w_tex_gray, (ix+1) + (iy+1)  * width);
        float w_contrast = abs(gv_1 + gv_2 + gv_3 + gv_4 + (-8) * gv_5 + gv_6 + gv_7 + gv_8 + gv_9)+0.005;

        
        // saturation (based on source image pixel)
        float max_value = max(max(pixel.x, pixel.y), pixel.z);
        float min_value = min(min(pixel.x, pixel.y), pixel.z);
        float w_saturation = (max_value - min_value)+0.005;


    

        // weighting // TODO
//       weight[width * iy + ix] = w_contrast * w_saturation * w_exposedness / 0.223f;
//        weight[width * iy + ix] = (pow(w_contrast, 1) * pow(w_saturation, 1) * pow(w_exposedness, 5));
        weight[width * iy + ix] = (pow(w_contrast, 1) * pow(w_saturation, 1) * pow(w_exposedness, 1));
    }
}


/*
 * Adds the values of the two given weightmaps at each pixel and writes the
 * result to the target weightmap.
 */
__global__ void addToTarget(float* target, float* add, int width, int height) {    
    const int ix = blockDim.x * blockIdx.x + threadIdx.x;
    const int iy = blockDim.y * blockIdx.y + threadIdx.y;

    if (ix < width && iy < height) {
        float temp = target[ix + iy * width];
        target[ix + iy * width] = temp + add[ix + iy * width];
    }
}

/*
 * Normalizes the given weightmap to the divisor given, so that all weightmaps included
 * in the divisor and normalized by this kernel add up to 1.
 * This kernel is intended to be used with the output of the addToTarget kernel.
 */
texture<float, 1, cudaReadModeElementType> w_tex_divisor;
__global__ void normalizeWeightmap(float* map, int width, int height) {    
    const int ix = blockDim.x * blockIdx.x + threadIdx.x;
    const int iy = blockDim.y * blockIdx.y + threadIdx.y;

    if (ix < width && iy < height) {
        float temp = map[ix + iy * width];
        map[ix + iy * width] =   temp / tex1Dfetch(w_tex_divisor, ix + iy * width);
    }
}


/*
 * Normalizes two given weightmaps, so that their values always add up to 1 at
 * each pixel position.
 */
__global__ void normalizeWeightmaps(float* map1, float* map2, int width, int height) {    
    const int ix = blockDim.x * blockIdx.x + threadIdx.x;
    const int iy = blockDim.y * blockIdx.y + threadIdx.y;

    if (ix < width && iy < height) {
    
        float p1 = map1[ix + iy * width];
        float p2 = map2[ix + iy * width];

        float d = p1 + p2 ;
        map1[ix + iy * width] = (p1 / d);
        map2[ix + iy * width] = (p2 / d);     

    }
}

