

texture<float4, 1, cudaReadModeElementType> c_tex_image;


__global__ void collapse(float4 *src_big, int width, int height) {
    const int ix = blockDim.x * blockIdx.x + threadIdx.x;
    const int iy = blockDim.y * blockIdx.y + threadIdx.y;

    if (ix < width && iy < height) {

        float4 t = tex1Dfetch(c_tex_image, ix + iy * width);



        float4 p1 = tex1Dfetch(c_tex_image, ix-1 + (iy-1) * width);
        float4 p2 = tex1Dfetch(c_tex_image, ix + (iy-1) * width);
        float4 p3 = tex1Dfetch(c_tex_image, ix+1 + (iy-1) * width);

        float4 p4 = tex1Dfetch(c_tex_image, ix-1 + iy * width);
        float4 p5 = tex1Dfetch(c_tex_image, ix + iy * width);
        float4 p6 = tex1Dfetch(c_tex_image, ix+1 + iy * width);

        float4 p7 = tex1Dfetch(c_tex_image, ix-1 + (iy+1) * width);
        float4 p8 = tex1Dfetch(c_tex_image, ix + (iy+1) * width);
        float4 p9 = tex1Dfetch(c_tex_image, ix+1 + (iy+1) * width);


        if (iy == (height-1)) {
            p7 = p5;
            p8 = p5;
            p9 = p5;
        }

        if (ix == (width-1)) {
            p3 = p5;
            p6 = p5;
            p9 = p5; 
        }


        float4 t1 = 1 * (p1 + 6*p2 + p3) +
                    6 * (p4 + 6*p5 + p6) +
                    1 * (p7 + 6*p8 + p9);
        float4 t2 = 1 * (4*p2 + 4*p3) +
                    6 * (4*p5 + 4*p6) +
                    1 * (4*p8 + 4*p9);
        float4 t3 = 4 * (p4 + 6*p5 + p6) +
                    4 * (p7 + 6*p8 + p9);
        float4 t4 = 4 * (4*p5 + 4*p6) +
                    4 * (4*p8 + 4*p9);


        float4 e1 = t1 / 64;
        float4 e2 = t2 / 64;
        float4 e3 = t3 / 64;
        float4 e4 = t4 / 64;


        float4 q1 = src_big[2*ix     + 2*iy     * width*2] + e1;
        float4 q2 = src_big[2*ix + 1 + 2*iy     * width*2] + e2;
        float4 q3 = src_big[2*ix     + (2*iy+1) * width*2] + e3;
        float4 q4 = src_big[2*ix + 1 + (2*iy+1) * width*2] + e4;

        src_big[2*ix     + 2*iy     * width*2] = clamp(q1, 0.0f, 1.0f);
        src_big[2*ix + 1 + 2*iy     * width*2] = clamp(q2, 0.0f, 1.0f);
        src_big[2*ix     + (2*iy+1) * width*2] = clamp(q3, 0.0f, 1.0f);
        src_big[2*ix + 1 + (2*iy+1) * width*2] = clamp(q4, 0.0f, 1.0f);
    }
}


