


texture<float4, 1, cudaReadModeElementType> ga_tex_image;


__global__ void gaussDown(float4 *target, int width, int height) {
    const int ix = blockDim.x * blockIdx.x + threadIdx.x;
    const int iy = blockDim.y * blockIdx.y + threadIdx.y;

    if (ix < width && iy < height) {

        int tx = 2 * ix;
        int ty = 2 * iy;
        int w = 2*width;

        float4 p1 = tex1Dfetch(ga_tex_image, tx - 2 + (ty -2) * w );
        float4 p2 = tex1Dfetch(ga_tex_image, tx - 1 + (ty -2) * w );
        float4 p3 = tex1Dfetch(ga_tex_image, tx - 0 + (ty -2) * w );
        float4 p4 = tex1Dfetch(ga_tex_image, tx + 1 + (ty -2) * w );
        float4 p5 = tex1Dfetch(ga_tex_image, tx + 2 + (ty -2) * w );

        float4 p6  = tex1Dfetch(ga_tex_image, tx - 2 + (ty -1) * w );
        float4 p7  = tex1Dfetch(ga_tex_image, tx - 1 + (ty -1) * w );
        float4 p8  = tex1Dfetch(ga_tex_image, tx - 0 + (ty -1) * w );
        float4 p9  = tex1Dfetch(ga_tex_image, tx + 1 + (ty -1) * w );
        float4 p10 = tex1Dfetch(ga_tex_image, tx + 2 + (ty -1) * w );

        float4 p11 = tex1Dfetch(ga_tex_image, tx - 2 + (ty -0) * w );
        float4 p12 = tex1Dfetch(ga_tex_image, tx - 1 + (ty -0) * w );
        float4 p13 = tex1Dfetch(ga_tex_image, tx - 0 + (ty -0) * w );
        float4 p14 = tex1Dfetch(ga_tex_image, tx + 1 + (ty -0) * w );
        float4 p15 = tex1Dfetch(ga_tex_image, tx + 2 + (ty -0) * w );

        float4 p16 = tex1Dfetch(ga_tex_image, tx - 2 + (ty +1) * w );
        float4 p17 = tex1Dfetch(ga_tex_image, tx - 1 + (ty +1) * w );
        float4 p18 = tex1Dfetch(ga_tex_image, tx - 0 + (ty +1) * w );
        float4 p19 = tex1Dfetch(ga_tex_image, tx + 1 + (ty +1) * w );
        float4 p20 = tex1Dfetch(ga_tex_image, tx + 2 + (ty +1) * w );

        float4 p21 = tex1Dfetch(ga_tex_image, tx - 2 + (ty +2) * w );
        float4 p22 = tex1Dfetch(ga_tex_image, tx - 1 + (ty +2) * w );
        float4 p23 = tex1Dfetch(ga_tex_image, tx - 0 + (ty +2) * w );
        float4 p24 = tex1Dfetch(ga_tex_image, tx + 1 + (ty +2) * w );
        float4 p25 = tex1Dfetch(ga_tex_image, tx + 2 + (ty +2) * w );

/*        float v_x = (1 * (p1.x + 4*p2.x + 6*p3.x + 4*p4.x + p5.x)
                     + 4 * (p6.x + 4*p7.x + 6*p8.x + 4*p9.x + p10.x)
                     + 6 * (p11.x + 4*p12.x + 6*p13.x + 4*p14.x + p15.x)
                     + 4 * (p16.x + 4*p17.x + 6*p18.x + 4*p19.x + p20.x)
                     + 1 * (p21.x + 4*p22.x + 6*p23.x + 4*p24.x + p25.x)) / 256;

        float v_y = (1 * (p1.y + 4*p2.y + 6*p3.y + 4*p4.y + p5.y)
                     + 4 * (p6.y + 4*p7.y + 6*p8.y + 4*p9.y + p10.y)
                     + 6 * (p11.y + 4*p12.y + 6*p13.y + 4*p14.y + p15.y)
                     + 4 * (p16.y + 4*p17.y + 6*p18.y + 4*p19.y + p20.y)
                     + 1 * (p21.y + 4*p22.y + 6*p23.y + 4*p24.y + p25.y)) / 256;

        float v_z = (1 * (p1.z + 4*p2.z + 6*p3.z + 4*p4.z + p5.z)
                     + 4 * (p6.z + 4*p7.z + 6*p8.z + 4*p9.z + p10.z)
                     + 6 * (p11.z + 4*p12.z + 6*p13.z + 4*p14.z + p15.z)
                     + 4 * (p16.z + 4*p17.z + 6*p18.z + 4*p19.z + p20.z)
                     + 1 * (p21.z + 4*p22.z + 6*p23.z + 4*p24.z + p25.z)) / 256;
*/

        float4 v =    (1 * (p1 +  4*p2 +  6*p3 +  4*p4 + p5)
                     + 4 * (p6 +  4*p7 +  6*p8 +  4*p9 + p10)
                     + 6 * (p11 + 4*p12 + 6*p13 + 4*p14 + p15)
                     + 4 * (p16 + 4*p17 + 6*p18 + 4*p19 + p20)
                     + 1 * (p21 + 4*p22 + 6*p23 + 4*p24 + p25)) / 256;

        
//        target[width * iy + ix] = make_color(v_x, v_y, v_z, 0);
        target[width * iy + ix] = v;
    }
}


// 1D texture, direct 1D-float
texture<float, 1, cudaReadModeElementType> ga_tex_image_f;

__global__ void gaussDown_f(float *target, int width, int height) {
    const int ix = blockDim.x * blockIdx.x + threadIdx.x;
    const int iy = blockDim.y * blockIdx.y + threadIdx.y;

    if (ix < width && iy < height) {

        int tx = 2 * ix;
        int ty = 2 * iy;
        int w = 2*width;

        float p1 = tex1Dfetch(ga_tex_image_f, tx - 2 + (ty -2) * w );
        float p2 = tex1Dfetch(ga_tex_image_f, tx - 1 + (ty -2) * w );
        float p3 = tex1Dfetch(ga_tex_image_f, tx - 0 + (ty -2) * w );
        float p4 = tex1Dfetch(ga_tex_image_f, tx + 1 + (ty -2) * w );
        float p5 = tex1Dfetch(ga_tex_image_f, tx + 2 + (ty -2) * w );

        float p6  = tex1Dfetch(ga_tex_image_f, tx - 2 + (ty -1) * w );
        float p7  = tex1Dfetch(ga_tex_image_f, tx - 1 + (ty -1) * w );
        float p8  = tex1Dfetch(ga_tex_image_f, tx - 0 + (ty -1) * w );
        float p9  = tex1Dfetch(ga_tex_image_f, tx + 1 + (ty -1) * w );
        float p10 = tex1Dfetch(ga_tex_image_f, tx + 2 + (ty -1) * w );

        float p11 = tex1Dfetch(ga_tex_image_f, tx - 2 + (ty -0) * w );
        float p12 = tex1Dfetch(ga_tex_image_f, tx - 1 + (ty -0) * w );
        float p13 = tex1Dfetch(ga_tex_image_f, tx - 0 + (ty -0) * w );
        float p14 = tex1Dfetch(ga_tex_image_f, tx + 1 + (ty -0) * w );
        float p15 = tex1Dfetch(ga_tex_image_f, tx + 2 + (ty -0) * w );

        float p16 = tex1Dfetch(ga_tex_image_f, tx - 2 + (ty +1) * w );
        float p17 = tex1Dfetch(ga_tex_image_f, tx - 1 + (ty +1) * w );
        float p18 = tex1Dfetch(ga_tex_image_f, tx - 0 + (ty +1) * w );
        float p19 = tex1Dfetch(ga_tex_image_f, tx + 1 + (ty +1) * w );
        float p20 = tex1Dfetch(ga_tex_image_f, tx + 2 + (ty +1) * w );

        float p21 = tex1Dfetch(ga_tex_image_f, tx - 2 + (ty +2) * w );
        float p22 = tex1Dfetch(ga_tex_image_f, tx - 1 + (ty +2) * w );
        float p23 = tex1Dfetch(ga_tex_image_f, tx - 0 + (ty +2) * w );
        float p24 = tex1Dfetch(ga_tex_image_f, tx + 1 + (ty +2) * w );
        float p25 = tex1Dfetch(ga_tex_image_f, tx + 2 + (ty +2) * w );

        float v_x = (1 * (p1 + 4*p2 + 6*p3 + 4*p4 + p5)
                     + 4 * (p6 + 4*p7 + 6*p8 + 4*p9 + p10)
                     + 6 * (p11 + 4*p12 + 6*p13 + 4*p14 + p15)
                     + 4 * (p16 + 4*p17 + 6*p18 + 4*p19 + p20)
                     + 1 * (p21 + 4*p22 + 6*p23 + 4*p24 + p25)) /256 ;

        

        target[width * iy + ix] = v_x;
    }
}



/*
 *  src image pixels:   a     b     c      dst image pixels:   A  B  C  D  E
 *                                                             F  G  H  I  J
 *                      d     e     f                          K  L  M  N  O
 *                                                             P  Q  R  S  T
 *                      g     h     i                          U  V  W  X  Y
 *  M =   1 * (a + 6b + c)
 *      + 6 * (d + 6e + f)
 *      + 1 * (g + 6h + i)
 *
 *  N =   1 * (4b + 4c)
 *      + 6 * (4e + 4f)
 *      + 1 * (4h + 4i)
 *
 *  R =   4 * (d + 6e + f)
 *      + 4 * (g + 6h + i)
 *
 *  S =   4 * (4e + 4f)
 *      + 4 * (4h + 4i)
 */
__global__ void gaussUp(float4 *target, int width, int height) {
    const int ix = blockDim.x * blockIdx.x + threadIdx.x;
    const int iy = blockDim.y * blockIdx.y + threadIdx.y;

    if (ix < width && iy < height) {

        int tx = 2 * ix;
        int ty = 2 * iy;

        float4 p1 = tex1Dfetch(ga_tex_image, ix-1 + (iy-1) * width);
        float4 p2 = tex1Dfetch(ga_tex_image, ix + (iy-1) * width);
        float4 p3 = tex1Dfetch(ga_tex_image, ix+1 + (iy-1) * width);

        float4 p4 = tex1Dfetch(ga_tex_image, ix-1 + iy * width);
        float4 p5 = tex1Dfetch(ga_tex_image, ix + iy * width);
        float4 p6 = tex1Dfetch(ga_tex_image, ix+1 + iy * width);

        float4 p7 = tex1Dfetch(ga_tex_image, ix-1 + (iy+1) * width);
        float4 p8 = tex1Dfetch(ga_tex_image, ix + (iy+1) * width);
        float4 p9 = tex1Dfetch(ga_tex_image, ix+1 + (iy+1) * width);


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


        float4 n1 = t1 / 64;
        float4 n2 = t2 / 64;
        float4 n3 = t3 / 64;
        float4 n4 = t4 / 64;

        
/*
        target[2*width * (ty) + tx] = make_color(n1.x, n1.y, n1.z, 0);
        target[2*width * (ty) + (ix * 2 + 1)] = make_color(n2.x, n2.y, n2.z, 0);
        target[2*width * (ty + 1) + tx] = make_color(n3.x, n3.y, n3.z, 0);
        target[2*width * (ty + 1) + (ix * 2 + 1)] = make_color(n4.x, n4.y, n4.z, 0);
*/
        target[2*width * (ty) + tx] = n1;
        target[2*width * (ty) + (ix * 2 + 1)] = n2;
        target[2*width * (ty + 1) + tx] = n3;
        target[2*width * (ty + 1) + (ix * 2 + 1)] = n4;
    }
}
