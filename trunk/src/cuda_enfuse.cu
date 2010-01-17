/*
 * cuda_enfuse.cu
 *
 * Uses CUDA to enfuse two or more input images of different
 * exposure qualities into one well-exposed image.
 *
 * The algorithm was adapted from the enfuse/enblend implementation
 * available at http://enblend.sourceforge.net/ and described in detail
 * in http://research.edm.uhasselt.be/%7Etmertens/papers/exposure_fusion_reduced.pdf
 *
 *
 * Authors: Florian Löffler <flo@geek-blog.de>
 * Licence: LGPLv3
 */


#include <stdio.h>

#include "cuda_wrappers.h"
#include "filter_wrappers.h"

#include "cuda_enfuse.h"





float4 *cuda_enfuse(uchar4 **h_imageInMems, int width, int height, int imageInNum) {
    return cuda_enfuse(h_imageInMems, width, height, imageInNum, 7);
}

float4 *cuda_enfuse(uchar4 **h_imageInMems, int width, int height, int imageInNum, int scaleDepth) {

    // Some benchmarking
    cudaEvent_t     start;
    cudaEvent_t     end;
    float           execution_time = 0.0f;

    // temporary variables to track change in image dimensions
    int             scale_width;
    int             scale_height;

    // Weightmap images in graphics memory
    float           *g_imageWeightMems[imageInNum];
    
    // Laplacian pyramid
    float4          *g_imageLaplaceMems[imageInNum][scaleDepth];
    float4          *g_imageWeightedLaplaceMems[imageInNum][scaleDepth];

    // Combined pyramid picture
    float4          *g_imageCombinedMems[scaleDepth];

    // Done
    float4          *g_imageOutMem;


    // DEBUGGING FLAGS

    bool dumpGrayscale          = false;
    bool dumpWeightmap          = false;
    bool dumpDownscaledImg      = false;
    bool dumpUpscaledImg        = false;
    bool dumpLaplaceImg         = false;
    bool dumpWeightDown         = false;
    bool dumpWeightedLaplace    = false;
    bool dumpCombined           = false;
    bool dumpCollapsed          = false;
/*
    bool dumpGrayscale          = true;
    bool dumpWeightmap          = true;
    bool dumpDownscaledImg      = true;
    bool dumpUpscaledImg        = true;
    bool dumpLaplaceImg         = true;
    bool dumpWeightDown         = true;
    bool dumpWeightedLaplace    = true;
    bool dumpCombined           = true;
    bool dumpCollapsed          = true;
*/    
//    bool quickEnfuse            = true;
    bool quickEnfuse            = false;
    bool verbose                = false;



	printCudaDeviceQuery();    


	fprintf(stdout, "-> Go CUDA!\n");

    fprintf(stdout, "\t * Information\n");
    fprintf(stdout, "\t\t-> Running in %s mode.\n", EMU?"emulation":"native");
    fprintf(stdout, "\t\t-> Got %i input images to process.\n", imageInNum);
    if (verbose) {
        fprintf(stdout, "\t\t-> Size of 'float4' is %i bytes.\n", sizeof(float4)); 
        fprintf(stdout, "\t\t-> Size of 'float3' is %i bytes.\n", sizeof(float3)); 
        fprintf(stdout, "\t\t-> Size of 'uchar4' is %i bytes.\n", sizeof(uchar4)); 
    }

    // Start timer for benchmark
    cuda_startTimer(&start, &end);



    if (quickEnfuse) {
        /*********************************************************************
         *                            QUICK ENFUSE                           *
         *********************************************************************/

        cudaArray *g_imageInMems[imageInNum];

       /*
        * For every image: Pre-process
        */
        for (int curImg = 0; curImg < imageInNum; curImg++) {

            fprintf(stdout, "\t * Pre-processing image #%i...\n", curImg);
            
            /*
             * Image upload
             */
            if (verbose) fprintf(stdout, "\t\t-> Uploading...\n");
            g_imageInMems[curImg] = uploadToArray(h_imageInMems[curImg], width, height);
            
            /*
             * Grayscaling the original image
             */
            if (verbose) fprintf(stdout, "\t\t-> Generating grayscale image...\n");
            float *g_imageGrayMem = genGrayscale(g_imageInMems[curImg], width, height);
            if (dumpGrayscale) writeGrayImage(g_imageGrayMem, width, height, "grayscale_%i_%i.bmp", curImg, 0);


            /*
             * Generating weightmap from grayscaled and original input image
             */

            if (verbose) fprintf(stdout, "\t\t-> Generating weightmap..\n");
            g_imageWeightMems[curImg] = genWeightmap(g_imageInMems[curImg], g_imageGrayMem, width, height);
            if (dumpWeightmap) writeGrayImage(g_imageWeightMems[curImg], width, height, "weightmap_%i_%i.bmp", curImg, 0);
            // free space: grayscale image
            cudaFree(g_imageGrayMem);
        }


        /*
         * THIS IS A SYNCHRONISATION POINT
         * NORMALISATION CAN ONLY BE DONE WHEN ALL WEIGHTMAPS ARE PRESENT
         */ 

        /*
         * Weightmap normalization
         */
        fprintf(stdout, "\t * Normalizing all image weightmaps.\n");
        normWeightmaps(g_imageWeightMems, imageInNum, width, height);


       /*
        * For every image: Process
        */
        for (int curImg = 0; curImg < imageInNum; curImg++) { 
            fprintf(stdout, "\t * Processing image #%i...\n", curImg);
            
            if (curImg == 0) {
                g_imageOutMem = genAppliedWeightmap(g_imageInMems[curImg], g_imageWeightMems[curImg], width, height);
            }
            else {
                g_imageOutMem = genSimpleEnfuse(g_imageOutMem, g_imageInMems[curImg], g_imageWeightMems[curImg], width, height);
            }

            // free space
            cudaFreeArray(g_imageInMems[curImg]);
            cudaFree(g_imageWeightMems[curImg]);
        }
            
    }
    else {
        /*********************************************************************
         *                         GOOD QUALITY ENFUSE                       *
         *********************************************************************/
    
    
       /*
        * For every image: Pre-process
        */
        for (int curImg = 0; curImg < imageInNum; curImg++) {

            fprintf(stdout, "\t * Pre-processing image #%i...\n", curImg);
            
            /*
             * Image upload
             */
            if (verbose) fprintf(stdout, "\t\t-> Uploading...\n");
            cudaArray *g_imageInMem = uploadToArray(h_imageInMems[curImg], width, height);
            
            /*
             * Grayscaling the original image
             */
            if (verbose) fprintf(stdout, "\t\t-> Generating grayscale image...\n");
            float *g_imageGrayMem = genGrayscale(g_imageInMem, width, height);
            if (dumpGrayscale) writeGrayImage(g_imageGrayMem, width, height, "grayscale_%i_%i.bmp", curImg, 0);


            /*
             * Generating weightmap from grayscaled and original input image
             */

            if (verbose) fprintf(stdout, "\t\t-> Generating weightmap..\n");
            g_imageWeightMems[curImg] = genWeightmap(g_imageInMem, g_imageGrayMem, width, height);
            if (dumpWeightmap) writeGrayImage(g_imageWeightMems[curImg], width, height, "weightmap_%i_%i.bmp", curImg, 0);
            // free space: grayscale image
            cudaFree(g_imageGrayMem);

            /*
             * Downscaling of the original image using Gauss
             */
            float4 *g_imageGaussDownMems[scaleDepth]; 
            if (verbose) fprintf(stdout, "\t\t-> Generating Gaussian downscale pyramid of the input image...\n\t\t\t");
            // NOTE: biggest scale image remains the same -> just copy
            if (verbose) fprintf(stdout, "#%i ", 0);
            g_imageGaussDownMems[0] = toFloat4(g_imageInMem, width, height);
            if (dumpDownscaledImg) writeImage(g_imageGaussDownMems[0], width, height, "scaleDown_%i_%i.bmp", curImg, 0);
            // free space: original input image
            cudaFreeArray(g_imageInMem);
            
            scale_width = width;
            scale_height = height;
            for (int j = 1; j < scaleDepth; j++) {
                 if (verbose) fprintf(stdout, "#%i ", j);
                 g_imageGaussDownMems[j] = genGaussDown(g_imageGaussDownMems[j-1], &scale_width, &scale_height);
                 if (dumpDownscaledImg) writeImage(g_imageGaussDownMems[j], scale_width, scale_height, "scaleDown_%i_%i.bmp", curImg, j);
            }
            if (verbose) fprintf(stdout, "\n");

            /*
             * Re-Upscaling of the previously downscaled images using Gauss
             */
            float4 *g_imageGaussUpMems[scaleDepth];
            if (verbose) fprintf(stdout, "\t\t-> Generating Gaussian re-upscale pyramid of the previously downscaled input images...\n\t\t\t");
            // NOTE: smallest scale image remains and is just copied by reference
            if (verbose) fprintf(stdout, "#%i ", scaleDepth - 1);
            g_imageGaussUpMems[scaleDepth - 1] =  g_imageGaussDownMems[scaleDepth - 1];
            if (dumpUpscaledImg) writeImage(g_imageGaussUpMems[scaleDepth - 1], scale_width, scale_height, "scaleUp_%i_%i.bmp", curImg, scaleDepth - 1);
            
            scale_width = floor(width / pow(2, scaleDepth-1));
            scale_height = floor(height / pow(2, scaleDepth-1));
            for (int j = scaleDepth - 2; j > -1; j--) {
                if (verbose) fprintf(stdout, "#%i ", j);
                g_imageGaussUpMems[j] = genGaussUp(g_imageGaussDownMems[j+1], &scale_width, &scale_height);
                if (dumpUpscaledImg) writeImage(g_imageGaussDownMems[j], scale_width, scale_height, "scaleUp_%i_%i.bmp", curImg, j);
            }
            if (verbose) fprintf(stdout, "\n");

            /*
             * Generating Laplacian pyramid as difference between downscaled and re-upscaled image
             */
             // TODO RAM usage could be minimized by processing each stage after another and not all at once
            if (verbose) fprintf(stdout, "\t\t-> Generating Laplacian pyramid as difference between downscaled and re-upscaled input images...\n\t\t\t");
            scale_width = width;
            scale_height = height;
            for (int j = 0; j < scaleDepth - 1; j++) {
                 if (verbose) fprintf(stdout, "#%i ", j);
                 g_imageLaplaceMems[curImg][j] = genLaplace(g_imageGaussDownMems[j], g_imageGaussUpMems[j], &scale_width, &scale_height);
                 if (dumpLaplaceImg) writeImage(g_imageLaplaceMems[curImg][j], scale_width*2, scale_height*2, "LP_%i_%i.bmp", curImg, j);
                 
                 // free space: down and re-upscaled image
                 cudaFree(g_imageGaussDownMems[j]);
                 cudaFree(g_imageGaussUpMems[j]);
            }

            // NOTE: The smallest image is the same in both gauss arrays so we use the same image for laplace
            // according to http://web.mit.edu/persci/people/adelson/pub_pdfs/spline83.pdf page 9
            if (verbose) fprintf(stdout, "#%i ", scaleDepth - 1);
            g_imageLaplaceMems[curImg][scaleDepth - 1] = g_imageGaussDownMems[scaleDepth - 1];
            if (dumpUpscaledImg) writeImage(g_imageLaplaceMems[curImg][scaleDepth - 1], scale_width, scale_height, "LP_%i_%i.bmp", curImg, scaleDepth - 1);
            // free space: down and re-upscaled image
    //        cudaFree(g_imageGaussDownMems[scaleDepth - 1]);  // this image is re-used in the laplacian pyramid!
            //cudaFree(g_imageGaussUpMems[scaleDepth - 1]);
            if (verbose) fprintf(stdout, "\n");
        }




        /*
         * THIS IS A SYNCHRONISATION POINT
         * NORMALISATION CAN ONLY BE DONE WHEN ALL WEIGHTMAPS ARE PRESENT
         */ 

        /*
         * Weightmap normalization
         */
        fprintf(stdout, "\t * Normalizing all image weightmaps.\n");
    //    normWeightmaps(g_imageWeightMems[0], g_imageWeightMems[1], width, height); // This works for 2 images only (but is faster!!)
        normWeightmaps(g_imageWeightMems, imageInNum, width, height);
    //    normWeightmaps(g_imageWeightMems, 2, width, height);
     
     
     
     
     
       /*
        * For every image: Process
        */
        for (int curImg = 0; curImg < imageInNum; curImg++) { 
            fprintf(stdout, "\t * Processing image #%i...\n", curImg);
            
            /*
             * Generating downscaled versions of the normalized weightmap using Gauss
             */
            float *g_imageWeightDownMem[scaleDepth];
            if (verbose) fprintf(stdout, "\t\t-> Generating Gaussian downscale pyramid of the normalised weightmap...\n\t\t\t");
            // NOTE: biggest scale weightmap image remains the same
            if (verbose) fprintf(stdout, "#%i ", 0);
            g_imageWeightDownMem[0] = g_imageWeightMems[curImg];
            if (dumpWeightDown) writeGrayImage(g_imageWeightDownMem[0], width, height, "weightNorm_%i_%i.bmp", curImg, 0);

            scale_width = width;
            scale_height = height;
            for (int j = 1; j < scaleDepth; j++) {
                if (verbose) fprintf(stdout, "#%i ", j);
                g_imageWeightDownMem[j] = genGaussDown_f(g_imageWeightDownMem[j-1], &scale_width, &scale_height);
                 //g_imageWeightDownMems[i][j] = genLinearDown_f(g_imageWeightDownMems[i][j-1], &scale_width, &scale_height);
                if (dumpWeightDown) writeGrayImage(g_imageWeightDownMem[j], scale_width, scale_height, "weightNorm_%i_%i.bmp", curImg, j);
            }
            if (verbose) fprintf(stdout, "\n");


            /*
             * For every image: Apply the weightmaps of each scale to the Laplacian pyramid of each scale
             */
            if (verbose) fprintf(stdout, "\t\t-> Generating weighted Laplace pyramid...\n\t\t\t");

            scale_width = width;
            scale_height = height;
            for (int j = 0; j < scaleDepth; j++) {
                if (verbose) fprintf(stdout, "#%i ", j);
                 g_imageWeightedLaplaceMems[curImg][j] = genWeightedLaplace(g_imageLaplaceMems[curImg][j], g_imageWeightDownMem[j], &scale_width, &scale_height);
                 if (dumpWeightedLaplace) writeImage(g_imageWeightedLaplaceMems[curImg][j], scale_width*2, scale_height*2, "weightedLP_%i_%i.bmp", curImg, j);

                 // free space: already applied downscaled weightmap sizes
                 cudaFree(g_imageWeightDownMem[j]);
            }
            if (verbose) fprintf(stdout, "\n");
        
        }

        
        
        /*
         * THIS IS A SYNCHRONISATION POINT
         * ADDING THE SCALE STAGES OF EACH IMAGE CAN ONLY BE DONE IF ALL IMAGES HAVE BEEN PROCESSED
         */ 

        /*
         * For every image: Add up the weighted Laplacian pyramid at every scale
         */
        if (verbose) fprintf(stdout, "\t * Generating sum of the weighted Laplace pyramid at every scale of every image.\n");
        scale_width = width;
        scale_height = height;
        if (verbose) fprintf(stdout, "\t\t-> Scales...\n\t\t\t");
        for (int j = 0; j < scaleDepth; j++) {

            // This is for the first two images
            if (verbose) fprintf(stdout, "#%i ", j);
            g_imageCombinedMems[j] = genAdded(g_imageWeightedLaplaceMems[0][j], g_imageWeightedLaplaceMems[1][j], &scale_width, &scale_height);
            // free space: already added scales
            cudaFree(g_imageWeightedLaplaceMems[0][j]);
            cudaFree(g_imageWeightedLaplaceMems[1][j]);

            // This is for images three and following  // FIXME
            for(int i = 2; i < imageInNum; i++) {
                scale_width *= 2;
                scale_height *= 2;
                g_imageCombinedMems[j] = genAdded(g_imageCombinedMems[j], g_imageWeightedLaplaceMems[i][j], &scale_width, &scale_height);
                // free space: already added scales
                cudaFree(g_imageWeightedLaplaceMems[i][j]);
            }

            if (dumpCombined) writeImage(g_imageCombinedMems[j], scale_width*2, scale_height*2, "sumWeightedLP_%i_%i.bmp", 0, j);

        }
        if (verbose) fprintf(stdout, "\n");
        


        /*
         * For every scale: Add up the smaller scale with the current scale upwards
         */
        if (verbose) fprintf(stdout, "\t * Collapsing weighted Laplace pyramid.\n");
        scale_width = floor(width / pow(2, scaleDepth-1));
        scale_height = floor(height / pow(2, scaleDepth-1));    
        if (dumpCollapsed) writeImage(g_imageCombinedMems[scaleDepth - 1], scale_width, scale_height, "collapsed_%i_%i.bmp", 0, scaleDepth - 1);
        if (verbose) fprintf(stdout, "\t\t-> Scales...\n\t\t\t");
        for (int i = scaleDepth - 1; i > 0; i--) {
            if (verbose) fprintf(stdout, "#%i ", i-1);
            g_imageCombinedMems[i-1] = genCollapsed(g_imageCombinedMems[i-1], g_imageCombinedMems[i], &scale_width, &scale_height);
            if (dumpCollapsed) writeImage(g_imageCombinedMems[i-1], scale_width, scale_height, "collapsed_%i_%i.bmp", 0, i-1);

            // free space: already added scales
            cudaFree(g_imageCombinedMems[i]);
        }
        if (verbose) fprintf(stdout, "\n");


        // Assigning pointer to final output image
        g_imageOutMem = g_imageCombinedMems[0];


        /*
         * DONE
         */

    }


    cuda_stopTimer(&start, &end, &execution_time);
    fprintf(stdout, "\t* Execution took %.5f ms\n", execution_time);
   	fprintf(stdout, "-> End CUDA.\n");



    
    //writeImage(g_imageOutMem, width, height, "out.bmp", 0, 0);

    float4 *h_imageOutMem = cuda_download<float4>(g_imageOutMem, width, height);
    cudaFree(g_imageOutMem);
    return h_imageOutMem;
}








