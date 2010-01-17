/*
 * main.cpp
 *
 * Sample usage of the cuda enfuse implementation.
 *
 *
 * Authors: Florian Löffler <flo@geek-blog.de>
 * Licence: LGPLv3
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "bmploader.h"

#include "cuda_enfuse.h"
#include "cuda_wrappers.h"



int main(int argc, char** argv) {

    // Input images
    const char  imageInPaths[][50] = {
                      "input_1.bmp", "input_2.bmp",  
    };
    int         imageInNum = sizeof(imageInPaths) / sizeof(imageInPaths[0]);
    uchar4      *imageInMems[imageInNum];

    // Output image
    const char  *imageOutPath = "output.bmp";
    
    
    // image information
    int         width, height;


    //printCudaDeviceQuery(); exit(0);


    fprintf(stdout, "-> Startup.\n");
    cuda_init(argc, argv);


    // Loading input images
    for(int i = 0; i < imageInNum; i++) {
        fprintf(stdout, "\t* Loading image '%s'.\n", imageInPaths[i]);
    	LoadBMPFile(&imageInMems[i], &width, &height, imageInPaths[i]);
    }


    // Enfusing the input images to one output image
    float4 *imageOutMem = cuda_enfuse(imageInMems, width, height, imageInNum, 7);


    // Writing output image
    fprintf(stdout, "\t* Writing image '%s'.\n", imageOutPath);
    SaveBMPFile(&imageOutMem, width, height, imageOutPath);

    fprintf(stdout, "-> End.\n");
    
    // cleaning up
    free(imageOutMem);
    //cuda_exit(argc, argv);
}




