#include <stdio.h>
#include <stdlib.h>
#include "bmploader.h"

#ifdef _WIN32
#   pragma warning( disable : 4996 ) // disable deprecated warning 
#endif

#pragma pack(1)

typedef struct {
    short type;
    int size;
    short reserved1;
    short reserved2;
    int offset;
} BMPHeader;

typedef struct {
    int size;
    int width;
    int height;
    short planes;
    short bitsPerPixel;
    unsigned compression;
    unsigned imageSize;
    int xPelsPerMeter;
    int yPelsPerMeter;
    int clrUsed;
    int clrImportant;
} BMPInfoHeader;





extern "C" void LoadBMPFile(uchar4 **dst, int *width, int *height, const char *name) {
    BMPHeader hdr;
    BMPInfoHeader infoHdr;
    int x, y;

    FILE *fd;


//    printf("Loading %s...\n", name);
    if (sizeof(uchar4) != 4) {
        printf("***Bad uchar4 size***\n");
        exit(EXIT_FAILURE);
    }

    if (!(fd = fopen(name,"rb"))) {
        printf("***BMP load error: file access denied***\n");
        exit(EXIT_FAILURE);
    }

    if (fread(&hdr, sizeof(hdr), 1, fd) != 1) {
        printf("***BMP load error: cannot read from file***\n");
        exit(EXIT_FAILURE);
    }
    if (hdr.type != 0x4D42) {
        printf("***BMP load error: bad file format***\n");
        exit(EXIT_FAILURE);
    }

    if (fread(&infoHdr, sizeof(infoHdr), 1, fd) != 1) {
        printf("***BMP load error: cannot read from file***\n");
        exit(EXIT_FAILURE);
    }
    if (infoHdr.bitsPerPixel != 24) {
        printf("***BMP load error: invalid color depth***\n");
        exit(EXIT_FAILURE);
    }

    if (infoHdr.compression) {
        printf("***BMP load error: compressed image***\n");
        exit(EXIT_FAILURE);
    }

    *width  = infoHdr.width;
    *height = infoHdr.height;
    *dst    = (uchar4 *)malloc(*width * *height * 4);

//    printf("BMP width: %u\n", infoHdr.width);
//    printf("BMP height: %u\n", infoHdr.height);

    fseek(fd, hdr.offset - sizeof(hdr) - sizeof(infoHdr), SEEK_CUR);

    for (y=0; y < infoHdr.height; y++) {
        for (x=0; x < infoHdr.width; x++) {
            (*dst)[(y * infoHdr.width + x)].z = fgetc(fd);
            (*dst)[(y * infoHdr.width + x)].y = fgetc(fd);
            (*dst)[(y * infoHdr.width + x)].x = fgetc(fd);
        }

        for (x=0; x < (4 - (3 * infoHdr.width) % 4) % 4; x++)
            fgetc(fd);
    }


    if (ferror(fd)) {
        printf("***Unknown BMP load error.***\n");
        free(*dst);
        exit(EXIT_FAILURE);
    } else
//        printf("BMP file loaded successfully!\n");

    fclose(fd);
}


extern "C" void SaveBMPFile(float4 **src, int width, int height, const char *name) {
    BMPHeader hdr;
    BMPInfoHeader infoHdr;
    int x, y, tmp_width, tmp;

    FILE *fd;

    hdr.type = 0x4D42;
    // width must be a multiple of 4
    tmp_width = 3*width;
    tmp = 0;
    if ((tmp=tmp_width%4) != 0) tmp_width += (4-tmp);
    hdr.size = sizeof(hdr) + sizeof(infoHdr) + height * tmp_width;
    hdr.reserved1 = 0x0;
    hdr.reserved2 = 0x0;
    hdr.offset = sizeof(hdr)+sizeof(infoHdr);

    infoHdr.size = sizeof(infoHdr);
    infoHdr.width = width;
    infoHdr.height = height;
    infoHdr.planes = 1;
    infoHdr.bitsPerPixel = 24;
    infoHdr.compression = 0;
    infoHdr.imageSize = height * tmp_width;
    infoHdr.xPelsPerMeter = 0x130B0000;
    infoHdr.yPelsPerMeter = 0x130B0000;
    infoHdr.clrUsed = 0;
    infoHdr.clrImportant = 0;

//    printf("Saving %s...\n", name);
    if (sizeof(uchar4) != 4) {
        printf("***Bad uchar4 size***\n");
        exit(EXIT_FAILURE);
    }

    if (!(fd = fopen(name, "w"))) {
        printf("***BMP write error: file access denied***\n");
        exit(EXIT_FAILURE);
    }

    if (fwrite(&hdr, sizeof(hdr), 1, fd) != 1) {
        printf("***BMP write error: cannot write to file***\n");
        exit(EXIT_FAILURE);
    }
    if (fwrite(&infoHdr, sizeof(infoHdr), 1, fd) != 1) {
        printf("***BMP write error: cannot write to file***\n");
        exit(EXIT_FAILURE);
    }
/*
    printf("BMP width: %u\n", width);
    printf("BMP height: %u\n", height);
    
    float min_value_x = 0;
    float min_value_y = 0;
    float min_value_z = 0;

    float max_value_x = 0;
    float max_value_y = 0;
    float max_value_z = 0;
    for (y=0; y < infoHdr.height; y++) {
        for (x=0; x < infoHdr.width; x++) {

            if ( min_value_x > (*src)[y * width + x].x ) {
                min_value_x = (*src)[y * width + x].x;
            }
              
            if ( min_value_y > (*src)[y * width + x].y ) {
                min_value_y = (*src)[y * width + x].y;
            }
              
            if ( min_value_z > (*src)[y * width + x].z ) {
                min_value_z = (*src)[y * width + x].z;
            }
                
            if ( max_value_x < (*src)[y * width + x].x ) {
                max_value_x = (*src)[y * width + x].x;
            }
              
            if ( max_value_y < (*src)[y * width + x].y ) {
                max_value_y = (*src)[y * width + x].y;
            }
              
            if ( max_value_z < (*src)[y * width + x].z ) {
                max_value_z = (*src)[y * width + x].z;
            }

        }
    }
    printf("Min. values: (%f, %f, %f)\n", min_value_x, min_value_y, min_value_z);
    printf("Max. values: (%f, %f, %f)\n", max_value_x, max_value_y, max_value_z);
*/
    for (y=0; y < infoHdr.height; y++) {
        for (x=0; x < infoHdr.width; x++) {

//            unsigned char value_x = (unsigned char)(255 * ((*src)[y * width + x].x - min_value_x)/2);
//            unsigned char value_y = (unsigned char)(255 * ((*src)[y * width + x].y - min_value_y)/2);
//            unsigned char value_z = (unsigned char)(255 * ((*src)[y * width + x].z - min_value_z)/2);

            unsigned char value_x = (unsigned char)(255 * (*src)[y * width + x].x);
            unsigned char value_y = (unsigned char)(255 * (*src)[y * width + x].y);
            unsigned char value_z = (unsigned char)(255 * (*src)[y * width + x].z);
            
            fputc(value_z, fd);
            fputc(value_y, fd);
            fputc(value_x, fd);
        }

        for (x=0; x < (4 - (3 * width) % 4) % 4; x++)
            fputc(0x0, fd);
    }


    if (ferror(fd)) {
        printf("***Unknown BMP save error.***\n");
        exit(EXIT_FAILURE);
    } else
//        printf("BMP file saved successfully!\n");

    fclose(fd);
}


extern "C" void SaveGrayBMPFile(float **src, int width, int height, const char *name) {
    BMPHeader hdr;
    BMPInfoHeader infoHdr;
    int x, y, tmp_width, tmp;

    FILE *fd;

    hdr.type = 0x4D42;
    // width must be a multiple of 4
    tmp_width = 3*width;
    tmp = 0;
    if ((tmp=tmp_width%4) != 0) tmp_width += (4-tmp);
    hdr.size = sizeof(hdr) + sizeof(infoHdr) + height * tmp_width;
    hdr.reserved1 = 0x0;
    hdr.reserved2 = 0x0;
    hdr.offset = sizeof(hdr)+sizeof(infoHdr);

    infoHdr.size = sizeof(infoHdr);
    infoHdr.width = width;
    infoHdr.height = height;
    infoHdr.planes = 1;
    infoHdr.bitsPerPixel = 24;
    infoHdr.compression = 0;
    infoHdr.imageSize = height * tmp_width;
    infoHdr.xPelsPerMeter = 0x130B0000;
    infoHdr.yPelsPerMeter = 0x130B0000;
    infoHdr.clrUsed = 0;
    infoHdr.clrImportant = 0;

//    printf("Saving %s...\n", name);
    if (sizeof(uchar4) != 4) {
        printf("***Bad uchar4 size***\n");
        exit(EXIT_FAILURE);
    }

    if (!(fd = fopen(name, "w"))) {
        printf("***BMP write error: file access denied***\n");
        exit(EXIT_FAILURE);
    }

    if (fwrite(&hdr, sizeof(hdr), 1, fd) != 1) {
        printf("***BMP write error: cannot write to file***\n");
        exit(EXIT_FAILURE);
    }
    if (fwrite(&infoHdr, sizeof(infoHdr), 1, fd) != 1) {
        printf("***BMP write error: cannot write to file***\n");
        exit(EXIT_FAILURE);
    }

/*
    float min_value = 0;

    float max_value = 0;
    for (y=0; y < infoHdr.height; y++) {
        for (x=0; x < infoHdr.width; x++) {

            if ( min_value > (*src)[y * width + x] ) {
                min_value = (*src)[y * width + x];
            }
              
                
            if ( max_value < (*src)[y * width + x] ) {
                max_value = (*src)[y * width + x];
            }
              

        }
    }
    printf("Min. value: %f\n", min_value);
    printf("Max. value: %f\n", max_value);
*/

//    printf("BMP width: %u\n", width);
//    printf("BMP height: %u\n", height);

    for (y=0; y < infoHdr.height; y++) {
        for (x=0; x < infoHdr.width; x++) {
            
            unsigned char value = (unsigned char)(255 * (*src)[y * width + x]);
            
//printf("%f\t", (*src)[y * width + x]);

            fputc( value, fd);
            fputc( value, fd);
            fputc( value, fd);
        }

        for (x=0; x < (4 - (3 * width) % 4) % 4; x++)
            fputc(0x0, fd);
    }


    if (ferror(fd)) {
        printf("***Unknown BMP save error.***\n");
        exit(EXIT_FAILURE);
    } else
//        printf("BMP file saved successfully!\n");

    fclose(fd);
}



