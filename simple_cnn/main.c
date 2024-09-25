#include <stdio.h>
#include <stdlib.h>
#include "lodepng/lodepng.h"
#include "simple_cnn.h"

#pragma GCC diagnostic ignored "-Wincompatible-pointer-types"

void load_parameters(char* file_name, int size, char* output)
{
	FILE* file = fopen(file_name, "r");
	if (file == NULL) {
		printf("Error opening %s\n", file_name);
		exit(1);
	}

	char value;
	int read = 0;
	while (fscanf(file, "%hhd", &value) && read < size) {
		output[read++] = value;
	}

	if (read < size) {
		printf("Error: Too few parameters in %s, expected %d, found %d.\n", file_name, size, read);
		exit(1);
	}

    fclose(file);
}

void load_image(char* file_name, IMAGE output)
{
	unsigned char *image;
    unsigned int width, height;
    int error = lodepng_decode32_file(&image, &width, &height, file_name);
    if(error) {
        printf("Error %u: %s\n", error, lodepng_error_text(error));
        exit(1);
    }

    if (width != INPUT_IMAGE_SIZE || height != INPUT_IMAGE_SIZE) {
    	printf("Error: Image size (%d, %d) does not match expected size (%d, %d).\n", width, height, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE);
    	exit(1);
    }
    
    for(unsigned int y=0; y<height; y++) {
        for(unsigned int x=0; x<width; x++) {
            int index = (y * width + x);
            output[y][x] = image[index * 4];
        }
    }

    free(image);
}

int main(int argc, char **argv)
{
	if (argc != 2) {
		printf("Usage: %s [IMAGE_FILE_PATH]\n", argv[0]);
		return 0;
	}

	IMAGE image = {{0}};
	load_image(argv[1], image);

	CONV_WEIGHT_MATRIX conv_weights = {{{0}}};
	load_parameters("./parameters/conv_weights.txt", TOTAL_CONV_WEIGHTS, conv_weights);

	CONV_BIAS_MATRIX conv_biases = {0};
	load_parameters("./parameters/conv_biases.txt", TOTAL_KERNELS, conv_biases);

	CONV_MAX_POOL_OUTPUT_MATRIX conv_pool_output = {{{0}}};
	convolution_max_pool(image, conv_weights, conv_biases, conv_pool_output);

    printf("Conv Max Pool Output:\n");
	for(int k=0; k<TOTAL_KERNELS; k++) {
	    for(int j=0; j<MAX_POOL_OUTPUT_SIZE; j++) {
	        for(int i=0; i<MAX_POOL_OUTPUT_SIZE; i++) {
	            printf("%d ", conv_pool_output[k][j][i]);
	        }
	        printf("\n");
	    }
	    printf("\n");
	}

	return 0;
}
