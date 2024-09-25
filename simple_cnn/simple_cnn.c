#include "simple_cnn.h"

int max(int a, int b)
{
	return a > b ? a : b;
}

int relu(int x)
{
	return max(0, x);
}

void convolution_max_pool(IMAGE input, CONV_WEIGHT_MATRIX weights, CONV_BIAS_MATRIX biases, CONV_MAX_POOL_OUTPUT_MATRIX output)
{
	for (int k = 0; k < TOTAL_KERNELS; k++)
	{
		int convolution_output[CONV_OUTPUT_SIZE][CONV_OUTPUT_SIZE];
		for (int j = 0; j < CONV_OUTPUT_SIZE; j++)
		{
			for (int i = 0; i < CONV_OUTPUT_SIZE; i++)
			{
				int _sum = 0;
				for (int y = 0; y < CONV_KERNEL_SIZE; y++)
				{
					for (int x = 0; x < CONV_KERNEL_SIZE; x++)
					{
						_sum += input[j + y][i + x] * weights[k][y][x];
					}
				}
				convolution_output[j][i] = relu(_sum + biases[k]);
				
			}
		}
		max_pool(k, convolution_output, output);
	}
}

void max_pool(int k, CONV_OUTPUT_MATRIX input, CONV_MAX_POOL_OUTPUT_MATRIX output)
{
	for (int j = 0; j < MAX_POOL_OUTPUT_SIZE; j++)
	{
		for (int i = 0; i < MAX_POOL_OUTPUT_SIZE; i++)
		{
			int _max = 0;
			for (int y = 0; y < MAX_POOL_WINDOW_SIZE; y++)
			{
				for (int x = 0; x < MAX_POOL_WINDOW_SIZE; x++)
				{
					_max = max(_max, input[j * MAX_POOL_STRIDE + y][i * MAX_POOL_STRIDE + x]);
				}
			}
			output[k][j][i] = _max;
		}
	}
}
