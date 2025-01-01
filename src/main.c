#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "matrix/matrix_g.h"

// TODO: return link to output from forward_linear_layer
// TODO: make softmax layer, also dont forget about substracting max value from matrix
// TODO: make so that you allocate very big matrix as output and after each forward it not allocated and just resizing, working as arena, but this is only for not learning, for learning there should be output matrix
// TODO: add batches, simple batches that if one matrix have one additional dimension, then first dimension count as batch and just repeat loop as much as amount of matches
// TODO: thinks about backpropagation (add backpropagation matrix to each matrix, add bool parameter if need back propagations to matrix or layer).
// TODO: add tests for everything as i will try to optimise it

// Premature optimisations
// TODO: remove allocation in transpose
// TODO: write a function to generate function for mul, div, add, sub for matrices of different size
// TODO: add dev mode that will check matrices sizes and print error if they are not compatible
// TODO: add pro mode that will remove types from matrices and you will should add types to functions mannualy
// TODO: end library and start optimising matrices, use vectors, gpu, etc.

void multiply_matrices_2d_f32(Matrix_f32 *matrix_a, Matrix_f32 *matrix_b, Matrix_f32 *result_matrix)
{
	if (matrix_a->num_dims != matrix_b->num_dims || matrix_a->dims[0] != matrix_b->dims[1])
	{
		printf("Matrices are not compatible for multiplication\n");
		return;
	}

	for (int i = 0; i < matrix_a->dims[0]; i++)
	{
		for (int j = 0; j < matrix_b->dims[1]; j++)
		{
			for (int k = 0; k < matrix_a->dims[1]; k++)
			{
				result_matrix->data[i * matrix_b->dims[1] + j] += matrix_a->data[i * matrix_a->dims[1] + k] * matrix_b->data[k * matrix_b->dims[1] + j];
			}
		}
	}
}

// void multiply_matrices_broadcast_f32(Matrix_f32 *matrix_a, Matrix_f32 *matrix_b, Matrix_f32 *result_matrix)
// {
// 	if (matrix_a->dims[0] % matrix_b->dims[0] != 0 || matrix_a->dims[1] % matrix_b->dims[1] != 0 || matrix_a->num_dims != matrix_b->num_dims || matrix_a->dims[0] != matrix_b->dims[1])
// 	{
// 		printf("Matrices are not compatible for multiplication\n");
// 		return;
// 	}

// 	int broadcast_1 = matrix_a->dims[0] / matrix_b->dims[0];
// 	int broadcast_1 = matrix_a->dims[1] / matrix_b->dims[1];

// 	for (int i = 0; i < matrix_a->dims[0]; i++)
// 	{
// 		for (int j = 0; j < matrix_b->dims[1] ; j++)
// 		{
// 			for (int k = 0; k < matrix_a->dims[1]; k++)
// 			{
// 				result_matrix->data[i * matrix_b->dims[1] + j] += matrix_a->data[i * matrix_a->dims[1] + k] * matrix_b->data[k * matrix_b->dims[1] + j];
// 			}
// 		}
// 	}
// }

void transpose_matrix_f32(Matrix_f32 *matrix)
{
	if (matrix->num_dims != 2)
	{
		printf("Transpose is only supported for 2D matrices.\n");
		return;
	}

	int rows = matrix->dims[0];
	int cols = matrix->dims[1];
	int total_elements = matrix->total_elements;

	float *transposed_data = (float *)malloc(total_elements * sizeof(float));

	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			int original_idx = i * cols + j;
			int transposed_idx = j * rows + i;
			transposed_data[transposed_idx] = matrix->data[original_idx];
		}
	}

	for (int i = 0; i < total_elements; i++)
	{
		matrix->data[i] = transposed_data[i];
	}

	matrix->dims[0] = cols;
	matrix->dims[1] = rows;

	free(transposed_data);
}

void add_matrices_2d_f32(Matrix_f32 *matrix_a, Matrix_f32 *matrix_b, Matrix_f32 *result_matrix)
{
	int rows = matrix_a->dims[0];
	int cols = matrix_a->dims[1];

	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			result_matrix->data[i * cols + j] = matrix_a->data[i * cols + j] + matrix_b->data[i * cols + j];
		}
	}
}

void add_matrices_broadcast_2d_f32(Matrix_f32 *matrix_a, Matrix_f32 *matrix_b, Matrix_f32 *result_matrix)
{
	if (matrix_a->dims[0] % matrix_b->dims[0] != 0 || matrix_a->dims[1] % matrix_b->dims[1] != 0)
	{
		printf("Matrices are not compatible for addition\n");
		return;
	}

	int rows = matrix_a->dims[0];
	int cols = matrix_a->dims[1];

	int rows_b = matrix_b->dims[0];
	int cols_b = matrix_b->dims[1];

	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			int b_row_index = i % rows_b;
			int b_col_index = j % cols_b;

			result_matrix->data[i * cols + j] = matrix_a->data[i * cols + j] + matrix_b->data[b_row_index * cols + b_col_index];
		}
	}
}

typedef struct
{
	Matrix_f32 *weights;
	Matrix_f32 *bias;
} Linear_Layer_f32;

void fill_random_matrix_f32(Matrix_f32 *matrix, int min, int max)
{
	for (int i = 0; i < matrix->total_elements; i++)
	{
		matrix->data[i] = (float)rand() / RAND_MAX * (max - min) + min;
	}
}

Linear_Layer_f32 *init_linear_layer_f32(int input_size, int output_size)
{
	Linear_Layer_f32 *layer = (Linear_Layer_f32 *)malloc(sizeof(Linear_Layer_f32));
	Matrix_f32 *weights = (Matrix_f32 *)create_matrix_f32((int[]){input_size, output_size}, 2, 0);
	Matrix_f32 *bias = (Matrix_f32 *)create_matrix_f32((int[]){1, output_size}, 2, 0);
	fill_random_matrix_f32(weights, -1, 1);
	fill_random_matrix_f32(bias, -1, 1);
	layer->weights = weights;
	layer->bias = bias;

	return layer;
}

void forward_linear_layer_f32(Linear_Layer_f32 *layer, Matrix_f32 *input, Matrix_f32 *output)
{
	multiply_matrices_2d_f32(input, layer->weights, output);
	add_matrices_2d_f32(output, layer->bias, output);
}

void relu_matrix_f32(Matrix_f32 *matrix)
{
	for (int i = 0; i < matrix->total_elements; i++)
	{
		if (matrix->data[i] < 0)
		{
			matrix->data[i] = 0;
		}
	}
}

int main()
{
	Linear_Layer_f32 *layer = init_linear_layer_f32(3, 4);
	Matrix_f32 *input = (Matrix_f32 *)create_matrix_f32((int[]){4, 3}, 2, 2);
	Matrix_f32 *input2 = (Matrix_f32 *)create_matrix_f32((int[]){3, 4}, 2, 2);
	Matrix_f32 *output = (Matrix_f32 *)create_matrix_f32((int[]){4, 4}, 2, 0);
	forward_linear_layer_f32(layer, input, output);

	print_matrix_f32(output);

	return 0;
}