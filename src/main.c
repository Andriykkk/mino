#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "matrix/matrix_g.h"

// TODO: add broadcasting as separate functions with broadcast, but this is later, right now usual multiplication, addition

// Premature optimisations
// TODO: remove allocation in transpose
// TODO: write a function to generate function for mul, div, add, sub for matrices of different size
// TODO: add dev mode that will check matrices sizes and print error if they are not compatible
// TODO: add pro mode that will remove types from matrices and you will should add types to functions mannualy
// TODO: end library and start optimising matrices, use vectors, gpu, etc.

void multiply_matrices_f32(Matrix_f32 *matrix_a, Matrix_f32 *matrix_b, Matrix_f32 *result_matrix)
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

void add_matrices_f32(Matrix_f32 *matrix_a, Matrix_f32 *matrix_b, Matrix_f32 *result_matrix)
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

void add_matrices_broadcast_f32(Matrix_f32 *matrix_a, Matrix_f32 *matrix_b, Matrix_f32 *result_matrix)
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

int main()
{
	Matrix_f32 *m_inputs = (Matrix_f32 *)create_matrix_f32((int[]){3, 4}, 2, 0);
	Matrix_f32 *m_weights = (Matrix_f32 *)create_matrix_f32((int[]){3, 4}, 2, 0);
	Matrix_f32 *m_bias = (Matrix_f32 *)create_matrix_f32((int[]){1, 3}, 2, 0);
	Matrix_f32 *m_result = (Matrix_f32 *)create_matrix_f32((int[]){3, 3}, 2, 0);

	m_inputs->data = (float[]){1, 2, 3, 2.5, 2, 5, -1, 2, -1.5, 2.7, 3.3, -0.8};
	m_weights->data = (float[]){0.2, 0.8, -0.5, 1.0, 0.5, -0.91, 0.26, -0.5, -0.26, -0.27, 0.17, 0.87};
	m_bias->data = (float[]){2.0, 3.0, 0.5};
	transpose_matrix_f32(m_weights);
	print_matrix_f32(m_inputs);
	printf("\n");
	print_matrix_f32(m_weights);

	multiply_matrices_f32(m_inputs, m_weights, m_result);
	printf("\n");
	print_matrix_f32(m_result);

	add_matrices_broadcast_f32(m_result, m_bias, m_result);
	printf("\n");
	print_matrix_f32(m_result);

	float inputs[4] = {1, 2, 3, 2.5};
	float weights[3][4] = {
		{0.2, 0.8, -0.5, 1.0},
		{0.5, -0.91, 0.26, -0.5},
		{-0.26, -0.27, 0.17, 0.87}};
	float bias[3] = {2.0, 3.0, 0.5};

	float output[3] = {0, 0, 0};

	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 4; j++)
		{
			output[i] += weights[i][j] * inputs[j];
		}

		output[i] += bias[i];
	}

	for (int i = 0; i < 3; i++)
	{
		printf("%f\n", output[i]);
	}

	return 0;
}