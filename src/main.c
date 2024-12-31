#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "matrix/matrix_g.h"

// TODO: write a function to generate function for mul, div, add, sub for matrices of different size

// TODO: add broadcasting as separate functions with broadcast, but this is later, right now usual multiplication, addition

// Premature optimisations
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

	for (int i = 0; i < rows; i++)
	{
		for (int j = i + 1; j < cols; j++)
		{
			int idx1 = i * cols + j;
			int idx2 = j * rows + i;

			float temp = matrix->data[idx1];
			matrix->data[idx1] = matrix->data[idx2];
			matrix->data[idx2] = temp;
		}
	}
}

int main()
{
	Matrix_f32 *matrix_b = (Matrix_f32 *)create_matrix_f32((int[]){3, 3}, 2, 1);

	matrix_b->data = (float[]){1, 2, 3, 4, 5, 6, 7, 8, 9};

	print_matrix_f32(matrix_b);
	transpose_matrix_f32(matrix_b);
	printf("sdfsd\n ");
	print_matrix_f32(matrix_b);
	// multiply_matrices_f32(matrix_a, matrix_b, matrix_c);

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