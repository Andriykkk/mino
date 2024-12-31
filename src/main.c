#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "matrix/matrix_g.h"

// TODO: make generator for a callee function, they will be hardcoded, there will be separate file with functions for generating different operations, mul, add, etc, and also calle, they will append to matrix code
// TODO: write a function to generate function for mul, div, add, sub for matrices of different size

// Premature optimisations
// TODO: add callee functions that will check type of matrix and call corresponding function
// TODO: add dev mode that will check matrices sizes and print error if they are not compatible
// TODO: add pro mode that will remove types from matrices and you will should add types to functions mannualy
// TODO: end library and start optimising matrices, use vectors, gpu, etc.

int main()
{
	Matrix_f32 *matrix = create_matrix_f32((int[]){3, 3, 3}, 3, 2);
	printf("%d\n", matrix->total_elements);
	print_matrix_f32(matrix);

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