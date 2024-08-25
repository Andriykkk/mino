#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include "mat/mat.h"
#include "macros/error.h"

#define BUFFER_SIZE 10024

struct Num
{
	int label;
	float data[784];
};
float eps = 0.01;

int load_dataset(char *filename, struct Num **numbers, int *numbers_count)
{
	char buffer[BUFFER_SIZE];
	FILE *file;

	int t_count = 0;
	int row = 0;
	int size = 1000;

	// allocate memory
	*numbers = (struct Num *)(malloc(sizeof(struct Num) * size));
	if (*numbers == NULL)
	{
		printf("Failed to allocate memory for numbers\n");
		return 1;
	}

	// Open the file
	file = fopen(filename, "r");
	if (file == NULL)
	{
		printf("Failed to open file\n");
		return 1;
	}

	// Read the file
	fgets(buffer, BUFFER_SIZE, file);

	while (fgets(buffer, BUFFER_SIZE, file) != NULL)
	{
		// allocate more memory if needed
		if (row >= size)
		{
			size += 1000;
			*numbers = (struct Num *)(realloc(*numbers, sizeof(struct Num) * size));
			if (*numbers == NULL)
			{
				printf("Failed to allocate memory for numbers\n");
				return 1;
			}
		}

		char *token = strtok(buffer, ",");

		t_count = 0;

		(*numbers)[row].label = atoi(token);

		while ((token = strtok(NULL, ",")) != NULL)
		{
			(*numbers)[row].data[t_count++] = atof(token) / 255.0;
		}

		row += 1;
		// if (row >= 30)
		// {
		// 	break;
		// }
	}

	*numbers_count = row;
	printf("Loaded %d numbers\n", *numbers_count);

	// Close the file
	fclose(file);

	return 0;
}

void print_number(struct Num *number)
{
	float threshold = 0.2;

	for (int i = 0; i < 28; i++)
	{
		printf("\n");
		for (int j = 0; j < 28; j++)
		{
			if (number->data[i * 28 + j] > threshold)
			{
				printf("#");
			}
			else
			{
				printf("_");
			}
		}
	}

	printf("\nLabel: %d", number->label);
}

void forward_pass(struct Mat *mat_l_1, struct Mat *mat_w_1, struct Mat *mat_b_1, struct Mat *mat_l_2, struct Mat *mat_w_2, struct Mat *mat_b_2, struct Mat *mat_l_3)
{
	mul_mats(mat_l_2, mat_l_1, mat_w_1);
	add_mats(mat_l_2, mat_l_2, mat_b_1);

	mul_mats(mat_l_3, mat_l_2, mat_w_2);
	add_mats(mat_l_3, mat_l_3, mat_b_2);

	// mat_softmax(mat_l_3);
}

int get_random(int min, int max)
{
	return rand() % (max - min + 1) + min;
}

float calc_square_error(struct Mat *mat_result, struct Mat *mat_output)
{
	float sum = 0.0;
	for (int k = 0; k < mat_result->cols; k++)
	{
		float diff = mat_output->data[k] - mat_result->data[k];
		sum += diff * diff;
	}

	return sum;
}

void calc_forw_back_paths(struct Mat *mat_result, struct Mat *mat_l_1, struct Mat *mat_w_1, struct Mat *mat_b_1, struct Mat *mat_l_2, struct Mat *mat_w_2, struct Mat *mat_b_2, struct Mat *mat_l_3, struct Mat *mat_change)
{
	int tmp_size = (mat_change->rows * mat_change->cols) / 10;
	if (tmp_size < 1)
	{
		tmp_size = 1;
	}

	struct Mat *mat_tmp = create_mat(2, tmp_size);

	forward_pass(mat_l_1, mat_w_1, mat_b_1, mat_l_2, mat_w_2, mat_b_2, mat_l_3);
	float sum1 = calc_square_error(mat_result, mat_l_3);

	for (int k = 0; k < tmp_size; k++)
	{
		int rand = get_random(0, mat_change->cols * mat_change->rows);
		mat_tmp->data[k] = rand;
		mat_tmp->data[k + tmp_size] = mat_change->data[rand];
		mat_change->data[rand] += eps;
	}

	printf("Error: %f\n", sum1);

	forward_pass(mat_l_1, mat_w_1, mat_b_1, mat_l_2, mat_w_2, mat_b_2, mat_l_3);

	float sum2 = calc_square_error(mat_result, mat_l_3);

	if (sum1 < sum2)
	{
		for (int k = 0; k < tmp_size; k++)
		{
			int pos = (int)mat_tmp->data[k];
			mat_change->data[pos] = mat_tmp->data[k + tmp_size] - eps;
		}
	}

	// printf("Error weight: %f\n", sum2);
}

void calc_error(struct Mat *mat_result, struct Mat *mat_l_1, struct Mat *mat_w_1, struct Mat *mat_b_1, struct Mat *mat_l_2, struct Mat *mat_w_2, struct Mat *mat_b_2, struct Mat *mat_l_3)
{
	for (int i = 0; i < 100; i++)
	{
		calc_forw_back_paths(mat_result, mat_l_1, mat_w_1, mat_b_1, mat_l_2, mat_w_2, mat_b_2, mat_l_3, mat_w_1);
		calc_forw_back_paths(mat_result, mat_l_1, mat_w_1, mat_b_1, mat_l_2, mat_w_2, mat_b_2, mat_l_3, mat_w_2);
		calc_forw_back_paths(mat_result, mat_l_1, mat_w_1, mat_b_1, mat_l_2, mat_w_2, mat_b_2, mat_l_3, mat_b_1);
		calc_forw_back_paths(mat_result, mat_l_1, mat_w_1, mat_b_1, mat_l_2, mat_w_2, mat_b_2, mat_l_3, mat_b_2);
	}
}

int main()
{
	srand(time(NULL));
	struct Num *numbers = NULL;
	int numbers_count = 0;

	load_dataset("../train.csv", &numbers, &numbers_count);

	// for (int i = 0; i < numbers_count; i++)
	// {
	// 	print_number(&numbers[i]);
	// }

	struct Mat *mat_l_1 = create_mat(1, 28 * 28);
	struct Mat *mat_w_1 = create_mat(28 * 28, 28);
	struct Mat *mat_b_1 = create_mat(1, 28);

	struct Mat *mat_l_2 = create_mat(1, 28);
	struct Mat *mat_w_2 = create_mat(28, 10);
	struct Mat *mat_b_2 = create_mat(1, 10);

	struct Mat *mat_l_3 = create_mat(1, 10);

	struct Mat *mat_result = create_mat(1, 10);

	fill_mat_randf(mat_w_1, -1, 1);
	fill_mat_randf(mat_b_1, -1, 1);

	fill_mat_randf(mat_w_2, -1, 1);
	fill_mat_randf(mat_b_2, -1, 1);

	for (int i = 0; i < 100; i++)
	{
		mat_l_1->data = numbers[i].data;

		fill_mat_numf(mat_result, 0);
		mat_result->data[numbers[i].label] = 1;

		calc_error(mat_result, mat_l_1, mat_w_1, mat_b_1, mat_l_2, mat_w_2, mat_b_2, mat_l_3);
	}

	return 0;
}