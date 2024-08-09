#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

int array[][2] = {
	{10, 20},
	{12, 24},
	{14, 28},
	{16, 32}};
float learning_rate = 0.001;

float randf()
{
	return (float)(rand()) / (float)(rand());
}

float count_error(float w, float x, float y)
{
	return y - (w * x);
}

float get_forward(float w, float x)
{
	return w * x;
}

int main()
{
	srand(time(NULL));

	float w = randf();
	float error_sum = 0;

	float result = get_forward(w, 20);
	printf("result: %f\n", result);

	for (int i = 0; i < 1000; i++)
	{
		for (int j = 0; j < 4; j++)
		{
			float error = count_error(w, array[j][0], array[j][1]);
			error_sum += error;
		}

		w = w + learning_rate * error_sum;
		printf("error sum: %f\n", error_sum);
		error_sum = 0;
	}

	result = get_forward(w, 20);
	printf("result: %f\n", result);

	return 0;
}