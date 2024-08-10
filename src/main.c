#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

float array[][2] = {
	{10, 20},
	{20, 40},
	{30, 60},
	{40, 80}};
float learning_rate = 0.0001;
float eps = 0.0001;

float randf()
{
	return (float)rand() / (float)RAND_MAX;
}

float get_forward(float w, float x)
{
	return w * x;
}

float count_error(float w, float x, float y)
{
	return 1.0 / 2.0 * powf(get_forward(w, x) - y, 2.0);
}

int main()
{
	srand(time(NULL));

	float w = 0.0;

	w = randf();

	float result = get_forward(w, array[0][0]);
	printf("result: %f\n", result);

	float f1 = 0;
	float f2 = 0;
	float error = 0;
	for (int i = 0; i < 200; i++)
	{
		for (int j = 0; j < 4; j++)
		{
			f1 = count_error(w, array[j][0], array[j][1]);
			f2 = count_error(w + eps, array[j][0], array[j][1]);
			error = (f2 - f1) / eps;
			w -= learning_rate * error;
		}
	}

	result = get_forward(w, array[0][0]);
	printf("result: %f\n", result);

	return 0;
}