#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "../macros/error.h"

struct Mat
{
    int rows;
    int cols;
    float *data;
    float *g;
};

// Matrix initialization
struct Mat *create_mat(int rows, int cols)
{
    srand(time(NULL));

    struct Mat *mat = (struct Mat *)malloc(sizeof(struct Mat));

    mat->rows = rows;
    mat->cols = cols;

    ASSERT_ERR(mat != NULL, "Failed to allocate memory for matrix");

    mat->data = (float *)malloc(rows * cols * sizeof(float));

    ASSERT_ERR(mat->data != NULL, "Failed to allocate memory for matrix data");

    mat->g = (float *)malloc(rows * cols * sizeof(float));

    ASSERT_ERR(mat->g != NULL, "Failed to allocate memory for matrix gradient");

    for (int i = 0; i < rows * cols; i++)
    {
        mat->data[i] = 0.0f;
    }
    for (int i = 0; i < rows; i++)
    {
        mat->g[i] = 0.0f;
    }

    return mat;
}

void free_mat(struct Mat *mat)
{
    free(mat->data);
    free(mat->g);
    free(mat);
}

void print_mat(struct Mat *mat)
{
    printf("[\n");
    for (int i = 0; i < mat->rows; i++)
    {
        printf("  [");
        for (int j = 0; j < mat->cols; j++)
        {
            printf("%f ", mat->data[i * mat->cols + j]);
        }
        printf("]\n");
    }
    printf("]\n");
}

// Matrix filling
void fill_mat_randf(struct Mat *mat, int min, int max)
{
    for (int i = 0; i < mat->rows * mat->cols; i++)
    {
        mat->data[i] = (float)rand() / RAND_MAX * (max - min) + min;
    }
}

void fill_mat_numf(struct Mat *mat, float num)
{
    for (int i = 0; i < mat->rows * mat->cols; i++)
    {
        mat->data[i] = num;
    }
}

// Matrix operations
void add_mats(struct Mat *result_mat, struct Mat *mat_a, struct Mat *mat_b)
{
    ASSERT_ERR(mat_a->cols == mat_b->cols && mat_a->rows == mat_b->rows, "Matrix dimensions do not match");
    ASSERT_ERR(result_mat->cols == mat_a->cols && result_mat->rows == mat_a->rows, "Result matrix dimensions do not match");

    for (int i = 0; i < mat_a->rows * mat_a->cols; i++)
    {
        result_mat->data[i] = mat_a->data[i] + mat_b->data[i];
    }
}

void sub_mats(struct Mat *result_mat, struct Mat *mat_a, struct Mat *mat_b)
{
    ASSERT_ERR(mat_a->cols == mat_b->cols && mat_a->rows == mat_b->rows, "Matrix dimensions do not match");
    ASSERT_ERR(result_mat->cols == mat_a->cols && result_mat->rows == mat_a->rows, "Result matrix dimensions do not match");

    for (int i = 0; i < mat_a->rows * mat_a->cols; i++)
    {
        result_mat->data[i] = mat_a->data[i] - mat_b->data[i];
    }
}

void mul_mats(struct Mat *result_mat, struct Mat *mat_a, struct Mat *mat_b)
{
    ASSERT_ERR(mat_a->cols == mat_b->rows, "Matrix dimensions do not match");
    ASSERT_ERR(result_mat->rows == mat_a->rows && result_mat->cols == mat_b->cols, "Result matrix dimensions do not match");

    float sum = 0;

    for (int i = 0; i < mat_a->rows; i++)
    {
        for (int j = 0; j < mat_b->cols; j++)
        {
            sum = 0;
            for (int k = 0; k < mat_a->cols; k++)
            {
                sum += mat_a->data[i * mat_a->cols + k] * mat_b->data[k * mat_b->cols + j];
            }
            result_mat->data[i * mat_b->cols + j] = sum;
        }
    }
}

// activations functions
void mat_softmax(struct Mat *mat)
{
    for (int i = 0; i < mat->rows; i++)
    {
        float max_value = 0;
        for (int j = 0; j < mat->cols; j++)
        {
            if (mat->data[i * mat->cols + j] > max_value)
            {
                max_value = mat->data[i * mat->cols + j];
            }
        }

        float exp_sum = 0;
        for (int j = 0; j < mat->cols; j++)
        {
            mat->data[i * mat->cols + j] = expf(mat->data[i * mat->cols + j] - max_value);
            exp_sum += mat->data[i * mat->cols + j];
        }

        for (int j = 0; j < mat->cols; j++)
        {
            mat->data[i * mat->cols + j] /= exp_sum;
        }
    }
}