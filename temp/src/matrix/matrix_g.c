#include <stdio.h>
#include <stdlib.h>
#include "matrix_g.h"
#include "matrix.h"

Matrix_f32 *create_matrix_f32(int *dims, int dims_size, float initial_value)
{
    Matrix_f32 *matrix = (Matrix_f32 *)malloc(sizeof(Matrix_f32));
    if (matrix == NULL)
    {
        printf("Failed to allocate memory for matrix struct\n");
        exit(1);
    }
    matrix->num_dims = dims_size;

    matrix->dims = (int *)malloc(sizeof(int) * matrix->num_dims);
    int total_size = 1;

    for (int i = 0; i < matrix->num_dims; i++)
    {
        matrix->dims[i] = dims[i];
        total_size *= dims[i];
    }

    matrix->total_elements = total_size;
    matrix->data_type = FLOAT_32;
    matrix->data = (float *)malloc(sizeof(float) * total_size);

    for (int i = 0; i < total_size; i++)
    {
        matrix->data[i] = initial_value;
    }

    if (matrix->data == NULL)
    {
        printf("Failed to allocate memory for matrix\n");
        exit(1);
    }

    return matrix;
}

void free_matrix_f32(Matrix_f32 *matrix)
{
    free(matrix->data);
    free(matrix->dims);
    free(matrix);
}

int get_data_index_f32(Matrix_f32 *matrix, int *indices)
{
    int index = 0;
    int stride = 1;

    for (int i = matrix->num_dims - 1; i >= 0; i--)
    {
        index += indices[i] * stride;
        stride *= matrix->dims[i];
    }

    return index;
}

void print_matrix_recursive_f32(Matrix_f32 *matrix, int *indices, int dim)
{
    if (dim == matrix->num_dims)
    {
        int idx = get_data_index_f32(matrix, indices);
        printf("%f ", matrix->data[idx]);
        return;
    }

    if (dim)
    {
        printf("(\n");
    }

    for (int i = 0; i < matrix->dims[dim]; i++)
    {
        indices[dim] = i;
        for (int j = 0; j < dim; j++)
            printf("  ");
        print_matrix_recursive_f32(matrix, indices, dim + 1);

        if (dim && i == matrix->dims[dim] - 1)
        {
            printf("\n");
        }
    }

    if (dim)
    {
        printf(")\n");
    }
}

void print_matrix_f32(Matrix_f32 *matrix)
{
    int *indices = (int *)malloc(matrix->num_dims * sizeof(int));
    print_matrix_recursive_f32(matrix, indices, 0);
    free(indices);
}

Matrix_char *create_matrix_char(int *dims, int dims_size, char initial_value)
{
    Matrix_char *matrix = (Matrix_char *)malloc(sizeof(Matrix_char));
    if (matrix == NULL)
    {
        printf("Failed to allocate memory for matrix struct\n");
        exit(1);
    }
    matrix->num_dims = dims_size;

    matrix->dims = (int *)malloc(sizeof(int) * matrix->num_dims);
    int total_size = 1;

    for (int i = 0; i < matrix->num_dims; i++)
    {
        matrix->dims[i] = dims[i];
        total_size *= dims[i];
    }

    matrix->total_elements = total_size;
    matrix->data_type = CHAR;
    matrix->data = (char *)malloc(sizeof(char) * total_size);

    for (int i = 0; i < total_size; i++)
    {
        matrix->data[i] = initial_value;
    }

    if (matrix->data == NULL)
    {
        printf("Failed to allocate memory for matrix\n");
        exit(1);
    }

    return matrix;
}

void free_matrix_char(Matrix_char *matrix)
{
    free(matrix->data);
    free(matrix->dims);
    free(matrix);
}

int get_data_index_char(Matrix_char *matrix, int *indices)
{
    int index = 0;
    int stride = 1;

    for (int i = matrix->num_dims - 1; i >= 0; i--)
    {
        index += indices[i] * stride;
        stride *= matrix->dims[i];
    }

    return index;
}

void print_matrix_recursive_char(Matrix_char *matrix, int *indices, int dim)
{
    if (dim == matrix->num_dims)
    {
        int idx = get_data_index_char(matrix, indices);
        printf("%c ", matrix->data[idx]);
        return;
    }

    if (dim)
    {
        printf("(\n");
    }

    for (int i = 0; i < matrix->dims[dim]; i++)
    {
        indices[dim] = i;
        for (int j = 0; j < dim; j++)
            printf("  ");
        print_matrix_recursive_char(matrix, indices, dim + 1);

        if (dim && i == matrix->dims[dim] - 1)
        {
            printf("\n");
        }
    }

    if (dim)
    {
        printf(")\n");
    }
}

void print_matrix_char(Matrix_char *matrix)
{
    int *indices = (int *)malloc(matrix->num_dims * sizeof(int));
    print_matrix_recursive_char(matrix, indices, 0);
    free(indices);
}

Matrix_short *create_matrix_short(int *dims, int dims_size, short initial_value)
{
    Matrix_short *matrix = (Matrix_short *)malloc(sizeof(Matrix_short));
    if (matrix == NULL)
    {
        printf("Failed to allocate memory for matrix struct\n");
        exit(1);
    }
    matrix->num_dims = dims_size;

    matrix->dims = (int *)malloc(sizeof(int) * matrix->num_dims);
    int total_size = 1;

    for (int i = 0; i < matrix->num_dims; i++)
    {
        matrix->dims[i] = dims[i];
        total_size *= dims[i];
    }

    matrix->total_elements = total_size;
    matrix->data_type = SHORT;
    matrix->data = (short *)malloc(sizeof(short) * total_size);

    for (int i = 0; i < total_size; i++)
    {
        matrix->data[i] = initial_value;
    }

    if (matrix->data == NULL)
    {
        printf("Failed to allocate memory for matrix\n");
        exit(1);
    }

    return matrix;
}

void free_matrix_short(Matrix_short *matrix)
{
    free(matrix->data);
    free(matrix->dims);
    free(matrix);
}

int get_data_index_short(Matrix_short *matrix, int *indices)
{
    int index = 0;
    int stride = 1;

    for (int i = matrix->num_dims - 1; i >= 0; i--)
    {
        index += indices[i] * stride;
        stride *= matrix->dims[i];
    }

    return index;
}

void print_matrix_recursive_short(Matrix_short *matrix, int *indices, int dim)
{
    if (dim == matrix->num_dims)
    {
        int idx = get_data_index_short(matrix, indices);
        printf("%hd ", matrix->data[idx]);
        return;
    }

    if (dim)
    {
        printf("(\n");
    }

    for (int i = 0; i < matrix->dims[dim]; i++)
    {
        indices[dim] = i;
        for (int j = 0; j < dim; j++)
            printf("  ");
        print_matrix_recursive_short(matrix, indices, dim + 1);

        if (dim && i == matrix->dims[dim] - 1)
        {
            printf("\n");
        }
    }

    if (dim)
    {
        printf(")\n");
    }
}

void print_matrix_short(Matrix_short *matrix)
{
    int *indices = (int *)malloc(matrix->num_dims * sizeof(int));
    print_matrix_recursive_short(matrix, indices, 0);
    free(indices);
}

Matrix_int *create_matrix_int(int *dims, int dims_size, int initial_value)
{
    Matrix_int *matrix = (Matrix_int *)malloc(sizeof(Matrix_int));
    if (matrix == NULL)
    {
        printf("Failed to allocate memory for matrix struct\n");
        exit(1);
    }
    matrix->num_dims = dims_size;

    matrix->dims = (int *)malloc(sizeof(int) * matrix->num_dims);
    int total_size = 1;

    for (int i = 0; i < matrix->num_dims; i++)
    {
        matrix->dims[i] = dims[i];
        total_size *= dims[i];
    }

    matrix->total_elements = total_size;
    matrix->data_type = INT;
    matrix->data = (int *)malloc(sizeof(int) * total_size);

    for (int i = 0; i < total_size; i++)
    {
        matrix->data[i] = initial_value;
    }

    if (matrix->data == NULL)
    {
        printf("Failed to allocate memory for matrix\n");
        exit(1);
    }

    return matrix;
}

void free_matrix_int(Matrix_int *matrix)
{
    free(matrix->data);
    free(matrix->dims);
    free(matrix);
}

int get_data_index_int(Matrix_int *matrix, int *indices)
{
    int index = 0;
    int stride = 1;

    for (int i = matrix->num_dims - 1; i >= 0; i--)
    {
        index += indices[i] * stride;
        stride *= matrix->dims[i];
    }

    return index;
}

void print_matrix_recursive_int(Matrix_int *matrix, int *indices, int dim)
{
    if (dim == matrix->num_dims)
    {
        int idx = get_data_index_int(matrix, indices);
        printf("%d ", matrix->data[idx]);
        return;
    }

    if (dim)
    {
        printf("(\n");
    }

    for (int i = 0; i < matrix->dims[dim]; i++)
    {
        indices[dim] = i;
        for (int j = 0; j < dim; j++)
            printf("  ");
        print_matrix_recursive_int(matrix, indices, dim + 1);

        if (dim && i == matrix->dims[dim] - 1)
        {
            printf("\n");
        }
    }

    if (dim)
    {
        printf(")\n");
    }
}

void print_matrix_int(Matrix_int *matrix)
{
    int *indices = (int *)malloc(matrix->num_dims * sizeof(int));
    print_matrix_recursive_int(matrix, indices, 0);
    free(indices);
}

