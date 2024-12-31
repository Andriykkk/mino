#include <stdio.h>
#include <stdlib.h>
#include "matrix_g.h"
#include "matrix.h"

//<?
Matrix___TYPEN__ *create_matrix___TYPEN__(int *dims, int dims_size, __TYPE__ initial_value)
{
    Matrix___TYPEN__ *matrix = (Matrix___TYPEN__ *)malloc(sizeof(Matrix___TYPEN__));
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
    matrix->data_type = __TYPET__;
    matrix->data = (__TYPE__ *)malloc(sizeof(__TYPE__) * total_size);

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

void free_matrix___TYPEN__(Matrix___TYPEN__ *matrix)
{
    free(matrix->data);
    free(matrix->dims);
    free(matrix);
}

int get_data_index___TYPEN__(Matrix___TYPEN__ *matrix, int *indices)
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

void print_matrix_recursive___TYPEN__(Matrix___TYPEN__ *matrix, int *indices, int dim)
{
    if (dim == matrix->num_dims)
    {
        int idx = get_data_index___TYPEN__(matrix, indices);
        printf("__TYPEP%f__ ", matrix->data[idx]);
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
        print_matrix_recursive___TYPEN__(matrix, indices, dim + 1);

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

void print_matrix___TYPEN__(Matrix___TYPEN__ *matrix)
{
    int *indices = (int *)malloc(matrix->num_dims * sizeof(int));
    print_matrix_recursive___TYPEN__(matrix, indices, 0);
    free(indices);
}

//?>