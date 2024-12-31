#include <stdio.h>
#include <stdlib.h>
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

int get_matrix_index___TYPEN__(int *indices, Matrix___TYPEN__ *matrix)
{
    int index = 0;
    int stride = 1;

    for (int i = matrix->num_dims; i >= 0; i--)
    {
        index += indices[i] * stride;
        stride *= matrix->dims[i];
    }
}

__TYPE__ *get_matrix_element___TYPEN__(int *indices, Matrix___TYPEN__ *matrix)
{
    return matrix->data + get_matrix_index___TYPEN__(indices, matrix);
}

void print_matrix_recursive___TYPEN__(int depth, int *indices, Matrix___TYPEN__ *matrix)
{
    if (depth == matrix->num_dims)
    {
        printf("__TYPEP%f__", *get_matrix_element___TYPEN__(indices, matrix));
    }
    else
    {
        printf("\n");
        for (int i = 0; i < depth; i++)
        {
            printf("   ");
        }
        printf("(");
        for (int i = 0; i < matrix->dims[depth]; i++)
        {
            indices[depth] = i;
            print_matrix_recursive___TYPEN__(depth + 1, indices, matrix);
            if (i < matrix->dims[depth] - 1)
            {
                printf("  ");
            }
        }
        printf(")\n");
    }
}

void print_matrix___TYPEN__(Matrix___TYPEN__ *matrix)
{
    int *indices = (int *)malloc(sizeof(int) * matrix->num_dims);

    print_matrix_recursive___TYPEN__(0, indices, matrix);
    printf("\n");

    free(indices);
}
//?>