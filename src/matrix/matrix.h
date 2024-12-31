#include "../defc/defc.h"

//<?
typedef struct
{
    int num_dims;
    int total_elements;
    int *dims;
    types data_type;
    __TYPE__ *data;

} Matrix___TYPEN__;

Matrix___TYPEN__ *create_matrix___TYPEN__(int *dims, int dims_size, __TYPE__ initial_value);
void free_matrix___TYPEN__(Matrix___TYPEN__ *matrix);
int get_matrix_index___TYPEN__(int *indices, Matrix___TYPEN__ *matrix);
__TYPE__ *get_matrix_element___TYPEN__(int *indices, Matrix___TYPEN__ *matrix);
void print_matrix_recursive___TYPEN__(int depth, int *indices, Matrix___TYPEN__ *matrix);
void print_matrix___TYPEN__(Matrix___TYPEN__ *matrix);
//?>
