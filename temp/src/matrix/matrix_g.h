#include "../defc/defc.h"

typedef struct
{
    int num_dims;
    int total_elements;
    int *dims;
    types data_type;
    float *data;

} Matrix_f32;

Matrix_f32 *create_matrix_f32(int *dims, int dims_size, float initial_value);
void free_matrix_f32(Matrix_f32 *matrix);

void print_matrix_f32(Matrix_f32 *matrix);
void print_matrix_recursive_f32(Matrix_f32 *matrix, int *indices, int dim);
int get_data_index_f32(Matrix_f32 *matrix, int *indices);
typedef struct
{
    int num_dims;
    int total_elements;
    int *dims;
    types data_type;
    char *data;

} Matrix_char;

Matrix_char *create_matrix_char(int *dims, int dims_size, char initial_value);
void free_matrix_char(Matrix_char *matrix);

void print_matrix_char(Matrix_char *matrix);
void print_matrix_recursive_char(Matrix_char *matrix, int *indices, int dim);
int get_data_index_char(Matrix_char *matrix, int *indices);
typedef struct
{
    int num_dims;
    int total_elements;
    int *dims;
    types data_type;
    short *data;

} Matrix_short;

Matrix_short *create_matrix_short(int *dims, int dims_size, short initial_value);
void free_matrix_short(Matrix_short *matrix);

void print_matrix_short(Matrix_short *matrix);
void print_matrix_recursive_short(Matrix_short *matrix, int *indices, int dim);
int get_data_index_short(Matrix_short *matrix, int *indices);
typedef struct
{
    int num_dims;
    int total_elements;
    int *dims;
    types data_type;
    int *data;

} Matrix_int;

Matrix_int *create_matrix_int(int *dims, int dims_size, int initial_value);
void free_matrix_int(Matrix_int *matrix);

void print_matrix_int(Matrix_int *matrix);
void print_matrix_recursive_int(Matrix_int *matrix, int *indices, int dim);
int get_data_index_int(Matrix_int *matrix, int *indices);
