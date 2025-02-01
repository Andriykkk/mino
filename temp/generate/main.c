#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "env.h"
#include "calle.h"
#include "templater.h"

int main(int argc, char *argv[])
{
    if (argc < 3)
    {
        printf("Usage: %s <file> <output file>\n", argv[0]);
        return 1;
    }

    FILE *f = fopen(argv[1], "r");
    if (f == NULL)
    {
        printf("Failed to open file: %s\n", argv[1]);
        return 1;
    }

    FILE *out = fopen(argv[2], "w");
    if (out == NULL)
    {
        printf("Failed to open file: %s\n", argv[2]);
        return 1;
    }
    char last_char = argv[2][strlen(argv[2]) - 1];

    process_template(f, out);

    // {
    //     fprintf(out, "%s", generate_calle(last_char, "void *create_matrix(int *dims, int dims_size, void *initial_value, types data_type)\n", "{\n  switch (data_type)\n{\n", "    case __TYPET__: \n return create_matrix___TYPEN__(dims, dims_size, *((__TYPE__ *)initial_value)); \n"));
    // }

    fclose(f);
    fclose(out);

    printf("File %s processed and saved to %s\n", argv[1], argv[2]);

    return 0;
}