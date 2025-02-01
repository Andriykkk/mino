#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "env.h"
#include "calle.h"
#include "templater.h"

char *generate_calle(char last_char, char *declaration, char *switch_block, char *case_block)
{

    char *code = (char *)malloc(sizeof(char) * (102400));

    strcat(code, declaration);
    if (last_char == 'h')
    {
        strcat(code, ";");
        return code;
    }

    strcat(code, switch_block);

    size_t type_len;

    for (int i = 0; i < active_types_size; i++)
    {
        char *new_case_block = (char *)malloc(sizeof(char) * (102400));
        strcpy(new_case_block, case_block);

        char *pos = new_case_block;

        types type = active_types[i];

        replace_function_types(pos, type, new_case_block, "__TYPE__", GEN_TYPE_FULL_NAME);
        replace_function_types(pos, type, new_case_block, "__TYPEN__", GEN_TYPE_SHORT_NAME);
        replace_function_types(pos, type, new_case_block, "__TYPEP%f__", GEN_TYPE_PRINT_NAME);
        replace_function_types(pos, type, new_case_block, "__TYPET__", GEN_TYPE_TYPES_NAME);

        strcat(code, new_case_block);
        free(new_case_block);
    }

    strcat(code, "\n  }\n}\n");

    return code;
}