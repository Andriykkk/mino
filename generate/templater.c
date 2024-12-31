#include "templater.h"
#include "env.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void replace_function_types(char *pos, types type, char *new_code, char *looking_string, gen_type gen_type)
{

    while ((pos = strstr(pos, looking_string)) != NULL)
    {
        size_t placeholder_len = strlen(looking_string);
        size_t type_len = strlen(type_to_string(type, gen_type));
        size_t offset = pos - new_code;

        memmove(pos + type_len, pos + placeholder_len, strlen(pos) + placeholder_len);

        memcpy(pos, type_to_string(type, gen_type), type_len);

        pos += type_len;
    }
}

char *generate_function(char *func_code, FILE *out, types type)
{
    char *new_code = (char *)malloc(sizeof(char) * (strlen(func_code) * 2));
    memcpy(new_code, func_code, strlen(func_code));

    char *pos_t = new_code;
    char *pos_n = new_code;
    size_t type_len;

    replace_function_types(pos_t, type, new_code, "__TYPE__", GEN_TYPE_FULL_NAME);
    replace_function_types(pos_t, type, new_code, "__TYPEN__", GEN_TYPE_SHORT_NAME);
    replace_function_types(pos_t, type, new_code, "__TYPEP%f__", GEN_TYPE_PRINT_NAME);
    replace_function_types(pos_t, type, new_code, "__TYPET__", GEN_TYPE_TYPES_NAME);

    printf("Functions with type %s:\n", type_to_string(type, GEN_TYPE_FULL_NAME));

    return new_code;
}

void process_template(FILE *f, FILE *out)
{
    char line[1024];
    int in_function_block = 0;
    char func_code[102400] = "";

    while (fgets(line, sizeof(line), f) != NULL)
    {
        if (strstr(line, "// <?") || strstr(line, "//<?"))
        {
            in_function_block = 1;
            strcpy(line, "");

            continue;
        }

        if (strstr(line, "// ?>") || strstr(line, "//?>"))
        {
            in_function_block = 0;

            for (int i = 0; i < active_types_size; i++)
            {
                char *new_code = generate_function(func_code, out, active_types[i]);
                fprintf(out, "%s", new_code);
                free(new_code);
            }

            continue;
        }

        if (in_function_block)
        {
            strcat(func_code, line);
        }

        if (!in_function_block)
        {
            fprintf(out, "%s", line);
        }
    }
}
