#ifndef ENV_H
#define ENV_H

typedef enum
{
    FLOAT_32,
    DOUBLE,
    CHAR,
    SHORT,
    INT,
    LONG
} types;

typedef enum
{
    GEN_TYPE_FULL_NAME,
    GEN_TYPE_SHORT_NAME,
    GEN_TYPE_PRINT_NAME,
    GEN_TYPE_TYPES_NAME
} gen_type;

extern types active_types[];
char *type_to_string(types, gen_type);

#endif