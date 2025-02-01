#include "env.h"

types active_types[] = {FLOAT_32, CHAR, SHORT, INT};
const int active_types_size = sizeof(active_types) / sizeof(active_types[0]);

char *type_to_string(types type, gen_type gen_type)
{
    if (gen_type == GEN_TYPE_SHORT_NAME)
    {

        if (type == FLOAT_32)
        {
            return "f32";
        }
        else if (type == DOUBLE)
        {
            return "double";
        }
        else if (type == CHAR)
        {
            return "char";
        }
        else if (type == SHORT)
        {
            return "short";
        }
        else if (type == INT)
        {
            return "int";
        }
        else if (type == LONG)
        {
            return "longint";
        }
    }
    else if (gen_type == GEN_TYPE_FULL_NAME)
    {
        if (type == FLOAT_32)
        {
            return "float";
        }
        else if (type == DOUBLE)
        {
            return "double";
        }
        else if (type == CHAR)
        {
            return "char";
        }
        else if (type == SHORT)
        {
            return "short";
        }
        else if (type == INT)
        {
            return "int";
        }
        else if (type == LONG)
        {
            return "long long";
        }
    }
    else if (gen_type == GEN_TYPE_PRINT_NAME)
    {
        if (type == FLOAT_32)
        {
            return "%f";
        }
        else if (type == DOUBLE)
        {
            return "%lf";
        }
        else if (type == CHAR)
        {
            return "%c";
        }
        else if (type == SHORT)
        {
            return "%hd";
        }
        else if (type == INT)
        {
            return "%d";
        }
        else if (type == LONG)
        {
            return "%lld";
        }
    }
    else if (gen_type == GEN_TYPE_TYPES_NAME)
    {
        if (type == FLOAT_32)
        {
            return "FLOAT_32";
        }
        else if (type == DOUBLE)
        {
            return "DOUBLE";
        }
        else if (type == CHAR)
        {
            return "CHAR";
        }
        else if (type == SHORT)
        {
            return "SHORT";
        }
        else if (type == INT)
        {
            return "INT";
        }
        else if (type == LONG)
        {
            return "LONG";
        }
    }
    return "unknown";
}