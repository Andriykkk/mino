#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "env.h"
#include "calle.h"

#ifndef TEMPLATER_H
#define TEMPLATER_H

void replace_function_types(char *pos, types type, char *new_code, char *looking_string, gen_type gen_type);
char *generate_function(char *func_code, FILE *out, types type);
void process_template(FILE *f, FILE *out);

#endif