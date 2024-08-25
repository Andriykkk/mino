#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>
#include <string.h>
#include "csv.h"

#define BUFFER_SIZE 10024

int load_dataset(char *filename, struct Num **numbers, int *numbers_count)
{
    char buffer[BUFFER_SIZE];
    FILE *file;

    int t_count = 0;
    int row = 0;
    int size = 1000;

    // allocate memory
    *numbers = (struct Num *)(malloc(sizeof(struct Num) * size));
    if (*numbers == NULL)
    {
        printf("Failed to allocate memory for numbers\n");
        return 1;
    }

    // Open the file
    file = fopen(filename, "r");
    if (file == NULL)
    {
        printf("Failed to open file\n");
        return 1;
    }

    // Read the file
    fgets(buffer, BUFFER_SIZE, file);

    while (fgets(buffer, BUFFER_SIZE, file) != NULL)
    {
        // allocate more memory if needed
        if (row >= size)
        {
            size += 1000;
            *numbers = (struct Num *)(realloc(*numbers, sizeof(struct Num) * size));
            if (*numbers == NULL)
            {
                printf("Failed to allocate memory for numbers\n");
                return 1;
            }
        }

        char *token = strtok(buffer, ",");

        t_count = 0;

        (*numbers)[row].label = atoi(token);

        while ((token = strtok(NULL, ",")) != NULL)
        {
            (*numbers)[row].data[t_count++] = atof(token) / 255.0;
        }

        row += 1;
    }

    *numbers_count = row;
    printf("Loaded %d numbers\n", *numbers_count);

    // Close the file
    fclose(file);

    return 0;
}

void print_number(struct Num *number)
{
    float threshold = 0.2;

    for (int i = 0; i < 28; i++)
    {
        printf("\n");
        for (int j = 0; j < 28; j++)
        {
            if (number->data[i * 28 + j] > threshold)
            {
                printf("#");
            }
            else
            {
                printf("_");
            }
        }
    }

    printf("\nLabel: %d", number->label);
}
