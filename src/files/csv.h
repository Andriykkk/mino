struct Num
{
    int label;
    float data[784];
};

int load_dataset(char *filename, struct Num **numbers, int *numbers_count);
void print_number(struct Num *number);