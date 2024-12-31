struct Mat
{
    int rows;
    int cols;
    float *data;
    float *g;
};

struct Mat *create_mat(int rows, int cols);

// Matrix initialization
void fill_mat_randf(struct Mat *mat, int min, int max);
void fill_mat_numf(struct Mat *mat, float num);
void print_mat(struct Mat *mat);
void free_mat(struct Mat *mat);

// Matrix operations
void add_mats(struct Mat *result_mat, struct Mat *mat_a, struct Mat *mat_b);
void sub_mats(struct Mat *result_mat, struct Mat *mat_a, struct Mat *mat_b);
void mul_mats(struct Mat *result_mat, struct Mat *mat_a, struct Mat *mat_b);

// Activation functions
void mat_softmax(struct Mat *mat);