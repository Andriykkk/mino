# Brain2: AI library
This is a simple AI library written in C. The library aims to support a variety of data types and hardware, such as GPUs (though it currently only works with CPUs). The goal is to eventually create powerful tools for AI using this library, such as GPT-2 or other language models, and image generators.

### Right now, I am focused on:
1. 2D Matrix Operations (for simplicity and efficiency).
2. Matrix Broadcasting (only works when the second matrix is smaller than the first).
3. Template Generation: Automatically generate similar functions for different data types using a custom generator.

### Current Restrictions:
1. Only works with 2D matrices (support for more dimensions is planned for the future).
2. Broadcasting is limited to cases where the second matrix is smaller than the first and not for all functions. (Future improvements will handle more cases).

### How to Run:
1. Build the generator:
```bash
make build_generator
```
2. Run the generator:
```bash
make generate
```
Run the generator to create the necessary code for different data types.
3. Write your code:
Use the generated functions to work with 2D matrices for your AI tasks.

### Example of code
```c
	Matrix_f32 *m_inputs = (Matrix_f32 *)create_matrix_f32((int[]){3, 4}, 2, 0);
	Matrix_f32 *m_weights = (Matrix_f32 *)create_matrix_f32((int[]){3, 4}, 2, 0);
	Matrix_f32 *m_bias = (Matrix_f32 *)create_matrix_f32((int[]){1, 3}, 2, 0);
	Matrix_f32 *m_result = (Matrix_f32 *)create_matrix_f32((int[]){3, 3}, 2, 0);

	m_inputs->data = (float[]){1, 2, 3, 2.5, 2, 5, -1, 2, -1.5, 2.7, 3.3, -0.8};
	m_weights->data = (float[]){0.2, 0.8, -0.5, 1.0, 0.5, -0.91, 0.26, -0.5, -0.26, -0.27, 0.17, 0.87};
	m_bias->data = (float[]){2.0, 3.0, 0.5};
	transpose_matrix_f32(m_weights);
	print_matrix_f32(m_inputs);
	printf("\n");
	print_matrix_f32(m_weights);

	multiply_matrices_f32(m_inputs, m_weights, m_result);
	printf("\n");
	print_matrix_f32(m_result);

	add_matrices_broadcast_f32(m_result, m_bias, m_result);
	printf("\n");
	print_matrix_f32(m_result);
```