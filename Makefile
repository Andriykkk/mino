CC = gcc
CFLAGS = -g
EXEC = main
GENERATOR = generator
SOURCE = src/main.c src/matrix/matrix.c src/matrix/matrix_g.c

all: $(EXEC)
	./$(EXEC)

$(EXEC): $(SOURCE)
	$(CC) $(CFLAGS) -o $(EXEC) $(SOURCE)

build_generator: generate/main.c
	$(CC) $(CFLAGS) -o $(GENERATOR) generate/main.c generate/env.c generate/calle.c

generate: $(GENERATOR)
	./$(GENERATOR) src/matrix/matrix.h src/matrix/matrix_g.h
	./$(GENERATOR) src/matrix/matrix.c src/matrix/matrix_g.c

clean:
	rm -f $(EXEC)
