ifeq ($(PREFIX),)
	PREFIX := /usr/local
endif

CC=nvcc -Wno-deprecated-gpu-targets
CFLAGS=-O2 -g -D_FORCE_INLINES -DCUDA_HOST_MALLOC
LFLAGS=-lcublas

objects = mrf.o functions.o ra.o
objects2 = test.o functions.o ra.o

all: main test

main: $(objects)
	$(CC) $(LFLAGS) $(objects) -o mrf
test: $(objects2)
	$(CC) $(LFLAGS) $(objects2) -o test
%.o: %.cu
	$(CC) $(CFLAGS) -dc $< -o $@

clean:
	rm -f *.o mrf test

install: mrf
	install -m 0755 mrf $(PREFIX)/bin
