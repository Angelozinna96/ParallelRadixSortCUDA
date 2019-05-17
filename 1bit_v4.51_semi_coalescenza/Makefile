CFLAGS=-O3 -march=native -g -Wall -Wextra
# sm_<architettura della GPU>
CUFLAGS=-O3 -arch=sm_30 -g -lineinfo
PROGS=radix_sort

%: %.cu
	nvcc $(CUFLAGS) $< -o $@

all: $(PROGS)

clean:
	$(RM) $(PROGS)

.PHONY: all clean
