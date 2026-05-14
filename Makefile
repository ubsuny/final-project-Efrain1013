PYTHON := python3
MODULE := fortran_backend
SRC := ./fortran/fortran_backend.f90

SO_GLOB := $(MODULE)*.so

.PHONY: all build clean rebuild test-import run

all: build

build:
	$(PYTHON) -m numpy.f2py -c -m $(MODULE) $(SRC)

clean:
	rm -f $(SO_GLOB) *.mod *.o

rebuild: clean build

test-import:
	$(PYTHON) -c "import fortran_backend; print(dir(fortran_backend.fortran_backend))"

run:
	$(PYTHON) relax_main.py
