OUTPUT_FOLDER = bin
RESULT_FOLDER = result
SIZE = 1024

all: serial parallel

mpi:
	mpic++ src/open-mpi/mpi.cpp -o $(OUTPUT_FOLDER)/mpi
	time mpirun -np 4 ./$(OUTPUT_FOLDER)/mpi < ./test_cases/$(SIZE).txt > $(RESULT_FOLDER)/mpi_$(SIZE).txt

parallel:

serial:
	g++ src/serial/serial.cpp -o $(OUTPUT_FOLDER)/serial
	time ./$(OUTPUT_FOLDER)/serial < ./test_cases/$(SIZE).txt > $(RESULT_FOLDER)/serial_$(SIZE).txt