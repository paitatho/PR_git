CXX=g++
CXXFLAGS=-Wall 
LDFLAGS=-lwiringPi
EXEC=prog
SRC= $(wildcard *.cpp)
OBJ= $(SRC:.cpp=.o)

all: $(EXEC)

prog: $(OBJ)
	@$(CXX) -o $@ $^ $(LDFLAGS)

control.o: control.h

main.o: control.h

%.o: %.c
	@$(CXX) -o $@ -c $< $(CXXFLAGS)

.PHONY: clean mrproper

clean:
	@rm -rf *.o

mrproper: clean
	@rm -rf $(EXEC)

			