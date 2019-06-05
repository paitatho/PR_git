CXX=g++
CXXFLAGS=-W -Wall -pedantic -std=c++11
LDFLAGS=-lwiringPi
EXEC=prog
LDFLAGS += -L/usr/lib/python3.5/config-3.5m-arm-linux-gnueabihf -L/usr/lib -lpython3.5m
INCLUDEPATH= -I/usr/include/python3.5m
SRC= $(wildcard *.cpp)
OPATH= obj/
OBJ= $(SRC:.cpp=.o)
OBJS = $(addprefix $(OPATH), $(OBJ))

all: $(EXEC)

prog: $(OBJ)
	@echo "generating executable ..."
	@$(CXX) $(INCLUDEPATH) $(LDFLAGS) -o $@ $(addprefix $(OPATH), $^)
	
%.o: %.cpp
	@echo "creating object ..."
	$(CXX) $(INCLUDEPATH) -c $< -o $(OPATH)$@

.PHONY: clean mrproper

clean:
	@rm -rf $(OPATH)*

mrproper: clean
	@rm -rf $(EXEC)
	
depend:
	makedepend -- $(CXXFLAGS) -- $(SRC)
			
# DO NOT DELETE

serial.o: /usr/include/wiringSerial.h /usr/include/wiringPi.h serial.h
serial.o: type.h /usr/include/errno.h /usr/include/features.h
serial.o: /usr/include/stdc-predef.h
controler.o: controler.h type.h serial.h /usr/include/errno.h
controler.o: /usr/include/features.h /usr/include/stdc-predef.h
servo.o: servo.h type.h serial.h /usr/include/errno.h
servo.o: /usr/include/features.h /usr/include/stdc-predef.h
main.o: controler.h type.h serial.h /usr/include/errno.h
main.o: /usr/include/features.h /usr/include/stdc-predef.h
