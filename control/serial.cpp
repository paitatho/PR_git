#include <wiringSerial.h> 
#include "serial.h"

using namespace std;
 

Serial::Serial(string d, unsigned int b) : device(d), baud(b),init(false), fd(-1)
{
	initialize();
}

RET Serial::initialize()
{
	cout << "Establishing the connexion ..."<<endl;
	if(!init)
	{
		if( (fd = serialOpen(device.c_str(),baud)) <0 )
		{
			cout << "Unable to open serial device: " << strerror(errno)<< endl;
			return ERROR; 
		}
		
		if(wiringPiSetup() == -1)
		{
			cout << "Unable to start wiringPi" <<endl;
			return ERROR; 
		}
	}
	cout << "Well connected" << endl;
	cout.flush();
	init = true;
	return NORM;	
}

Serial::~Serial()
{
	if(init)
		serialClose(fd);
}


RET Serial::send(string cmd){
	if(init)
	{
		serialPuts(fd, cmd.c_str()); 
		return NORM;
	}
	return ABNORM;
}

string Serial::receive(){
	string msg("");
	int nbData = serialDataAvail(fd);
	
	while(nbData >0){
		msg += (char)serialGetchar(fd);
		nbData = serialDataAvail(fd);
	}
	return msg;
	
}

void Serial::pause(unsigned int t){
	delay(t);
	return;
}




