#include <wiringPi.h>
#include <wiringSerial.h> 
#include "control.h"

using namespace std;

RET ssc::init()
{
	cout << "Establishing the connexion ..."<<endl;
	if(!i)
	{
		if( (fd = serialOpen("/dev/ttyAMA0",9600)) <0 )
		{
			cout << "Unable to open serial device: " << strerror(errno)<< endl;
			return ERROR; 
		}
	}
	
	return NORM;	
}


void ssc::move()
{

	if(wiringPiSetup() == -1)
	{
		cout << "Unable to start wiringPi" <<endl;
	}
	
	cout.flush();
	cout << "Well connected" << endl;
	string cmd("#0P1500 T2000\r");
	//~ string cmdCancel("#0P1500 S750\e");
	//~ string cmd1("#3P1000S750");
	serialPuts(fd, cmd.c_str());
	delay(3000);

	int nbData(0);
	bool test(true);

	nbData = serialDataAvail(fd);
	if (nbData> 0)
	{
		cout<< (char)serialGetchar(fd);
		cout.flush();
		test = true;
	}
	else if (nbData == -1 && test)
	{
		cout << strerror(errno)<<endl;
		test = false;
	}
}
