#include "controler.h"


//"/dev/ttyAMA0",9600
using namespace std;


Controler::Controler() : sscUart("/dev/ttyAMA0",9600)
{
	
}

void Controler::move()
{
	string cmd("#0P1200 T3000\r");
	sscUart.send(cmd);
	sscUart.pause(3100);
	unsigned short int test (done());
	cout<< "done : "<<test<<endl;
	sscUart.pause(4000);
}


unsigned short int Controler::done()
{
	sscUart.send("Q\r");
	sscUart.pause(50);
	string msg = sscUart.receive();
	cout <<"message : "<<msg<<endl;
	if(msg == ".")
		return true;
	else if (msg == "+")
		return false;
	else if (msg == "")	
		return 2;
	return 3;
}

