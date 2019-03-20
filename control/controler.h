#ifndef H_CONTROLER
#define H_CONTROLER

#include "type.h"
#include "serial.h"
  
class Controler
{
	private:

		Serial sscUart;
	public:
	
		Controler();
		void move();
		unsigned short int done();

	
};

#endif
