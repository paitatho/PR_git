#ifndef H_CONTROL
#define H_CONTROL

#include <iostream>
#include <cstring>
#include <errno.h>
#include <string>

enum RET
{
	NORM = 0,
	ERROR =1
};

class ssc
{
	private:
	
		int fd;
		bool i;
	
	public:
	
		ssc() : fd(-1), i(false){;}
		RET init();
		void move();
	
};

#endif
