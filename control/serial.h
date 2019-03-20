#ifndef H_SERIAL
#define H_SERIAL

#include "type.h"
#include <iostream>
#include <cstring>
#include <errno.h>
#include <string>

class Serial
{
	unsigned int baud;
	std::string device;
	bool init;
	int fd;
	
public:
	Serial(std::string d, unsigned int b);
	~Serial();
	RET initialize();
	RET send(std::string cmd);
	std::string receive();
	void pause(unsigned int t);
};

#endif
