#ifndef H_SERIAL
#define H_SERIAL

#include "type.h"
#include <wiringPi.h>

/*
 * La classe Serial permet d'établir une connexion UART avec 
 * la carte SSC-32U. De plus elle fournit une interface pour y envoyer 
 * des commandes et lire les message.
 */ 

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
