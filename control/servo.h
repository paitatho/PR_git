#ifndef H_SERVO
#define H_SERVO

#include "type.h"
#include "serial.h"

/*
 * La classe servo met a disposition une interface pour controler les servos moteurs.
 * Pour ce faire elle utilise la classe Serial afin de communiquer
 * suivant par uart avec la carte ssc-32u 
 */ 
class Servo
{
	friend class Controler;
	
	private:
		unsigned int min, max ,defaut, current;
		// pin de la carte ssc-32U sur laquelle est branché le moteur 
		unsigned short int pin;
		//liaison pour communiquer avec la carte
		static Serial sscUart;
	public:
	
		Servo(unsigned int p):min(500),max(2500),defaut(1500),pin(p)
		{
			current = getPulseWidth()*10;
		}
		
		Servo(unsigned int p,unsigned int d ):min(500),max(2500),pin(p)
		{
			if(d >= min && d<=max)
				defaut= d;
			else 
				defaut = 1500;
			current = getPulseWidth()*10;
		}
		
		//fonction temporaire de test
		void test(){}
	
		//permet de deplacer un moteur
		RET move(unsigned int angleSsc,unsigned int time=DEFAULT_TIME);
		
		//permet de savoir si l'action envoyé est finie
		//0 : en cours, 1: finie, 2: pas de message, 3: message inconnu 
		unsigned short int done();

		//stop le moteur
		void stop();
		
		//Met le moteur à sa position par défaut
		void initPos();
		
		//
		int getPulseWidth();
		
		static Serial* getSsc(){
			return &sscUart;
		}
		
		static void init(std::vector<Servo> s);
};

#endif
