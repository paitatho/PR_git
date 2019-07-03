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
		unsigned int currentAngle;	//angle en degré 


	public:
		/* CONSTRUCTEURS:
		 * p = pin associé au servo moteur
		 * d = valeur par défaut utilisée à l'initialisation
		 * mi = angle min que peut prendre le moteur (contrainte physique 
		 * 		lié au montage du bras de robot)
		 * ma = angle max que peut prendre le moteur (contrainte physique 
		 * 		lié au montage du bras de robot)
		 */ 
	
		Servo(unsigned int p):min(500),max(2500),defaut(1500),pin(p)
		{
			current = getPulseWidth()*10;
		}
		
		Servo(unsigned int p,unsigned int d ):min(500),max(2500),pin(p)
		{
			setDefault(d);
			current = getPulseWidth()*10;
		}
		Servo(unsigned int p,unsigned int d,unsigned int mi ,unsigned int ma):min(500),max(2500),pin(p)
		{
			setDefault(d);
			setMin(mi);
			setMax(ma);
			current = getPulseWidth()*10;
		}
		
		
		//fonction temporaire de test
		void test(){}
	
		//permet de deplacer un moteur
		RET move(unsigned int angleSsc,unsigned int time=DEFAULT_TIME);
		
		//permet de savoir si l'action envoyé est finie
		//0 : en cours, 1: finie, 2: pas de message, 3: message inconnu 
		static unsigned short int done();

		//stop le moteur
		void stop();
		
		//Met le moteur à sa position par défaut
		void initPos();
		
		void initPos(unsigned int time);
		
		//
		int getPulseWidth();
		
		int getCurrent(){
			//current = getPulseWidth()*10;
			return current;
		}
		
		static Serial* getSsc(){
			return &sscUart;
		}
		
		static void init(std::vector<Servo> s);
		
		void setDefault(unsigned int d);
		void setMin(unsigned int m);
		void setMax(unsigned int m);
};

#endif
