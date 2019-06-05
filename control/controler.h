#ifndef H_CONTROLER
#define H_CONTROLER

#include "type.h"
#include "serial.h"
#include "servo.h"


/*
 *La classe Controler gère le controle de chaque servo moteur
 */

class Controler
{
	private:
		std::vector<Servo> servos;
	public:
	
		Controler();
		
		//fonction temporaire de test
		void test();
	
		//permet de tourner la base du bras
		RET moveBase(int angle);
		
		//permet de deplacer le bras
		RET moveArm(int angle1,int angle2);
		
		//permet de deplacer un moteur
		RET moveHand(int angle);
		

		//stop le moteur désiré 
		void stop(unsigned int servo);
		
		void waitForDone();
		
		void init(unsigned int speed = DEFAULT_TIME);
		
		void coucou();
		
		
};

#endif
