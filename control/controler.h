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
		float arm1Angle;
		float arm2Angle;
		float arm3Angle;
	public:
	
		Controler();
		
		//fonction temporaire de test
		void test();
	
		//permet de tourner la base du bras
		RET moveBase(float angle);
		
		//permet de deplacer le bras
		RET moveArm(float angle1,float angle2,float angle3);
		
		//permet d'attraper l'objet
		RET catchObject();
		

		//stop le moteur désiré 
		void stop(unsigned int servo);
		
		//attend que l'action en cours soit finie
		void waitForDone();
		
		void init();
		
		void init(unsigned int time);

		
		//fonction de test
		void coucou();
		
		/*
		 * Cette fonction permet de deplacer les moteurs à l'aide
		 * de controle clavier. 
		 * le flêche droite et gauche permettent respectivement 
		 * d'augmenter ou de réduire l'angle d'un moteur
		 */ 
		void keyboardControl();
};

/*
 *Cette classe permet de générer une commande de déplacement
 * compréhensible par la carte SSC 
 */ 
class Command
{
	std::string cmd;
	
	public:
		Command():cmd(""){}
		
		Command(int pin,int power)
		{
			setPinPower(pin,power);
			setCarriageReturn();
		}
		
		//pin : le pin du servo qu'on désire commander
		//power: l'angle au format SSC (500 à 2500) lire la docu
		//time: le temps accordé pour ce dépacement 
		Command(int pin,int power,int time)
		{
			setPinPower(pin,power);
			setTime(time);
			//setSpeed();
			setCarriageReturn();
		}
		
		void setPinPower(int pin,int power){
			cmd +="#"+ std::to_string(pin)+"P"+std::to_string(power);
		}
		
		void setTime(int time){
			cmd +="T"+std::to_string(time);
		}
		
		void setSpeed(int speed=DEFAULT_SPEED){
			cmd +="S"+std::to_string(speed);
		}
		void setCarriageReturn(){
			cmd +="\r";
		}
		
		std::string getStr(){return cmd;}
		

};

#endif
