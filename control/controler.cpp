#include "controler.h"

/*
 * Produit en croix pour savoir ce que représente l'angle dans la forme
 * utilisé par la carte ssc. 2000 -> 180° 
 * On ajoute la valeur à la position courante pour faire, si possible,
 * la rotation du bon angle.
 */ 

#define angleToSsc(angle, servo) \
		( servo.current + (angle * 2000 /180) )

//"/dev/ttyAMA0",9600
using namespace std;


Controler::Controler()
{
	servos.push_back(Servo(3));
	servos.push_back(Servo(0));
	Servo::init(servos);
}

void Controler::test()
{
	srand(time(0));
	//servos[0].initPos();
	moveBase(190);
	delay(4000);
} 



void Controler::stop(unsigned int servo)
{
	servos[servo].stop();
}

void Controler::waitForDone(){
	while(!servos[0].done()){}
}

void Controler::init(){
	Serial ssc = Servo::getSsc();
	std::string cmd("");
	for(int i = 0; i<servos.size(); ++i){
		servos[i].initPos();
	}
}

RET Controler::moveBase(int angle){
	return servos[1].move(angleToSsc(angle, servos[1]));
}

RET Controler::moveArm(int angle1,int angle2){
	return NORM;
}

RET Controler::moveHand(int angle){
	return NORM;
}

