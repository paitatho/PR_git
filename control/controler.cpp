#include "controler.h"

/*
 * Produit en croix pour savoir ce que représente l'angle dans la forme
 * utilisé par la carte ssc. 2500 -> 180° 
 * On ajoute la valeur à la position courante pour faire, si possible,
 * la rotation du bon angle.
 */ 

#define angleToSsc(angle, servo) \
		( servo.current + (angle * 2500 /180) )

//"/dev/ttyAMA0",9600
using namespace std;


Controler::Controler()
{
	servos.push_back(Servo(0));
	servos.push_back(Servo(3));
	servos.push_back(Servo(6,1300));
}


void Controler::stop(unsigned int servo)
{
	servos[servo].stop();
}

void Controler::waitForDone(){
	while(!servos[0].done()){}
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

//non fonctionnel 
void Controler::init(unsigned int speed){
	Serial* ssc = Servo::getSsc();
	string cmd("");
	for (int i=0;i<servos.size();++i){
		cmd +="#"+ to_string(servos[i].pin)+"P"+to_string(servos[i].defaut);
		servos[i].current = servos[i].defaut;
	}
	cmd +="T"+to_string(speed)+ "\r";
	ssc->send(cmd);
	return ;
}


//########### FONCTIONS DE TESTS ###############

void Controler::test()
{
	srand(time(0));
	cout << "Initialisation terminée \n"<<endl; 
	cout.flush();
	while(1){
		coucou();
	}
	
	
} 

void Controler::coucou(){
	Serial* ssc = Servo::getSsc();
	int t = 2200;
	// montée de la main
	cout << "montée"<<endl; 
	string cmd("");
	for (int i=0;i<servos.size();++i){
		cmd +="#"+ to_string(servos[i].pin)+"P"+to_string(1900);
		servos[i].current = servos[i].defaut;
	}
	cmd +="T"+to_string(t)+ "\r";
	ssc->send(cmd);
	waitForDone();
	
	//descente de la main
	cout << "descente"<<endl; 
	cmd = "";
	for (int i=0;i<servos.size();++i){
		cmd +="#"+ to_string(servos[i].pin)+"P"+to_string(1000);
		servos[i].current = servos[i].defaut;
	}
	cmd +="T"+to_string(t)+ "\r";
	ssc->send(cmd);
	waitForDone();
	return ;
}

