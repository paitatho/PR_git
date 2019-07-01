#include "controler.h"
#include <unistd.h>
#include <termios.h>

/*
 * Produit en croix pour savoir ce que représente l'angle dans la forme
 * utilisé par la carte ssc. 2500 -> 180° 
 * On ajoute la valeur à la position courante pour faire, si possible,
 * la rotation du bon angle.
 */ 

#define angleToSsc(angle, current) \
		( current + (angle * 2500 /180) )

//"/dev/ttyAMA0",9600
using namespace std;


Controler::Controler()
{
	servos.push_back(Servo(0,837));  	//main
	servos.push_back(Servo(3,1331));	//poignet
	servos.push_back(Servo(6,867));		//bras3
	servos.push_back(Servo(8,2247));	//bras2	
	servos.push_back(Servo(12,2023));	//bras1
	
	arm1Angle = 135;   // angle entre bras 1 et le sol
	arm2Angle = 45;    // angle entre bras 1 et bras 2
	arm3Angle = 90;    // angle entre bras 2 et bras 3

	//initialise les servos moteurs 
	init();
}


void Controler::stop(unsigned int servo)
{
	servos[servo].stop();
}

void Controler::waitForDone(){
	while(!Servo::done()){}
}



RET Controler::moveBase(int angle){
	return servos[1].move(angleToSsc(angle, servos[1].current));
}

RET Controler::moveArm(int angle1,int angle2)
{
	//##### BRAS 1
	
	//différence entre l'angle qu'il faut avoir et le notre
	int angleTmp = arm1Angle - angle1;
	
	if(servos[PART::ARM1].move(angleToSsc(angleTmp, servos[PART::ARM1].current)) !=NORM)
		return ERROR;
		
	arm1Angle = angle1;

	//##### BRAS 2	
	
	//inverse car le moteur est monté dans l'autre sens
	angleTmp = -(arm2Angle - angle2);
	if(servos[PART::ARM2].move(angleToSsc(angleTmp, servos[PART::ARM2].current)) !=NORM)
		return ERROR;
	
	arm2Angle = angle2;
	
	return NORM;
}

RET Controler::moveHand(int angle){
	
	int angleTmp = arm3Angle - angle;
	
	if(servos[PART::ARM3].move(angleToSsc(angleTmp, servos[PART::ARM3].current)) != NORM)
		return ERROR;
		
	arm3Angle = angle;
	
	return NORM;
}

void Controler::init(unsigned int speed){
		
	for (int i=0;i<servos.size();++i)
	{
		servos[i].initPos();
		waitForDone();
	}
}


//########### FONCTIONS DE TESTS ###############


char getchBis() {
        char buf = 0;
        struct termios old = {0};
        if (tcgetattr(0, &old) < 0)
                perror("tcsetattr()");
        old.c_lflag &= ~ICANON;
        old.c_lflag &= ~ECHO;
        old.c_cc[VMIN] = 1;
        old.c_cc[VTIME] = 0;
        if (tcsetattr(0, TCSANOW, &old) < 0)
                perror("tcsetattr ICANON");
        if (read(0, &buf, 1) < 0)
                perror ("read()");
        old.c_lflag |= ICANON;
        old.c_lflag |= ECHO;
        if (tcsetattr(0, TCSADRAIN, &old) < 0)
                perror ("tcsetattr ~ICANON");
        return (buf);
}


void Controler::keyboardControl()
{
	printf("on est là \n");
	static unsigned int currentMotor(0);
	int angle(1);
	int s(100);
	
	Servo::init(servos);
		
	while(1)
	{
		char c;
		c = getchBis();
		
		switch((int)c){
			case (int)'d': //tourne de + angle de moteur courant
				servos[currentMotor].move(angleToSsc(angle, servos[currentMotor].getCurrent()),s);
				printf("current : %d\n",servos[currentMotor].current);
				break;
				
			case (int)'q': //tourne de - angle de moteur courant
				servos[currentMotor].move(angleToSsc(-angle, servos[currentMotor].getCurrent()),s);
				printf("current : %d\n",servos[currentMotor].current);
				break;

			case (int)'s':	// switch de moteur 
				currentMotor=(currentMotor+1)%servos.size();
				printf("switch de pin : %d \n",currentMotor);
				break;
		}
		
	}
} 

void Controler::test()
{
	srand(time(0));
	cout << "Initialisation terminée \n"<<endl; 
	cout.flush();
	servos[0].move(1600);
	waitForDone();
	servos[0].move(800);
	
	/*
	while(1){
		coucou();
	}*/
	
	
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

