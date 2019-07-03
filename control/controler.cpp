#include "controler.h"
#include <unistd.h>
#include <termios.h>

/*
 * Produit en croix pour savoir ce que représente l'angle dans la forme
 * utilisé par la carte ssc. 2000 -> 180° 
 * On ajoute la valeur à la position courante pour faire, si possible,
 * la rotation du bon angle.
 */ 

#define angleToSsc(angle, current) \
		((int)( current + (angle * 2000 /180)) )

//"/dev/ttyAMA0",9600
using namespace std;


Controler::Controler()
{
	servos.push_back(Servo(0,837));  	//main
	servos.push_back(Servo(3,2222));	//poignet
	servos.push_back(Servo(6,867));		//bras3
	servos.push_back(Servo(8,2093));	//bras2	
	servos.push_back(Servo(12,2023));	//bras1
	servos.push_back(Servo(15));		//base
	
	arm1Angle = 135;   // angle entre bras 1 et le sol
	arm2Angle = 62;    // angle entre bras 1 et bras 2
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



RET Controler::moveBase(float angle){
	
	if(servos[PART::BASE].move(angleToSsc(angle, servos[PART::BASE].current),DEFAULT_TIME/3) !=NORM)
		return ERROR;
	else
		waitForDone();
	
	return NORM;
}

RET Controler::moveArm(float angle1,float angle2,float angle3)
{
	
	//##### BRAS 3	
	float angleTmp = -(arm3Angle - angle3);
	
	if(servos[PART::ARM3].move(angleToSsc(angleTmp, servos[PART::ARM3].current)) != NORM)
		return ERROR;
	else{
		cout<<"    bras 3 : " << angleTmp<<"°  "<<endl;
		waitForDone();
	}
		
	arm3Angle = angle3;
	
	//##### BRAS 2	
	
	angleTmp = arm2Angle - angle2;
	
	//cout << "angleCourant : "<<servos[PART::ARM2].current;
	//cout << "angleSsc : "<<angleToSsc(angleTmp, servos[PART::ARM2].current)<<endl;
	
	if(servos[PART::ARM2].move(angleToSsc(angleTmp, servos[PART::ARM2].current)) !=NORM)
		return ERROR;
	else{
		cout<<"    bras 2 : " << angleTmp<<"°  "<<endl;
		waitForDone();
	}	
	arm2Angle = angle2;
	
	//##### BRAS 1
	
	//différence entre l'angle qu'il faut avoir et le notre
	 angleTmp = -(arm1Angle - angle1);
	
	//cout<< "	angleCourant : "<<servos[PART::ARM1].current;
	//cout<< "	angleSsc : "<<angleToSsc(angleTmp, servos[PART::ARM1].current)<<endl;
	
	if(servos[PART::ARM1].move(angleToSsc(angleTmp, servos[PART::ARM1].current),DEFAULT_TIME*1.5) !=NORM)
		return ERROR;
	else{
		cout<<"    bras 1 : " << angleTmp<<"°  "<<endl;
		waitForDone();
	}
		
	arm1Angle = angle1;


	
	return NORM;
}

RET Controler::catchObject(){
	
	if(servos[PART::HAND].move(2100) != NORM)
		return ERROR;
	else{
		cout<<"    attrapage de l'objet" <<endl;
		waitForDone();
	}
	return NORM;
}

void Controler::init(){
		
	for (int i=servos.size();i>=0;--i)
	{
		servos[i].initPos();
		waitForDone();
	}
}

void Controler::init(unsigned int time){
		
	servos[PART::ARM1].initPos(time);
	waitForDone();
	
	servos[PART::ARM2].initPos(time);
	waitForDone();
	
	servos[PART::BASE].initPos(time);
	waitForDone();
	
	servos[PART::ARM3].initPos(time);
	waitForDone();
	
	servos[PART::ROT_HAND].initPos(time);
	waitForDone();
	
	servos[PART::HAND].initPos(time);
	waitForDone();
	
	arm1Angle = 135;   // angle entre bras 1 et le sol
	arm2Angle = 62;    // angle entre bras 1 et bras 2
	arm3Angle = 90;    // angle entre bras 2 et bras 3
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

