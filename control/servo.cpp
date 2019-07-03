#include "servo.h"
#include "controler.h"

using namespace std;

Serial Servo::sscUart = Serial("/dev/ttyAMA0",9600);

//le paramètre angle n'est pas en degré mais dans l'unité utilisée par
//la carte ssc-32U
RET Servo::move(unsigned int angleSsc,unsigned int time)
{
	if(angleSsc <= max && angleSsc >= min)
	{
		//string cmd("#"+ to_string(pin)+"P"+to_string(angleSsc)+ "T"+to_string(time)+"\r");
		Command cmd(pin,angleSsc,time);
		sscUart.send(cmd.getStr());
		current = angleSsc;
		return NORM;
	}
	cerr<< "error move servo pin: "<<pin<<" with power: "<<angleSsc<<endl;
	return ERROR;
}

unsigned short int Servo::done()
{
	sscUart.send("Q\r");
	sscUart.pause(50);
	string msg (sscUart.receive());
	//cout << msg <<endl;
	if(msg == ".")
		return 1;
	else if (msg == "+")
		return 0;
	else if (msg == "")	
		return 2;
	return 3;
}

void Servo::stop()
{
	cerr << "STOP servo pin : "<<pin<<endl;
	string cmd("STOP "+to_string(pin)+"\r");
	sscUart.send(cmd);
}

void Servo::initPos(){
	Command cmd(pin,defaut);
	sscUart.send(cmd.getStr());
	current = defaut;
}

void Servo::initPos(unsigned int time){
	Command cmd(pin,defaut,time);
	sscUart.send(cmd.getStr());
	current = defaut;
}

int Servo::getPulseWidth(){
	string cmd("QP "+ to_string(pin)+ "\r");
	sscUart.send(cmd);
	sscUart.pause(50);
	string msg (sscUart.receive());
	const char* c = msg.c_str();
	//cout << int(*c) <<endl;
	return int(*c);
}

void Servo::init(std::vector<Servo> s){
	string cmd("");
	int speed(500);
	for (int i=0;i<s.size();++i){
		cmd +="#"+ to_string(s[i].pin)+"P"+to_string(s[i].defaut)+"S"+to_string(speed);
		s[i].current = s[i].defaut;
	}
	//cmd +="T"+to_string(DEFAULT_TIME)+ "\r";
	cmd +="\r";
	sscUart.send(cmd);
	return ;
}

void Servo::setDefault(unsigned int d)
{
	if(d >= min && d<=max)
		defaut= d;
	else {
		defaut = 1500;
		std::cerr << "default out of range, setting to: "<< defaut<<std::endl;
	}
}

void Servo::setMin(unsigned int m)
{
	if(m >= min && m<=max)
		min= m;
	else {
		std::cerr << "min out of range, actual value: "<< min<<std::endl;
	}
}

void Servo::setMax(unsigned int m)
{
	if(m >= min && m<=max)
		max = m;
	else {
		std::cerr << "max out of range, actual value: "<< max<<std::endl;
	}
}



