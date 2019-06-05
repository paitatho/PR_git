#include "servo.h"

using namespace std;

Serial Servo::sscUart = Serial("/dev/ttyAMA0",9600);

//le paramètre angle n'est pas en degré mais dans l'unité utilisée par
//la carte ssc-32U
RET Servo::move(unsigned int angleSsc,unsigned int time)
{
	if(angleSsc <= max && angleSsc >= min)
	{
		string cmd("#"+ to_string(pin)+"P"+to_string(angleSsc)+ "T"+to_string(time)+"\r");
		sscUart.send(cmd);
		current = angleSsc;
		return NORM;
	}
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
	cout << "STOP servo pin : "<<pin<<endl;
	string cmd("STOP "+to_string(pin)+"\r");
	sscUart.send(cmd);
}

void Servo::initPos(){
	move(defaut);
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
	for (int i=0;i<s.size();++i){
		cmd +="#"+ to_string(s[i].pin)+"P"+to_string(s[i].defaut);
		s[i].current = s[i].defaut;
	}
	cmd +="T"+to_string(DEFAULT_TIME)+ "\r";
	sscUart.send(cmd);
	return ;
}



