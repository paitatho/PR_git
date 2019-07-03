#ifndef H_TYPE
#define H_TYPE

#include <iostream>
#include <cstring>
#include <errno.h>  
#include <string>
#include <cstdlib>
#include <time.h>
#include <vector>

#define DEFAULT_TIME 2000
#define DEFAULT_SPEED 1000

//type de retour pour savoir s'il y a une erreur
enum RET
{
	NORM = 0,
	ABNORM =1,
	ERROR =2
};

//Ordre des moteurs dans le tableau de la classe controler
enum PART
{
	HAND=0,
	ROT_HAND=1,
	ARM3= 2,
	ARM2= 3,
	ARM1 = 4,
	BASE = 5
};

enum CHOICE
{
	RED_BEAR=0,
	GREEN_BEAR=1,
	RED_CROCO= 2,
	GREEN_CROCO= 3,
	CARAMBAR = 4,
	UNKNOW =5
};

std::string enumToString(CHOICE c);

#endif
