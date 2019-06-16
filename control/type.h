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

enum RET
{
	NORM = 0,
	ABNORM =1,
	ERROR =2
};

enum PART
{
	BASE = 0,
	ARM1 =1,
	ARM2=2,
	ARM3=3,
	HAND=4
};


#endif
