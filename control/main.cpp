#include "controler.h"
#include <Python.h>

using namespace std;

int initPython(PyObject **pName,PyObject **pModule,PyObject **pFunc,PyObject **pArgs,PyObject **pValue,vector<int> val);
vector<int> listTupleToVector_Int(PyObject* incoming);
CHOICE choiceCandy();
int setArgs(PyObject **pArgs,PyObject **pValue,vector<int> val) ;


/*
 * valeur retourner par le python = tableau de 5 cases
 * 0 : booléen pour savoir si la detection a marché 
 * 1 : angle de la base	
 * 2 : angle du bras 1
 * 3 : angle du bras 2
 * 4 : angle du bras 3
*/
int main()
{
	Controler ssc;
	//ssc.keyboardControl();
	//ssc.test();
	
	CHOICE candy(CHOICE::UNKNOW);
	bool rotBas(false);
	bool catchCandy(false);
	//angle de rotation si on trouve pas de bonbon
	int angleBase = 30; 
	
	PyObject *pName(NULL), *pModule(NULL), *pFunc(NULL);
    PyObject *pArgs(NULL), *pValue(NULL);
    vector<int> tmp; tmp.push_back(6);tmp.push_back(7);
    
	if(initPython(&pName,&pModule,&pFunc,&pArgs,&pValue,tmp) == 1){
		if(pFunc !=NULL)
			Py_XDECREF(pFunc);
		if(pModule !=NULL)	
			Py_DECREF(pModule);
	}
	
	while(1)
	{	
		
		if(candy == CHOICE::UNKNOW)
		{
			candy =choiceCandy();
			cout<<"#### Bonbon choisit : "<<enumToString(candy)<<"####"<<endl<<endl;
		}
		else if (!rotBas)
		{
			vector<int> t; t.push_back(candy);t.push_back(-1);
			setArgs(&pArgs,&pValue,t);
			pValue = PyObject_CallObject(pFunc, pArgs);
		
			if (pValue != NULL) {
				vector<int> result = listTupleToVector_Int(pValue);
				//premier entier indique s'il y a bien eu une détection
				if(result[0] == 0)
				{
					//si pas de détection on balaye l'environnement
					if(ssc.moveBase(angleBase) == ERROR)
						angleBase = -angleBase;
					else{
						cout<<"[C++][BASE ROTATE] rotation de la base de "<<  angleBase<<"° à la recherche du bonbon"<<endl;
						ssc.waitForDone();
					}
				}
				else
				{
					if(result[1]!=0 && ssc.moveBase(-result[1]) == ERROR){
						cout<<"[C++][BASE ROTATE] ERREUR ROTATION DE LA BASE DE "<<  result[1]<<"° "<<endl;
					}
					else if(result[1]==0){
						rotBas=true;
					}
					else{
						cout<<"[C++][BASE ROTATE] rotation de la base de "<<  result[1]<<"° pour faire face au bonbon"<<endl;
						ssc.waitForDone();
					}
					
				}
				
				/*for(int i=0; i<result.size();i++)
				{
					printf("Result of call: %d\n", result[i]);
				}*/
			}
			else {
				Py_DECREF(pFunc);
				Py_DECREF(pModule);
				PyErr_Print();
				fprintf(stderr,"Call failed\n");
				return 1;
			}
			
			
		}
		else if(!catchCandy)
		{
			int depth;
			cout<<"veuillez entrez la distance à l'objet !"<<endl;
			cin>>depth;
			vector<int> t; t.push_back((int)candy);t.push_back(depth);
			setArgs(&pArgs,&pValue,t);
			pValue = PyObject_CallObject(pFunc, pArgs);
		
			if (pValue != NULL) {
				vector<int> result = listTupleToVector_Int(pValue);
				//result.push_back(1);result.push_back(1);result.push_back(110);result.push_back(75);
				
				if(result[0] == 0)
				{
					cout<<"[C++][ARM ROTATE] ERREUR: AUCUN BONBON DETECTE "<<endl;
				}
				else
				{
					if(ssc.moveArm(result[2],result[3],result[4]) == ERROR){
						cout<<"[C++][ARM ROTATE] ERREUR ROTATION DU BRAS"<<endl;
						cout<<"    bras 1 : " << result[2]<<"°"<<endl;
						cout<<"    bras 2 : " << result[3]<<"°"<<endl;
					}
					else{
						cout<<"[C++][ARM ROTATE] Rotation du bras "<<endl;
						//cout<<"    bras 1 : " << result[2]<<"°"<<endl;
						//cout<<"    bras 2 : " << result[3]<<"°"<<endl;
					}
				}
				
				/*for(int i=0; i<result.size();i++)
				{
					printf("Result of call: %d\n", result[i]);
				}*/
			}
			else {
				Py_DECREF(pFunc);
				Py_DECREF(pModule);
				PyErr_Print();
				fprintf(stderr,"Call failed\n");
				return 1;
			}
			
			catchCandy=true;
		}
		else
		{
			candy= CHOICE::UNKNOW;
			rotBas=false;
			catchCandy=false;
		}

	}
	
	Py_XDECREF(pFunc);
	Py_DECREF(pModule);
    Py_Finalize();
    
	return 0;
}

int initPython(PyObject **pName,PyObject  **pModule,PyObject  **pFunc,PyObject  **pArgs,PyObject  **pValue,vector<int> val)
{
		
	char path[] ="main";
	char name[] ="main";
	
    int i;
    Py_Initialize();
	PyRun_SimpleString("import sys");
	PyRun_SimpleString("import sys");
	PyRun_SimpleString("import sys");
	PyRun_SimpleString("import sys");
	PyRun_SimpleString("import sys");
	PyRun_SimpleString("import sys");
	PyRun_SimpleString("import sys");
	PyRun_SimpleString("import sys");
	PyRun_SimpleString("sys.path.append(\"/home/pi/Desktop/rasp/PR_git/Python/\")");
	
    *pName = PyUnicode_FromString(path);
    /* Error checking of pName left out */

    *pModule = PyImport_Import(*pName);
    Py_DECREF(*pName);

    if (*pModule != NULL) {
        *pFunc = PyObject_GetAttrString(*pModule, name);
        /* pFunc is a new reference */

        if (*pFunc && PyCallable_Check(*pFunc)) {
           /* //n argument à passer à la fonction python
            *pArgs = PyTuple_New(val.size());
    
            //conversion argument en type python
            for(int i=0;i<val.size();i++){
				*pValue = PyLong_FromLong(val[i]);
				if(!*pValue){
					Py_DECREF(*pArgs);
					Py_DECREF(*pModule);
					fprintf(stderr, "Cannot convert argument\n");
					return 1;
				}
			
			PyTuple_SetItem(*pArgs, i, *pValue);
			}*/

        }
        else {
            if (PyErr_Occurred())
                PyErr_Print();
            fprintf(stderr, "Cannot find function \"%s\"\n", name);
        }

    }
    else {
        PyErr_Print();
        fprintf(stderr, "Failed to load \"%s\"\n", path);
        return 1;
    }
    return 0;
}

int setArgs(PyObject **pArgs,PyObject **pValue,vector<int> val) 
{
	if(*pArgs != NULL)
		Py_DECREF(*pArgs);
		
	*pArgs = PyTuple_New(val.size());
    
	//conversion argument en type python
	for(int i=0;i<val.size();i++){
		*pValue = PyLong_FromLong(val[i]);
		if(!*pValue){
			Py_DECREF(*pArgs);
			fprintf(stderr, "Cannot convert argument\n");
			return 1;
		}
	
		PyTuple_SetItem(*pArgs, i, *pValue);
	}
}


vector<int> listTupleToVector_Int(PyObject* incoming) {
	vector<int> data;
	if (PyTuple_Check(incoming)) {
		for(Py_ssize_t i = 0; i < PyTuple_Size(incoming); i++) {
			PyObject *value = PyTuple_GetItem(incoming, i);
			data.push_back( PyFloat_AsDouble(value) );
		}
	} else {
		if (PyList_Check(incoming)) {
			for(Py_ssize_t i = 0; i < PyList_Size(incoming); i++) {
				PyObject *value = PyList_GetItem(incoming, i);
				data.push_back( PyFloat_AsDouble(value) );
			}
		} else {
			throw logic_error("Passed PyObject pointer was not a list or tuple!");
		}
	}
	return data;
}

std::string enumToString(CHOICE c)
{
	switch(c)
	{
		case 0:
			return string("red bear");
		case 1:
			return string("green bear");
		case 2:
			return string("red croco");
		case 3:
			return string("green croco");
		case 4:
			return string("carambar");
		default:
			return string("unknow");
	}	
}

CHOICE choiceCandy()
{
	printf("choix bonbon\n\
RED_BEAR	=	0\n\
GREEN_BEAR	=	1\n\
RED_CROCO	= 	2\n\
GREEN_CROCO	= 	3\n\
CARAMBAR 	= 	4\n\
UNKNOW 		=	5\n\n");
	fflush(stdout);
	int choix;
	scanf("%d",&choix);
	return (CHOICE)choix;
}
