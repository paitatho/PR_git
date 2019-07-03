#include "controler.h"
#include <Python.h>

using namespace std;

int initPython(PyObject **pName,PyObject **pModule,PyObject **pFunc,PyObject **pArgs,PyObject **pValue,vector<int> val);
vector<int> listTupleToVector_Int(PyObject* incoming);
CHOICE choiceCandy();
int setArgs(PyObject **pArgs,PyObject **pValue,vector<float> val) ;
vector<float> listTupleToVector_Float(PyObject* incoming);

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
	
	/*cette fonction permet de déplacer le robot à l'aide de touches clavier
	  voir définition pour plus de détails*/
	//ssc.keyboardControl();
	
	CHOICE candy(CHOICE::UNKNOW);
	bool rotBas(false);
	bool moveArm(false);
	bool catchCandy(false);
	//angle de rotation si on trouve pas de bonbon
	int angleBase = 30; 
	
	PyObject *pName(NULL), *pModule(NULL), *pFunc(NULL);
    PyObject *pArgs(NULL), *pValue(NULL);
    
    vector<int> tmp; //tmp.push_back(6);tmp.push_back(7);
    
	if(initPython(&pName,&pModule,&pFunc,&pArgs,&pValue,tmp) == 1){
		if(pFunc !=NULL)
			Py_XDECREF(pFunc);
		if(pModule !=NULL)	
			Py_DECREF(pModule);
	}
	
	//boucle principale du programme
	while(1)
	{	
		//d'abord on demande le choix du bonbon
		if(candy == CHOICE::UNKNOW)
		{
			candy =choiceCandy();
			cout<<"#### Bonbon choisi : "<<enumToString(candy)<<" ####"<<endl<<endl;
		}
		//Ensuite on tourne la base tant que le bras n'est pas aligné au bonbon
		else if (!rotBas)
		{
			vector<float> t; t.push_back(candy);t.push_back(-1.0);
			setArgs(&pArgs,&pValue,t);
			pValue = PyObject_CallObject(pFunc, pArgs);
		
			if (pValue != NULL) {
				vector<float> result = listTupleToVector_Float(pValue);
				//premier entier indique s'il y a bien eu une détection
				//si pas de détection on balaye l'environnement
				if(result[0] == 0.0)
				{
					//si on arrive au bout d'un côté on change de côté
					if(ssc.moveBase(angleBase) == ERROR)
						angleBase = -angleBase;
					else{
						cout<<"[C++][BASE ROTATE] rotation de la base de "<<  angleBase <<"° à la recherche du bonbon"<<endl;
						ssc.waitForDone();
					}
				}
				else
				{
					if(result[1]!=0.0 && ssc.moveBase(result[1]) == ERROR){
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
		//Ensuite on calcule la cinématique inverse et on les applique au bras
		else if(!moveArm)
		{
			float depth;
			cout<<"veuillez entrez la distance à l'objet !"<<endl;
			cin>>depth;
			depth -=depth/10;
			vector<float> t; t.push_back((int)candy);t.push_back(depth);
			setArgs(&pArgs,&pValue,t);
			pValue = PyObject_CallObject(pFunc, pArgs);
		
			if (pValue != NULL) {
				vector<float> result = listTupleToVector_Float(pValue);
				
				if(result[0] == 0)
				{
					cout<<"[C++][ARM ROTATE] ERREUR: AUCUN BONBON DETECTE "<<endl;
				}
				else
				{
					if(ssc.moveArm(result[2]+5,result[3],result[4]) == ERROR){
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
			
			moveArm=true;
		}
		//On attrape le bonbon
		else if(!catchCandy)
		{
			ssc.catchObject();
			catchCandy=true;
		}
		//On réinitialise le tout et on revient à la position de départ
		else
		{
			candy= CHOICE::UNKNOW;
			rotBas=false;
			catchCandy=false;
			moveArm=false;
			//retour à la position d'origine
			ssc.init(1000);
		}
	}
	
	Py_XDECREF(pFunc);
	Py_DECREF(pModule);
    Py_Finalize();
    
	return 0;
}

/*
 * Cette fonction permet d'initialiser une fonction python
 * */
int initPython(PyObject **pName,PyObject  **pModule,PyObject  **pFunc,PyObject  **pArgs,PyObject  **pValue,vector<int> val)
{
		
	char path[] ="main";	// nom du script sans le .py (marche que si c'est main)	
	char name[] ="main";	// nom de la fontion
	
    int i;
    Py_Initialize();
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

//permet de set les arguments à envoyer à la fonction python
int setArgs(PyObject **pArgs,PyObject **pValue,vector<float> val) 
{
	if(*pArgs != NULL)
		Py_DECREF(*pArgs);
		
	*pArgs = PyTuple_New(val.size());
    
	//conversion argument en type python
	for(int i=0;i<val.size();i++){
		*pValue = PyFloat_FromDouble(val[i]);
		if(!*pValue){
			Py_DECREF(*pArgs);
			fprintf(stderr, "Cannot convert argument\n");
			return 1;
		}
	
		PyTuple_SetItem(*pArgs, i, *pValue);
	}
}

// PyObject -> Vector int
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

// PyObject -> Vector float
vector<float> listTupleToVector_Float(PyObject* incoming) {
	vector<float> data;
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
