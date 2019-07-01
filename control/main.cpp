#include "controler.h"
#include <Python.h>

using namespace std;

int initPython(PyObject **pName,PyObject **pModule,PyObject **pFunc,PyObject **pArgs,PyObject **pValue,vector<int> val);
vector<int> listTupleToVector_Int(PyObject* incoming);
CHOICE choiceCandy();


int main()
{
	//Controler ssc;
	//ssc.test();
	
	CHOICE candy(CHOICE::UNKNOW);
	bool rotBas(false);
	bool catchCandy(false);
	
	
	PyObject *pName(NULL), *pModule(NULL), *pFunc(NULL);
    PyObject *pArgs(NULL), *pValue(NULL);
    vector<int> tmp; tmp.push_back(6);tmp.push_back(7);
    
	if(initPython(&pName,&pModule,&pFunc,&pArgs,&pValue,tmp) == 1){
		if(pFunc !=NULL)
			Py_XDECREF(pFunc);
		if(pModule !=NULL)	
			Py_DECREF(pModule);
	}
	
	while(1){	
		
		if(candy == CHOICE::UNKNOW)
		{
			candy =choiceCandy();
			cout<<"choix bonbon : "<<enumToString(candy)<<endl;
		}
		else if (!rotBas)
		{
			pValue = PyObject_CallObject(pFunc, pArgs);
		
			if (pValue != NULL) {
				vector<int> result = listTupleToVector_Int(pValue);
				for(int i=0; i<result.size();i++)
				{
					printf("Result of call: %d\n", result[i]);
				}
				//printf("Result of call: %ld\n", PyLong_AsLong(pValue));
			}
			else {
				Py_DECREF(pFunc);
				Py_DECREF(pModule);
				PyErr_Print();
				fprintf(stderr,"Call failed\n");
				return 1;
			}
			
			rotBas=true;
		}
		else if(!catchCandy)
		{
			
		}
		else
		{
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
	char name[] ="ttt";
	
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
            //1 argument à passer à la fonction python
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
			}

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
RED_BEAR=0\n\
GREEN_BEAR=1\n\
RED_CROCO= 2\n\
GREEN_CROCO= 3\n\
CARAMBAR = 4\n\
UNKNOW =5\n");
	fflush(stdout);
	int choix;
	scanf("%d",&choix);
	return (CHOICE)choix;
}
