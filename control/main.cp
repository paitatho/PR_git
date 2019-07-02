#include "controler.h"
#include <Python.h>

using namespace std;

int initPython(PyObject **pName,PyObject **pModule,PyObject **pFunc,PyObject **pArgs,PyObject **pValue);

int main()
{
	//Controler ssc;
	//ssc.test();
	
	
	PyObject *pName, *pModule, *pFunc;
    PyObject *pArgs, *pValue;
	initPython(&pName,&pModule,&pFunc,&pArgs,&pValue);
	
	while(1){	
		pValue = PyObject_CallObject(pFunc, pArgs);
		
		if (pValue != NULL) {
			printf("Result of call: %ld\n", PyLong_AsLong(pValue));
		}
		else {
			Py_DECREF(pFunc);
			Py_DECREF(pModule);
			PyErr_Print();
			fprintf(stderr,"Call failed\n");
			return 1;
		}
	}
    Py_Finalize();
    
	return 0;
}

int initPython(PyObject **pName,PyObject  **pModule,PyObject  **pFunc,PyObject  **pArgs,PyObject  **pValue)
{
		
	char path[] ="main";
	char name[] ="test";
	
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
            *pArgs = PyTuple_New(1);
    
            //conversion argument en type python
            *pValue = PyLong_FromLong(3);
            if(!*pValue){
				Py_DECREF(*pArgs);
				Py_DECREF(*pModule);
				fprintf(stderr, "Cannot convert argument\n");
				return 1;
			}
			
			PyTuple_SetItem(*pArgs, 0, *pValue);
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
