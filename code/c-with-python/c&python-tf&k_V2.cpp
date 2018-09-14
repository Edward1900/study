// opencvproject.cpp : 定义控制台应用程序的入口点。
//
#include "stdafx.h"
#include "opencv2/core.hpp"
#include <iostream>
#include <string>
#include "Python.h"



class TfInterface
{
public:
	TfInterface();
	~TfInterface();

	PyObject *pName, *pModule, *pDict, *pFunc, *pArgs, *pvalue;
	int TfInit();

	int TfTest(char* ImgPath);

};

TfInterface::TfInterface()
{
	Py_Initialize();
	// 检查初始化是否成功
	if (!Py_IsInitialized())
	{
		std::cout << "py_initialized is failed" << std::endl;
	}
}


TfInterface::~TfInterface()
{
	Py_DECREF(pName);
	Py_DECREF(pArgs);
	Py_DECREF(pModule);
	Py_DECREF(pDict);
	Py_DECREF(pFunc);
	Py_DECREF(pvalue);
	Py_Finalize();
}

int TfInterface::TfInit()
{
	PyRun_SimpleString("import sys");
	PyRun_SimpleString("sys.path.append('./')");
	PyRun_SimpleString("import ftry");
	pName = PyUnicode_FromString("ftry");
	pModule = PyImport_Import(pName);

	if (!pModule)
	{
		printf("can't find *.py");
		PyErr_Print();
		std::exit(1);
		getchar();
		return -1;
	}
	pDict = PyModule_GetDict(pModule);
	if (!pDict)
	{
		return -1;
	}
	pFunc = PyDict_GetItemString(pDict, "preload");
	if (!pFunc || !PyCallable_Check(pFunc))
	{
		printf("can't find function [preload]");
		getchar();
		return -1;
	}
	// 参数进栈
	pArgs = PyTuple_New(0);
	// 调用Python函数
	pvalue = PyObject_CallObject(pFunc, pArgs);

	return 0;

}

int TfInterface::TfTest(char* ImgPath)
{
	pFunc = PyDict_GetItemString(pDict, "pretest");
	if (!pFunc || !PyCallable_Check(pFunc))
	{
		printf("can't find function [pretest]");
		getchar();
		return -1;
	}
	pArgs = PyTuple_New(1);
	//PyTuple_SetItem(pArgs1, 0, Py_BuildValue("s","J:/keras_project/fine-tuning/test.jpg")); //
	PyTuple_SetItem(pArgs, 0, PyUnicode_FromString(ImgPath));
	pvalue = PyObject_CallObject(pFunc, pArgs);
	int res = PyLong_AsLong(pvalue);
	std::cout << "covt:" << res << std::endl;

	return res;

}



int main(int argc, char** argv)
{
	TfInterface det;
	det.TfInit();

	double t = (double)cvGetTickCount();
	det.TfTest("J:/keras_project/fine-tuning/test.jpg");
	t = ((double)cvGetTickCount() - t) / (1000 * (double)cvGetTickFrequency());
	std::cout << "first time:" << t << std::endl;

	double t2 = (double)cvGetTickCount();
	det.TfTest("J:/keras_project/fine-tuning/test.jpg");
	t2 = ((double)cvGetTickCount() - t2) / (1000 * (double)cvGetTickFrequency());
	std::cout << "secound time:" << t2 << std::endl;



	return 0;

}