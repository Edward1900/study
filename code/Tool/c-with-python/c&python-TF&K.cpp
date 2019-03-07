// opencvproject.cpp : 定义控制台应用程序的入口点。
//
#include "stdafx.h"
#include "opencv2/core.hpp"
#include <iostream>
#include <string>
#include "Python.h"
int main(int argc, char** argv)
{
	// 初始化Python
	//在使用Python系统前，必须使用Py_Initialize对其
	//进行初始化。它会载入Python的内建模块并添加系统路
	//径到模块搜索路径中。这个函数没有返回值，检查系统
	//是否初始化成功需要使用Py_IsInitialized。
	Py_Initialize();
	// 检查初始化是否成功
	if (!Py_IsInitialized())
	{
		return -1;
	}
	// 添加当前路径
	//把输入的字符串作为Python代码直接运行，返回0
	//表示成功，-1表示有错。大多时候错误都是因为字符串
	//中有语法错误。
	PyRun_SimpleString("import sys");
	PyRun_SimpleString("sys.path.append('./')");
	PyRun_SimpleString("import ftry");
	//	PyRun_SimpleString("pytest.add(1,2)");

	PyObject *pName, *pModule, *pDict, *pFunc, *pFunc1, *pArgs, *pArgs1, *pvalue, *pvalue1;
	//// 载入名为pytest的脚本
	pName = PyUnicode_FromString("ftry");
	pModule = PyImport_Import(pName);
	//pModule = PyImport_ImportModule("pytest");
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
	// 找出函数名为add的函数
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


	pFunc1 = PyDict_GetItemString(pDict, "pretest");
	if (!pFunc1 || !PyCallable_Check(pFunc1))
	{
		printf("can't find function [pretest]");
		getchar();
		return -1;
	}
	pArgs1 = PyTuple_New(1);
	//PyTuple_SetItem(pArgs1, 0, Py_BuildValue("s","J:/keras_project/fine-tuning/test.jpg")); //
	PyTuple_SetItem(pArgs1, 0, PyUnicode_FromString("J:/keras_project/fine-tuning/test.jpg"));
	pvalue1 = PyObject_CallObject(pFunc1, pArgs1);
	int res = PyLong_AsLong(pvalue1);
	std::cout << "covt:" << res << std::endl;

	Py_DECREF(pName);
	//	Py_DECREF(pArgs);
	//Py_DECREF(pModule);
	// 关闭Python
	Py_Finalize();
	return 0;
}