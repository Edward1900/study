// opencvproject.cpp : �������̨Ӧ�ó������ڵ㡣
//
#include "stdafx.h"
#include "opencv2/core.hpp"
#include <iostream>
#include <string>
#include "Python.h"
int main(int argc, char** argv)
{
	// ��ʼ��Python
	//��ʹ��Pythonϵͳǰ������ʹ��Py_Initialize����
	//���г�ʼ������������Python���ڽ�ģ�鲢���ϵͳ·
	//����ģ������·���С��������û�з���ֵ�����ϵͳ
	//�Ƿ��ʼ���ɹ���Ҫʹ��Py_IsInitialized��
	Py_Initialize();
	// ����ʼ���Ƿ�ɹ�
	if (!Py_IsInitialized())
	{
		return -1;
	}
	// ��ӵ�ǰ·��
	//��������ַ�����ΪPython����ֱ�����У�����0
	//��ʾ�ɹ���-1��ʾ�д����ʱ���������Ϊ�ַ���
	//�����﷨����
	PyRun_SimpleString("import sys");
	PyRun_SimpleString("sys.path.append('./')");
	PyRun_SimpleString("import ftry");
	//	PyRun_SimpleString("pytest.add(1,2)");

	PyObject *pName, *pModule, *pDict, *pFunc, *pFunc1, *pArgs, *pArgs1, *pvalue, *pvalue1;
	//// ������Ϊpytest�Ľű�
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
	// �ҳ�������Ϊadd�ĺ���
	pFunc = PyDict_GetItemString(pDict, "preload");
	if (!pFunc || !PyCallable_Check(pFunc))
	{
		printf("can't find function [preload]");
		getchar();
		return -1;
	}
	// ������ջ
	pArgs = PyTuple_New(0);

	// ����Python����
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
	// �ر�Python
	Py_Finalize();
	return 0;
}