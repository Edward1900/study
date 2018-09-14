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
	PyRun_SimpleString("import pytest");
	PyRun_SimpleString("pytest.add(1,2)");

	PyObject *pName, *pModule, *pDict, *pFunc, *pFunc1, *pArgs, *pArgs1,*pvalue, *pvalue1;
	//// ������Ϊpytest�Ľű�
	pName = PyUnicode_FromString("pytest");
	pModule = PyImport_Import(pName);
	//pModule = PyImport_ImportModule("pytest");
	if (!pModule)
	{
		printf("can't find pytest.py");
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
	pFunc = PyDict_GetItemString(pDict, "add");
	if (!pFunc || !PyCallable_Check(pFunc))
	{
		printf("can't find function [add]");
		getchar();
		return -1;
	}
	// ������ջ
	pArgs = PyTuple_New(2);
	//  PyObject* Py_BuildValue(char *format, ...)
	//  ��C++�ı���ת����һ��Python���󡣵���Ҫ��
	//  C++���ݱ�����Pythonʱ���ͻ�ʹ������������˺���
	//  �е�����C��printf������ʽ��ͬ�����õĸ�ʽ��
	//  s ��ʾ�ַ�����
	//  i ��ʾ���ͱ�����
	//  f ��ʾ��������
	//  O ��ʾһ��Python����
	PyTuple_SetItem(pArgs, 0, Py_BuildValue("l", 3));
	PyTuple_SetItem(pArgs, 1, Py_BuildValue("l", 4));
	// ����Python����
	pvalue = PyObject_CallObject(pFunc, pArgs);

	std::string res = PyUnicode_AsUTF8(pvalue);
	std::cout << "covt:" << res << std::endl;
	//��������ǲ��Һ���foo ��ִ��foo
	pFunc1 = PyDict_GetItemString(pDict, "foo");
	if (!pFunc1 || !PyCallable_Check(pFunc1))
	{
		printf("can't find function [foo]");
		getchar();
		return -1;
	}
	pArgs1 = PyTuple_New(1);
	PyTuple_SetItem(pArgs1, 0, Py_BuildValue("l", 2)); //
	pvalue1 = PyObject_CallObject(pFunc1, pArgs1);
	int res2 = PyLong_AsLong(pvalue1);
	std::cout << "covt2:" << res2 << std::endl;

	Py_DECREF(pName);
//	Py_DECREF(pArgs);
	//Py_DECREF(pModule);
	// �ر�Python
	Py_Finalize();
	return 0;
}