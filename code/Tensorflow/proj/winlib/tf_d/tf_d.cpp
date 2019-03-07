// opencvproject.cpp : 定义控制台应用程序的入口点。
//
#include "stdafx.h"
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>
#include <string>
#include "Python.h"

//#pragma comment( linker, "/subsystem:windows /entry:mainCRTStartup" )

PyObject *pName, *pModule, *pDict, *pFunc, *pArgs, *pvalue;
//	
//extern "C" __declspec(dllimport) int TfInit();
////void finalize();
//extern "C" __declspec(dllimport) int TfTest(char* ImgPath);
//

extern "C" __declspec(dllexport) int TfTest(char* ImgPath)
{
	pFunc = PyDict_GetItemString(pDict, "test");
	if (!pFunc || !PyCallable_Check(pFunc))
	{
		printf("can't find function [pretest]");
		getchar();
		return -2;
	}
	pArgs = PyTuple_New(1);
	//PyTuple_SetItem(pArgs1, 0, Py_BuildValue("s","J:/keras_project/fine-tuning/test.jpg")); //
	PyTuple_SetItem(pArgs, 0, PyUnicode_FromString(ImgPath));
	pvalue = PyObject_CallObject(pFunc, pArgs);
	int res = PyLong_AsLong(pvalue);
	//std::cout << "covt:" << res << std::endl;

	return res;

}

extern "C" __declspec(dllexport) int TfInit()
{
	Py_Initialize();
	// 检查初始化是否成功
	if (!Py_IsInitialized())
	{
		std::cout << "py_initialized is failed" << std::endl;
	}
	PyRun_SimpleString("import sys");
	PyRun_SimpleString("sys.path.append('./')");
	PyRun_SimpleString("import det");
	pName = PyUnicode_FromString("det");
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
		return -5;
	}
	pFunc = PyDict_GetItemString(pDict, "load_models");
	if (!pFunc || !PyCallable_Check(pFunc))
	{
		printf("can't find function [preload]");
		getchar();
		return -3;
	}
	// 参数进栈
	pArgs = PyTuple_New(0);
	// 调用Python函数
	pvalue = PyObject_CallObject(pFunc, pArgs);
	TfTest("t.jpg");
	return 0;

}

extern "C" __declspec(dllexport) void finalize()
{
	Py_DECREF(pName);
	Py_DECREF(pArgs);
	Py_DECREF(pModule);
	Py_DECREF(pDict);
	Py_DECREF(pFunc);
	Py_DECREF(pvalue);
	Py_Finalize();
}



//int main(int argc, char** argv)
//{
//	
//	TfInit();
//
//	double t = (double)cvGetTickCount();
//	int f = TfTest("1.jpg");
//	t = ((double)cvGetTickCount() - t) / (1000 * (double)cvGetTickFrequency());
//	std::cout << "R:" << f << std::endl << "first time:" << t << std::endl;
//
//	cv::Mat src = cv::imread("1.jpg");
//	cv::imshow("src",src);
//	cv::waitKey(0);
//	
//	double t2 = (double)cvGetTickCount();
//	int f1 = TfTest("1.jpg");
//	t2 = ((double)cvGetTickCount() - t2) / (1000 * (double)cvGetTickFrequency());
//	std::cout << "R:" << f1 << std::endl << "secound time:" << t2 << std::endl;
//
//	return 0;
//
//}



#include <cstdlib>
#include <iostream>
#include <windows.h>
#include <stdlib.h>
#include <winsock.h>
#include <time.h>
#include "dbghelp.h"
#include <process.h>

#pragma comment(lib, "Ws2_32.lib")
#pragma comment(lib, "Dbghelp.lib")

void findIP(char *ip, int size)
{
	WORD v = MAKEWORD(1, 1);
	WSADATA wsaData;
	WSAStartup(v, &wsaData); // 加载套接字库

	struct hostent *phostinfo = gethostbyname("");
	char *p = inet_ntoa(*((struct in_addr *)(*phostinfo->h_addr_list)));
	strcpy_s(ip, size-1, p);
	ip[size - 1] = '\0';
	WSACleanup();
}

char *IP = "127.0.0.1";

bool sendmsg(char * sendData);
unsigned __stdcall UpdateTicketsLoop(LPVOID arguments);

bool sendToServer()
{
	WORD socketVersion = MAKEWORD(2, 2);
	WSADATA wsaData;
	if (WSAStartup(socketVersion, &wsaData) != 0)
	{
		return 0;
	}
//	//启动发送消息线程
//	HANDLE threadHandle;
//	unsigned threadId = 0;
//	threadHandle = (HANDLE)_beginthreadex(NULL, 0, UpdateTicketsLoop, NULL, 0, &threadId);
	//创建接收消息socket
	SOCKET sclient = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
	sockaddr_in sin;
	sin.sin_family = AF_INET;
	sin.sin_port = htons(8001);
	//findIP(IP, sizeof(IP));
	//printf("%s\n", IP);
	sin.sin_addr.S_un.S_addr = inet_addr(IP);
	int len = sizeof(sin);
	// 绑定套接字  
	int nRet = bind(sclient, (SOCKADDR*)&sin, sizeof(SOCKADDR));
	while (true)
	{
		char recvData[255];
		int ret = recvfrom(sclient, recvData, 255, 0, (sockaddr *)&sin, &len);
			
		if (ret > 0)
		{
			recvData[ret] = 0x00;
			if (strcmp(recvData, "done") == 0)break;
			printf("\n");
			printf(recvData);
			int f1 = TfTest(recvData);
			printf("\n%d\n", f1);
			char s = f1 + '0';
			sendmsg(&s);

		}

	}


	closesocket(sclient);
	WSACleanup();
	return true;
}


int main()
{
	TfInit();
	sendToServer();
	 
	return 0;
}


bool sendmsg(char * sendData)
{
	SOCKET clienttoserver = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
	sockaddr_in sin;
	sin.sin_family = AF_INET;
	sin.sin_port = htons(8000);
	//	findIP(IP, sizeof(IP));
	sin.sin_addr.S_un.S_addr = inet_addr(IP);
	int len = sizeof(sin);
	//const char * sendData = "hellow server";
	//while (true)
	{
		Sleep(1000);
		sendto(clienttoserver, sendData, strlen(sendData), 0, (sockaddr *)&sin, len);
		printf(sendData);
		printf("\n");
	}
	closesocket(clienttoserver);
	return true;
}





//启动发送线程
unsigned __stdcall UpdateTicketsLoop(LPVOID arguments)
{
	SOCKET clienttoserver = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
	sockaddr_in sin;
	sin.sin_family = AF_INET;
	sin.sin_port = htons(8000);
//	findIP(IP, sizeof(IP));
	sin.sin_addr.S_un.S_addr = inet_addr(IP);
	int len = sizeof(sin);
	const char * sendData = "hellow server";
	while (true)
	{
		Sleep(1000);
		sendto(clienttoserver, sendData, strlen(sendData), 0, (sockaddr *)&sin, len);
		printf(sendData);
		printf("\n");
	}
	closesocket(clienttoserver);
	return 0;
}
