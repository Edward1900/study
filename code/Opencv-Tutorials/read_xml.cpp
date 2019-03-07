// opencvproject.cpp : 定义控制台应用程序的入口点。
//
#include "stdafx.h"
#include "opencv2/core.hpp"
#include <iostream>
#include <string>
using std::string;
using std::cout;
using std::endl;
using std::cerr;
using std::ostream;
using namespace cv;




struct MyData
{
	string name;
	int id;
	double F1;
	double F2;
	double F3;


	void write(FileStorage& fs) const //Write serialization for this class
	{
		fs << "{" << "name" << name << "ID" << id << "F1" << F1 << "F2" << F2 << "F3" << F3 << "}";
	}
	void read(const FileNode& node)  //Read serialization for this class
	{
		name = (string)node["name"];
		id = (int)node["ID"];
		F1 = (double)node["F1"];
		F2 = (double)node["F2"];
		F3 = (double)node["F3"];

	}
	MyData() :
		name("name"), id(0), F1(1.0), F2(1.0), F3(1.0)
	{
	}

};



//These write and read functions must exist as per the inline functions in operations.hpp
static void write(FileStorage& fs, const std::string&, const MyData& x) {
	x.write(fs);
}
static void read(const FileNode& node, MyData& x, const MyData& default_value = MyData()) {
	if (node.empty())
		x = default_value;
	else
		x.read(node);
}
static ostream& operator<<(ostream& out, const MyData& m) {
	out << "{ name = " << m.name << ",";
	out << "ID = " << m.id << ", ";
	out << "F1 = " << m.F1 << ", ";
	out << "F2 = " << m.F2 << ", ";
	out << "F3 = " << m.F3 << "}";
	return out;
}

int write_xml(string filename, MyData m)
{
	FileStorage fs(filename, FileStorage::WRITE);
	fs << "mdata" << m;

	return 0;

}

int read_xml(string filename, MyData& m)
{
	FileStorage fs(filename, FileStorage::READ);
	if (!fs.isOpened())
	{
		cerr << "failed to open " << filename << endl;
		return 1;
	}

	fs["mdata"] >> m;
	return 0;
}


int main(int ac, char** av)
{
	string filename = "dat.xml";

	MyData m;
	m.name = "feature";
	m.id = 1; m.F1 = 2.0;m.F2 = 2.1;m.F3 = 2.2;
	cout << "write mdata\n";
	cout << m << endl;
	//write
	write_xml(filename, m);


	//read
	MyData m1;
	read_xml(filename, m1);

	cout << "read mdata\n";
	cout << m1 << endl;

	return 0;
}