/*
��ͼ�񸽼Ӷ���߿����Ĥ����ͼ�μ�����
1�����Ӷ���߿򣨳��ھ����ʹ�ã�
������copyMakeBorder( src, dst, top, bottom, left, right, borderType, value );
����:
src: ԭͼ��
dst: Ŀ��ͼ��
top, bottom, left, right: ���߽�Ŀ�ȣ��˴�����Ϊԭͼ��ߴ��5%��
borderType: �߽����ͣ��˴�����ѡ�����߽���߸��Ʊ߽硣
value: ��� borderType ������ BORDER_CONSTANT, ��ֵ�������߽����ء�
2����ӿ��Ӧ���Ǹ���0��Ĥ
Result.row(0).setTo(Scalar(0));             // �ϱ߽�
Result.row(Result.rows-1).setTo(Scalar(0)); // �±߽�
Result.col(0).setTo(Scalar(0));             // ��߽�
Result.col(Result.cols-1).setTo(Scalar(0)); // �ұ߽�

3����ͼ�μ�����
ͼ�κ�����
Point ��ͼ���ж��� 2D ��
����Լ�Ϊ��ʹ�� Scalar()
���� line() �� ֱ��
���� ellipse() �� ��Բ
���� rectangle() �� ����
���� circle() �� Բ
���� fillPoly() �� ���Ķ����
���ֺ�����
putText( image, "Testing text ", org, rng.uniform(0,8),
rng.uniform(0,100)*0.05+0.1, randomColor(rng), rng.uniform(1, 10), lineType);
�����У�
�� image �ϻ������� ��Testing text �� ��
���ֵ����½ǽ��õ� org ָ����
�����������һ����  ֮������������塣
��������ű������ñ��ʽ rng.uniform(0, 100)x0.05 + 0.1 ָ��(��ʾ���ķ�Χ�� )��
�������ɫ������� (��Ϊ randomColor(rng))��
����Ĵ�ϸ��Χ�Ǵ� 1 �� 10, ��ʾΪ rng.uniform(1,10) ��*/
#include "stdafx.h"
#include "opencv2/opencv.hpp"
#include <iostream>
#include <iomanip>
using namespace std;
using namespace cv;
Scalar value;
RNG rng(12345);

void main()
{
	Mat src, dst;

	/// װ��ͼ��
	src = imread("2.jpg");
	imshow("src", src);

	putText(src, "opencv goes from 0 to N", Point(80, 80), 1, 1.5, Scalar(0, 255, 0), 1, 8);
	rectangle(src, Rect(10, 10, 50, 50), Scalar(0, 255, 0), 1, 8);

	int top = (int)(0.05*src.rows), bottom = (int)(0.05*src.rows);
	int left = (int)(0.05*src.cols), right = (int)(0.05*src.cols);
	dst = src;
	value = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
	copyMakeBorder(src, dst, top, bottom, left, right, BORDER_CONSTANT, value);

	imshow("dst", dst);

	waitKey(0);
}