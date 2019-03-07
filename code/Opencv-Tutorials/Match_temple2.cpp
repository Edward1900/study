/**
work by yafeng


ģ��ƥ�䣬��״ƥ��
1����������ֵƥ��
����ģ��ͼ������һ����������ͼ����ĳ�ֱȶԷ�ʽ(�����֮�ͣ���ضȵ�)�������ƶȣ�
�ڽ��ͼ����Ѱ��������Сֵ��������ͬ��ЩֵԽ��Խ���ƣ���Щ�෴����Ϊƥ������
������matchTemplate(src, templ, result, match_method)

2����������ƥ��
�����ģ����������������ľ����������������ͼ��������������������ģ�����������
�Ƚϣ���ֵ��С�ļ�Ϊƥ������
������matchShapes(contours1, contours2, CV_CONTOURS_MATCH_I1, 0.0);

*/
#include "stdafx.h"
#include "opencv2/opencv.hpp"
#include <iostream>
#include <iomanip>
using namespace std;
using namespace cv;

Mat TempleMatch(int match_method, Mat&templ, Mat&img, Point& matchLoc, double* matchValue);
int ObjectMatch0(Mat&templ, Mat& srcImg, float th1);
int ObjectMatch1(Mat& srcImg, Mat& srcImg2, float th1);
int ObjectMatch(Mat& srcImg, Mat&templ, Mat& srcImg2, float th1);

void main()
{
	Mat srcImg0 = imread("tp.jpg");  //ģ��ͼ��
									 //imshow("src0", srcImg0);

	Mat srcImg1 = imread("tp1.jpg");  //ģ��ͼ��
									  //imshow("src1", srcImg1);

	Mat srcImg2 = imread("src1.BMP");  //������ͼƬ
									   //imshow("src2", srcImg2);

									   //ShapeMatch(srcImg0, srcImg2);
									   //ObjectMatch0(srcImg1, srcImg2,90);
									   //ObjectMatch1(srcImg0, srcImg2, 100);
	ObjectMatch(srcImg0, srcImg1, srcImg2, 100);

	waitKey(0);
}


int ObjectMatch0(Mat&templ, Mat& srcImg, float th1)
{


	float th = th1 / 10000;
	Point matchLoc;
	double matchValue;
	Mat resault = TempleMatch(1, templ, srcImg, matchLoc, &matchValue);

	vector<Point> points;

	for (int j = 0; j < resault.rows; j++)
	{
		for (int i = 0; i < resault.cols; i++)
		{
			if (resault.at<float>(j, i) <= th)
			{
				points.push_back(Point(i, j));
				cout << "matchValue=" << resault.at<float>(j, i) << endl;
			}
		}

	}

	for (size_t i = 0; i < points.size(); i++)
	{
		rectangle(srcImg, points[i], Point(points[i].x + templ.cols, points[i].y + templ.rows), Scalar(0, 255, 0), 1, 8, 0);
		//cout << "matchValue=" << matchValue << endl;

	}




	imwrite("dst0.jpg", srcImg);


	return 0;

}

int ObjectMatch1(Mat& srcImg, Mat& srcImg2, float th1)
{

	float th = th1 / 100;
	cvtColor(srcImg, srcImg, CV_BGR2GRAY);
	threshold(srcImg, srcImg, 150, 255, CV_THRESH_BINARY);

	vector<vector<Point>> contours;
	vector<Vec4i> hierarcy;
	findContours(srcImg, contours, hierarcy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

	//Mat dstImg = srcImg2.clone();
	Mat src_gray;
	cvtColor(srcImg2, src_gray, CV_BGR2GRAY);
	threshold(src_gray, src_gray, 150, 255, CV_THRESH_BINARY);
	//imshow("src2_1", srcImg2);
	vector<vector<Point>> contours2;

	vector<Vec4i> hierarcy2;
	findContours(src_gray, contours2, hierarcy2, CV_RETR_TREE, CV_CHAIN_APPROX_NONE);

	for (int i = 0; i<contours2.size(); i++)
	{
		double Areadiff = abs(contourArea(contours2[i]) - contourArea(contours[0]));

		double matchRate = matchShapes(contours[0], contours2[i], CV_CONTOURS_MATCH_I1, 0.0);//��״ƥ��:ֵԽСԽ����
																							 //cout << "index=" << i << "---" << setiosflags(ios::fixed) << matchRate << endl;//setiosflags(ios::fixed)���ö��㷽ʽ��ʾʵ����������ͬλ������ͬ��ʽ���
		if ((matchRate <= th) && (Areadiff <= 10000))
		{
			drawContours(srcImg2, contours2, i, Scalar(0, 255, 0), 2, 8);

		}


	}

	imwrite("dst1.jpg", srcImg2);


	return 0;

}


int ObjectMatch(Mat& srcImg, Mat&templ, Mat& srcImg2, float th1)
{
	float th = th1 / 100;

	cvtColor(srcImg, srcImg, CV_BGR2GRAY);
	threshold(srcImg, srcImg, 150, 255, CV_THRESH_BINARY);

	vector<vector<Point>> contours;
	vector<Vec4i> hierarcy;
	findContours(srcImg, contours, hierarcy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

	//Mat dstImg = srcImg2.clone();
	Mat src_gray;
	cvtColor(srcImg2, src_gray, CV_BGR2GRAY);
	threshold(src_gray, src_gray, 150, 255, CV_THRESH_BINARY);
	//imshow("src2_1", srcImg2);
	vector<vector<Point>> contours2;

	vector<Vec4i> hierarcy2;
	findContours(src_gray, contours2, hierarcy2, CV_RETR_TREE, CV_CHAIN_APPROX_NONE);

	for (int i = 0; i<contours2.size(); i++)
	{
		double Areadiff = abs(contourArea(contours2[i]) - contourArea(contours[0]));

		double matchRate = matchShapes(contours[0], contours2[i], CV_CONTOURS_MATCH_I1, 0.0);//��״ƥ��:ֵԽСԽ����
																							 //cout << "index=" << i << "---" << setiosflags(ios::fixed) << matchRate << endl;//setiosflags(ios::fixed)���ö��㷽ʽ��ʾʵ����������ͬλ������ͬ��ʽ���
		if ((matchRate <= th) && (Areadiff <= 10000))
		{
			drawContours(srcImg2, contours2, i, Scalar(0, 255, 0), 2, 8);
			Rect r0 = boundingRect(contours2[i]);
			//RotatedRect r1 = minAreaRect(contours2[i]);
			//boxPoints(r1, cont);
			//Rect r2 = r1.boundingRect();
			//rectangle(srcImg2,r0, Scalar(0, 0, 255),1,8,0);
			Mat t_img = srcImg2(r0).clone();
			Point matchLoc;
			double matchValue;
			TempleMatch(1, templ, t_img, matchLoc, &matchValue);
			//rectangle(srcImg2,r2, Scalar(255, 0, 0),1, 8, 0);

			//! [imshow]
			/// Show me what you got
			Point matchLoc_d(r0.x + matchLoc.x, r0.y + matchLoc.y);
			rectangle(srcImg2, matchLoc_d, Point(matchLoc_d.x + templ.cols, matchLoc_d.y + templ.rows), Scalar::all(90), 1, 8, 0);
			cout << "matchValue=" << matchValue << endl;

			//! [imshow]
		}



		//imshow("dst", dstImg2);

	}

	imwrite("dst.jpg", srcImg2);

	return 0;

}


Mat TempleMatch(int match_method, Mat&templ, Mat&img, Point& matchLoc, double* matchValue)
{
	//! [copy_source]
	/// Source image to display
	bool use_mask = false;
	Mat  result, mask;
	//img.copyTo(img_display);
	//! [copy_source]

	//! [create_result_matrix]
	/// Create the result matrix
	/*int result_cols = img.cols - templ.cols + 1;
	int result_rows = img.rows - templ.rows + 1;
	cout << "result_cols=" << result_cols << endl;
	cout << "result_rows=" << result_rows << endl;
	result.create(result_rows, result_cols, CV_32FC1);*/
	//! [create_result_matrix]

	//! [match_template]
	/// Do the Matching and Normalize
	bool method_accepts_mask = (CV_TM_SQDIFF == match_method || match_method == CV_TM_CCORR_NORMED);
	if (use_mask && method_accepts_mask)
	{
		matchTemplate(img, templ, result, match_method, mask);
	}
	else
	{
		matchTemplate(img, templ, result, match_method);
	}
	//! [match_template]

	//! [normalize]

	//normalize(result, result, 0, 255, NORM_MINMAX, -1, Mat());
	//! [normalize]

	//! [best_match]
	/// Localizing the best match with minMaxLoc
	double minVal; double maxVal; Point minLoc; Point maxLoc;
	//Mat res;//(result.rows, result.cols, CV_8U);
	//result.convertTo(res, CV_8U);

	minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, Mat());
	//! [best_match]

	//! [match_loc]
	/// For SQDIFF and SQDIFF_NORMED, the best matches are lower values. For all the other methods, the higher the better
	if (match_method == TM_SQDIFF || match_method == TM_SQDIFF_NORMED)
	{
		matchLoc = minLoc;
		*matchValue = minVal;
		//cout << minVal << endl;
	}
	else
	{
		matchLoc = maxLoc;
		*matchValue = maxVal;
	}
	//! [match_loc]
	//rectangle(result, matchLoc, Point(matchLoc.x + templ.cols, matchLoc.y + templ.rows), Scalar::all(0), 2, 8, 0);
	//cout << result << endl;
	//imshow("dst", result);


	return result;
}
