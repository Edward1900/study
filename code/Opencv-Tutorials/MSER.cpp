/*
����Ӧ��ֵ
MSER
*/
#include "stdafx.h"
#include "opencv2/opencv.hpp"
#include <iostream>
#include <iomanip>
using namespace std;
using namespace cv;



std::vector<cv::Rect>mserGet(cv::Mat image, cv::Mat& image2)
{
	cv::Mat gray, gray_neg;
	//�Ҷ�ת��
	cv::cvtColor(image, gray, CV_BGR2GRAY);
	//equalizeHist(gray, gray);
	//ȡ��ֵ�Ҷ�
	gray_neg = 255 - gray;
	std::vector<std::vector<cv::Point> > regcontours;
	std::vector<std::vector<cv::Point> > charcontours;
	// ����MSER����
	//��ʾ�Ҷ�ֵ�ı仯������⵽���������ķ�Χ�����ı仯�ʣ���С�ı任��
	cv::Ptr<cv::MSER> mesr1 = cv::MSER::create(5, 60, 14400, 0.25, 0.3);
	//cv::Ptr<cv::MSER> mesr1 = cv::MSER::create(2, 100, 800, 0.7, 0.3);
	cv::Ptr<cv::MSER> mesr2 = cv::MSER::create(5, 5000, 10000, 1.2, 0.3);
	std::vector<cv::Rect> box1;
	std::vector<cv::Rect> box2;
	// MSER+ ���
	mesr1->detectRegions(gray, regcontours, box1);
	// MSER-����
	mesr2->detectRegions(gray_neg, charcontours, box2);
	cv::Mat mserMat = cv::Mat::zeros(image.size(), CV_8UC1);
	cv::Mat mserNegMat = cv::Mat::zeros(image.size(), CV_8UC1);
	for (int i = regcontours.size() - 1; i >= 0; i--)
	{
		// ���ݼ�����������mser+���
		const std::vector<cv::Point>tmp = regcontours[i];
		for (int j = 0; j < tmp.size(); j++)
		{
			cv::Point p = tmp[j];
			mserMat.at<uchar>(p) = 255;
		}
	}
	for (int i = charcontours.size() - 1; i >= 0; i--)
	{
		// ���ݼ�����������mser-���
		const std::vector<cv::Point>tmp = charcontours[i];
		for (int j = 0; j < tmp.size(); j++)
		{
			cv::Point p = tmp[j];
			mserNegMat.at<uchar>(p) = 255;
		}
	}
	// mser������
	cv::Mat resultMat;
	// mser+��mser-λ�����
	//resultMat = mserMat&mserNegMat;
	resultMat = mserMat;
	// �ղ������ӷ�϶
	cv::morphologyEx(resultMat, resultMat, cv::MORPH_CLOSE, cv::Mat::ones(1, 20, CV_8UC1));

	// Ѱ���ⲿ����
	std::vector<std::vector<cv::Point> > contours;
	vector<Vec4i> hierarchy;
	cv::findContours(resultMat, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
	RNG rng(12345);
	std::vector<cv::Rect> candidates;
	for (int i = 0; i < contours.size(); i++)
	{

		cv::Rect rect = cv::boundingRect(contours[i]);

		drawContours(image2, contours, i, Scalar(rng.uniform(1, 255), rng.uniform(1, 255), rng.uniform(1, 255)), -1, 8, hierarchy);


		//double ratio = 1. * rect.width / rect.height;

		if (rect.width>15 && rect.height > 15 && rect.width<100 && rect.height < 100 && ratio > 0.6 && ratio < 1.8)
		{


			rectangle(image2, Rect(rect.tl().x - 5, rect.tl().y - 5, rect.width + 5, rect.height + 5), Scalar(255, 255, 255), 1, 8, 0);

			candidates.push_back(rect);


		}

	}

	return candidates;
}


void AdThreshold(Mat& image1)
{
	//Mat image = imread(argv[1]);
	Mat image;
	cvtColor(image1, image, CV_RGB2GRAY); //ԭͼ������ͨ��������ͼҲ����ͨ��
	Mat dst;

	cout << image.type() << image.rows << image.cols << endl;

	adaptiveThreshold(image,             // input image
		dst,    // output binary image
		255,               // max value for output
		cv::ADAPTIVE_THRESH_GAUSSIAN_C, // adaptive method����
		cv::THRESH_BINARY, // threshold type
		21,         // size of the block
		10); //diff

			 //GaussianBlur(imageIntegral, imageIntegral, Size(3, 3), 0, 0, BORDER_DEFAULT);
	Mat element = getStructuringElement(MORPH_ELLIPSE,
		Size(2, 2),
		Point(1, 1));


	imshow("res", dst);

}





int main(int argc, char** argv)
{

	Mat img = imread("src3.jpg", 1);
	imshow("img", img);
	//imwrite("test.bmp", img);//imwrite("test.png", img);

	//Mat tem1(img.rows, img.cols, img.type());
	Mat tem = img.clone();

	mserGet(img, tem);
	//AdThreshold(img);

	imshow("dst", tem);


	waitKey(0);

}