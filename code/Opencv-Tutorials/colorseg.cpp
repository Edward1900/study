Mat hsvreg(Mat& img)

{


	Mat imgHSV;

	//蓝色笔筒颜色的HSV范围

	int iLowH = 165 ;
	int iHighH = 180 ;


	int iLowS = 50;
	int iHighS = 255;


	int iLowV = 70;
	int iHighV = 255;



	cvtColor(img, imgHSV, COLOR_BGR2HSV);//转为HSV



	//imwrite("hsv.jpg", imgHSV);

	Mat imgThresholded;



	inRange(imgHSV, Scalar(iLowH, iLowS, iLowV), Scalar(iHighH, iHighS, iHighV), imgThresholded); //Threshold the image  

																								  //开操作 (去除一些噪点)  如果二值化后图片干扰部分依然很多，增大下面的size
	
	Mat element = getStructuringElement(MORPH_RECT, Size(3, 3));

	morphologyEx(imgThresholded, imgThresholded, MORPH_OPEN, element);

	morphologyEx(imgThresholded, imgThresholded, MORPH_CLOSE, element);


	vector<vector<Point>> conours;
	vector<Vec4i> hierarchy;
	//
	//	InversColor(&binaryImage);
	findContours(imgThresholded, conours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point()); // 只测外围连通：RETR_EXTERNAL
	vector<Rect> boundRect(conours.size());
	Point p = Point(5, 5);

	double area = 0;
	int id = 0;
	//	cout << conours.size() << endl;
	Mat mrk = Mat(imgThresholded.rows, imgThresholded.cols, imgThresholded.type(), Scalar(0));

	for (size_t i = 0; i < conours.size(); i++)
	{
		double a = contourArea(conours[i]);
		boundRect[i] = boundingRect(Mat(conours[i]));

		//cout << "i:" << i << "  area:" << a << endl;
		if (a>area)
		{
			area = a;
			id = i;
		}
		if (a >= 500)
		{
		//	rectangle(srcImage, Point(boundRect[i].tl().x, boundRect[i].tl().y), Point(boundRect[i].br().x, boundRect[i].br().y), Scalar(0, 255, 0), 1, 8, 0);
			drawContours(mrk, conours, i, Scalar(255), -1, 8, hierarchy);
		}

	}

	dilate(mrk, mrk, element, Point(-1, -1), 3);//白色区域膨胀  
	return mrk;

	//imwrite("end.jpg", img);


}



int YCrCbSeg( Mat& srcImage, Mat &marks)
{


	Mat dstImage, CrCbImage;

	cvtColor(srcImage, CrCbImage, CV_BGR2YCrCb);


	//切割图像  
	vector<Mat> channels, channels1;
	Mat Y, Cr, Cb, Cb1;
	split(CrCbImage, channels);
	Y = channels.at(0);
	Cr = channels.at(1);
	Cb = channels.at(2);


	dstImage.create(CrCbImage.rows, CrCbImage.cols, CV_8UC1);

	for (int j = 1; j < Y.rows - 1; j++)
	{
		uchar* currentCr = Cr.ptr< uchar>(j);
		uchar* currentCb = Cb.ptr< uchar>(j);
		uchar* current = dstImage.ptr< uchar>(j);
		for (int i = 1; i < Y.cols - 1; i++)
		{
			//if ((currentCr[i] > 60) && (currentCr[i] < 120))// || (currentCb[i] > 125)|| (currentCr[i] > 160))
			//if (currentCr[i] > 165)
			if ((currentCr[i] > 150) && (currentCr[i] < 200))//&& (currentCb[i] > 100) && (currentCb[i] < 130))
				current[i] = 255;
			else
				current[i] = 0;

		}
	}
	//erode(dstImage, dstImage, Mat());  
	Mat element = getStructuringElement(MORPH_CROSS, Size(3, 3));
	dilate(dstImage, dstImage, element, Point(-1, -1), 3);//白色区域膨胀  
	erode(dstImage, dstImage, element, Point(-1, -1), 3);//白色区域腐蚀  

	vector<vector<Point>> conours;
	vector<Vec4i> hierarchy;
	//
	//	InversColor(&binaryImage);
	findContours(dstImage, conours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point()); // 只测外围连通：RETR_EXTERNAL
	vector<Rect> boundRect(conours.size());
	Point p = Point(5, 5);

	double area = 0;
	int id = 0;
	//	cout << conours.size() << endl;
	Mat mrk = Mat(dstImage.rows, dstImage.cols, dstImage.type(), Scalar(0));

	for (size_t i = 0; i < conours.size(); i++)
	{
		double a = contourArea(conours[i]);
		boundRect[i] = boundingRect(Mat(conours[i]));

		//cout << "i:" << i << "  area:" << a << endl;
		if (a>area)
		{
			area = a;
			id = i;
		}
		if (a >= 10)
		{
			rectangle(srcImage, Point(boundRect[i].tl().x, boundRect[i].tl().y), Point(boundRect[i].br().x, boundRect[i].br().y), Scalar(0, 255, 0), 1, 8, 0);
			drawContours(mrk, conours, i, Scalar(255), -1, 8, hierarchy);
		}

	}

	dilate(mrk, mrk, element, Point(-1, -1), 3);//白色区域膨胀  

	marks = mrk.clone();

	return 0;
}
