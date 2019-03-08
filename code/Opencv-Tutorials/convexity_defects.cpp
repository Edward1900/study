/*
͹���㷨��

��ʵ�ܼ򵥣�������һ����͹�����Χס���еĵ㡣�ͺ��������������ͼ������һ����������Ƥ�����ȫ��Χ����һ�����㷨��ϸ���裺

1. �ҵ����е���������y��С�ĵ磬Ҳ������Щ����������ĵ㣬��Ϊp0��

2. Ȼ������������õ��������x��֮��нǵ�����ֵ������Щ�㰴�������͵������ֵ�Ӵ�С��������õĵ��Ϊp1, p2, p3,



*/


void convexity_defects(Mat src)
{
	Mat src_copy = src.clone();
	Mat threshold_output, src_gray;
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	cvtColor(src, src_gray, CV_BGR2GRAY);
	imshow("src", src);
	imshow("src_gray", src_gray);
	/// Detect edges using Threshold
	threshold(src_gray, threshold_output, 60, 255, THRESH_BINARY);
	imshow("threshold_output", threshold_output);
	/// Find contours
	findContours(threshold_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

	/// Find the convex hull object for each contour
	vector<vector<Point> >hull(contours.size());
	// Int type hull
	vector<vector<int>> hullsI(contours.size());
	// Convexity defects
	vector<vector<Vec4i>> defects(contours.size());

	for (size_t i = 0; i < contours.size(); i++)
	{
		convexHull(Mat(contours[i]), hull[i], false);
		// find int type hull
		convexHull(Mat(contours[i]), hullsI[i], false);
		// get convexity defects
		convexityDefects(Mat(contours[i]), hullsI[i], defects[i]);

	}

	/// Draw contours + hull results
	Mat drawing = Mat::zeros(threshold_output.size(), CV_8UC3);
	for (size_t i = 0; i< contours.size(); i++)
	{
		Scalar color = Scalar(210, 50, 255);
		drawContours(drawing, contours, i, color, 1, 8, vector<Vec4i>(), 0, Point());
		drawContours(drawing, hull, i, color, 1, 8, vector<Vec4i>(), 0, Point());

		// draw defects
		size_t count = contours[i].size();
		std::cout << "Count : " << count << std::endl;
		if (count < 300)
			continue;

		vector<Vec4i>::iterator d = defects[i].begin();

		while (d != defects[i].end()) {
			Vec4i& v = (*d);
			//if(IndexOfBiggestContour == i)
			{

				int startidx = v[0];
				Point ptStart(contours[i][startidx]); // point of the contour where the defect begins
				int endidx = v[1];
				Point ptEnd(contours[i][endidx]); // point of the contour where the defect ends
				int faridx = v[2];
				Point ptFar(contours[i][faridx]);// the farthest from the convex hull point within the defect
				int depth = v[3] / 256; // distance between the farthest point and the convex hull

				if (depth > 2 && depth < 500)
				{
					line(drawing, ptStart, ptFar, CV_RGB(0, 255, 0), 2);
					line(drawing, ptEnd, ptFar, CV_RGB(0, 255, 0), 2);
					circle(drawing, ptStart, 4, Scalar(255, 0, 0), 2);
					circle(drawing, ptEnd, 4, Scalar(255, 0, 0), 2);
					circle(drawing, ptFar, 4, Scalar(100, 0, 255), 2);
				}

				/*printf("start(%d,%d) end(%d,%d), far(%d,%d)\n",
				ptStart.x, ptStart.y, ptEnd.x, ptEnd.y, ptFar.x, ptFar.y);*/
			}
			d++;
		}

	}
	/// Show in a window
	namedWindow("Hull demo", CV_WINDOW_AUTOSIZE);
	imshow("Hull demo", drawing);


}

int main(int argc, char** argv)
{
	string filename = "QWQW.jpg";
	if (filename.empty())
	{
		cout << "\nDurn, empty filename" << endl;
		return 1;
	}
	Mat image = imread(filename, 1);
	if (image.empty())
	{
		cout << "\n Durn, couldn't read image filename " << filename << endl;
		return 1;
	}

	convexity_defects(image);

	cvWaitKey(0);

	return 0;
}