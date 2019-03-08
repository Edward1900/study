/*
1��������
��ˮ��ָ������һ�ֻ����������۵���ѧ��̬ѧ�ķָ���������˼���ǰ�ͼ�����ǲ��ѧ�ϵ����˵�ò��
ͼ����ÿһ�����صĻҶ�ֵ��ʾ�õ�ĺ��θ߶ȣ�ÿһ���ֲ���Сֵ����Ӱ�������Ϊ��ˮ�裬����ˮ��ı߽����γɷ�ˮ�롣
��ˮ��ĸ�����γɿ���ͨ��ģ����������˵������ÿһ���ֲ���Сֵ���棬�̴�һ��С�ף�Ȼ�������ģ����������ˮ�У�
���Ž���ļ��ÿһ���ֲ���Сֵ��Ӱ��������������չ����������ˮ���ϴ�������ӣ����γɷ�ˮ�롣
��ˮ��ļ��������һ��������ע���̡���ˮ��ȽϾ���ļ��㷽����L. Vincent����ġ��ڸ��㷨�У���ˮ�������������裬
һ����������̣�һ������û���̡����ȶ�ÿ�����صĻҶȼ����дӵ͵�������Ȼ���ڴӵ͵���ʵ����û�����У�
��ÿһ���ֲ���Сֵ��h�׸߶ȵ�Ӱ��������Ƚ��ȳ�(FIFO)�ṹ�����жϼ���ע����ˮ��任�õ���������ͼ��ļ�ˮ��ͼ��
��ˮ��֮��ı߽�㣬��Ϊ��ˮ�롣��Ȼ����ˮ���ʾ��������ͼ�񼫴�ֵ�㡣��ˣ�Ϊ�õ�ͼ��ı�Ե��Ϣ��
ͨ�����ݶ�ͼ����Ϊ����ͼ��

��ˮ���㷨���ʵ��˼·;ͬʱ�ݹ��Ѱ�Ҹ�����ǵ�����ڵ����ص�(��עˮ����)�������ڵ����ص����ǵ��غϣ���õ�Ϊ��ˮ��ĵ㣬
�����ˮ������ڶ����ǵ㣬�������ݶ�ֵ�����仮�ֵ��ݶȽ�С������
*/

//2���ӿڵ������£�
Mat water(Mat srcImg, Rect rect)

{
	imshow("src", srcImg);
	Mat copyImg = srcImg.clone();

	//1.������ͼ��markers

	Mat markers(srcImg.size(), CV_8UC1, Scalar(0));

	//��Ǳ���

	rectangle(markers, Point(1, 1), Point(srcImg.cols - 2, srcImg.rows - 2), Scalar(50), 1, 8);

	rectangle(markers, Point(5, srcImg.rows - 5), Point(srcImg.cols - 2, srcImg.rows - 2), Scalar(150), -1, 8);

	//���ǰ��
	rectangle(markers, rect, Scalar(255), -1, 8);

	imshow("markers-input", markers);

	//2.���ڱ��ͼ��ķ�ˮ���㷨

	//��markersת����32λ��ͨ��ͼ�񣨷�ˮ�뺯��Ҫ��

	markers.convertTo(markers, CV_32S);

	//��ˮ���㷨

	watershed(srcImg, markers);

	//��markersת����8λ��ͨ��

	markers.convertTo(markers, CV_8UC1);

	//imshow("markers-output", markers);



	Mat resault(srcImg.size(), CV_8UC3, Scalar(255, 255, 255));
	//Mat result = srcImg*0.5 + copyImg*0.5;

	threshold(markers, markers, 200, 255, THRESH_BINARY);
	srcImg.copyTo(resault, markers);


	imshow("dst", resault);

	cvWaitKey(0);

	return resault;

}

int main_w(int argc, char** argv)
{
	string filename = "dm.png";
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


	Rect rect(Point(110, 130), Point(220, 290));
	water(image, rect);
	//rectangle(image, rect, Scalar(255), 2, 8);


	cvWaitKey(0);

	return 0;
}
