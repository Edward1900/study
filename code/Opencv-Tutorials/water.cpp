/*
1，概述：
分水岭分割方法，是一种基于拓扑理论的数学形态学的分割方法，其基本思想是把图像看作是测地学上的拓扑地貌，
图像中每一点像素的灰度值表示该点的海拔高度，每一个局部极小值及其影响区域称为集水盆，而集水盆的边界则形成分水岭。
分水岭的概念和形成可以通过模拟浸入过程来说明。在每一个局部极小值表面，刺穿一个小孔，然后把整个模型慢慢浸入水中，
随着浸入的加深，每一个局部极小值的影响域慢慢向外扩展，在两个集水盆汇合处构筑大坝，即形成分水岭。
分水岭的计算过程是一个迭代标注过程。分水岭比较经典的计算方法是L. Vincent提出的。在该算法中，分水岭计算分两个步骤，
一个是排序过程，一个是淹没过程。首先对每个像素的灰度级进行从低到高排序，然后在从低到高实现淹没过程中，
对每一个局部极小值在h阶高度的影响域采用先进先出(FIFO)结构进行判断及标注。分水岭变换得到的是输入图像的集水盆图像，
集水盆之间的边界点，即为分水岭。显然，分水岭表示的是输入图像极大值点。因此，为得到图像的边缘信息，
通常把梯度图像作为输入图像。

分水岭算法编程实现思路;同时递归的寻找各个标记点的相邻的像素点(即注水操作)，当相邻的像素点与标记点重合，则该点为分水岭的点，
如果分水岭点属于多个标记点，计算其梯度值，将其划分到梯度较小的区域。
*/

//2，接口调用如下：
Mat water(Mat srcImg, Rect rect)

{
	imshow("src", srcImg);
	Mat copyImg = srcImg.clone();

	//1.定义标记图像markers

	Mat markers(srcImg.size(), CV_8UC1, Scalar(0));

	//标记背景

	rectangle(markers, Point(1, 1), Point(srcImg.cols - 2, srcImg.rows - 2), Scalar(50), 1, 8);

	rectangle(markers, Point(5, srcImg.rows - 5), Point(srcImg.cols - 2, srcImg.rows - 2), Scalar(150), -1, 8);

	//标记前景
	rectangle(markers, rect, Scalar(255), -1, 8);

	imshow("markers-input", markers);

	//2.基于标记图像的分水岭算法

	//将markers转换成32位单通道图像（分水岭函数要求）

	markers.convertTo(markers, CV_32S);

	//分水岭算法

	watershed(srcImg, markers);

	//将markers转换成8位单通道

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
