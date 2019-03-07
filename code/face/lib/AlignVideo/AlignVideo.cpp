//#include <opencv2/opencv.hpp>
//#include <opencv2/objdetect.hpp>
//#include <dlib/image_processing/frontal_face_detector.h>  
//#include <dlib/image_processing/render_face_detections.h>  
//#include <dlib/image_processing.h>  
//#include <dlib/gui_widgets.h>  
//#include"face.h"
//
//
//
//void show_num(cv::Mat& src, CvPoint po, double shownum)
//
//{
//	char res[20];
//
//	sprintf_s(res, "%.2f", shownum);
//
//	cv::putText(src, res, po, CV_FONT_HERSHEY_SIMPLEX,1.0, cv::Scalar(0, 255, 0),1);
//
//}
//
//void show_text(cv::Mat& src, CvPoint po, string res)
//
//{
//	cv::putText(src, res, po, CV_FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 1);
//
//}
//
//void Rect2rectangle(std::vector<cv::Rect>& cv_faces, std::vector<rectangle>& faces)
//{
//	for (int i = 0;i < cv_faces.size();i++)
//	{
//		rectangle r(cv_faces[i].x, cv_faces[i].y, cv_faces[i].width, cv_faces[i].height);
//		faces.push_back(r);
//	}
//
//}
//
//
//
//
//frontal_face_detector detector1 = get_frontal_face_detector();
//
//
//int main_t()
//{
//		
//	try
//	{
//		cv::VideoCapture cap(0);
//		if (!cap.isOpened())
//		{
//			cerr << "Unable to connect to camera" << endl;
//			return 1;
//		}  
//
//		//shape_predictor pose_model;
//		//deserialize("shape_predictor_5_face_landmarks.dat") >> pose_model;
//
//		//matrix<rgb_pixel> src;
//		init();
//		cv::CascadeClassifier Mycascade;
//		if (!Mycascade.load("lbpcascade_frontalface_improved.xml")) { cout << "load face_det model failed" << endl; };
//
//		// Grab and process frames until the main window is closed by the user.  
//		while (cv::waitKey(10) != 27)
//		{
//			// Grab a frame  
//			double t = (double)cvGetTickCount();
//
//			cv::Mat temp;
//			cap >> temp;
//
//			cv::Mat temp_gray(temp.size(), CV_8U);
//			cv::Mat temp_ehist;
//			cvtColor(temp, temp_gray, CV_BGR2GRAY);
//			cv::equalizeHist(temp_gray, temp_ehist);
//
//			//cv::imwrite("temp.jpg", temp);
//
//			matrix<rgb_pixel> dimg;
//			assign_image(dimg, cv_image<rgb_pixel>(temp));
//			cv_image<bgr_pixel> cimg(temp);
//			
//			// Detect faces   
//			std::vector<rectangle> faces = detector1(dimg);
//			// Find the pose of each face.  
//		//	std::vector<cv::Rect> cv_faces;
//		//	Mycascade.detectMultiScale(temp_ehist, cv_faces, 1.1, 2, 0, cv::Size(50, 50));
//			
//
//			if (faces.size()>0) {
//				cv::rectangle(temp, cv::Rect(faces[0].left(), faces[0].top(), faces[0].width(), faces[0].height()), cv::Scalar(0, 0, 255), 2, 8, 0);
//				int f = compare("feng.jpg", dimg, faces);
//				if (f == 1)
//				{
//					show_text(temp, cv::Point(faces[0].left(), faces[0].top()), "Feng");
//				}
//				else {
//					show_text(temp, cv::Point(faces[0].left(), faces[0].top()), "Unknow");
//				}
//			}
//
//			t = (double)cvGetTickCount() - t;
//			printf("检测所用时间 = %gms\n", t / ((double)cvGetTickFrequency()*1000.));//打印到屏幕
//
//			//Display it all on the screen  
//			imshow("Dlib特征点", temp);
//
//		}
//	}
//	catch (serialization_error& e)
//	{
//		cout << "You need dlib's default face landmarking model file to run this example." << endl;
//		cout << "You can get it from the following URL: " << endl;
//		cout << "   http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2" << endl;
//		cout << endl << e.what() << endl;
//	}
//	catch (exception& e)
//	{
//		cout << e.what() << endl;
//	}
//}
//
//
//int main(int argc, char** argv) try
//{
//
//	// We will also use a face landmarking model to align faces to a standard pose:  (see face_landmark_detection_ex.cpp for an introduction)
//	init();
//	matrix<rgb_pixel> img;
//	matrix<rgb_pixel> img2;
//	cv::Mat face_img = cv::imread("face1.jpg");
//	cv::Mat face_img2 = cv::imread("face2.jpg");
//	/*load_image(img, "face.jpg");
//	load_image(img2, "face2.jpg");*/
//	assign_image(img, cv_image<rgb_pixel>(face_img));
//	assign_image(img2, cv_image<rgb_pixel>(face_img2));
//
//
//	double t = (double)cvGetTickCount();
//	double tt = (double)cvGetTickCount();
//	std::vector<rectangle> face = detector1(img);
//	tt = (double)cvGetTickCount() - tt;
//	printf("人脸检测 = %gms\n", tt / ((double)cvGetTickFrequency()*1000.));
//
//	int f = compare("src.jpg", img, face);
//	t = (double)cvGetTickCount() - t;
//	printf("检测所用时间 = %gms\n", t / ((double)cvGetTickFrequency()*1000.));
//
//	cv::rectangle(face_img, cv::Rect(face[0].left(), face[0].top(), face[0].width(), face[0].height()), cv::Scalar(0, 0, 255), 2, 8, 0);
//	//cv::imwrite("dst.jpg",face_img);
//
//	std::vector<rectangle> face2 = detector1(img2);
//	cout << "resault1:" << f << endl;
//
//	double t1 = (double)cvGetTickCount();
//	int f1 = FaceID("src.jpg", "face1.jpg");
//	//int f1 = compare("feng.jpg", img2, face2);
//	t1 = (double)cvGetTickCount() - t1;
//	printf("检测所用时间 = %gms\n", t1 / ((double)cvGetTickFrequency()*1000.));
//
//
////	cv::rectangle(face_img2, cv::Rect(face2[0].left(), face2[0].top(), face2[0].width(), face2[0].height()), cv::Scalar(0, 0, 255), 2, 8, 0);
////	cv::imwrite("dst2.jpg", face_img2);
//
//	cout << "resault2:" << f1 << endl;
//
//	cin.get();
//}
//catch (std::exception& e)
//{
//	cout << e.what() << endl;
//}
