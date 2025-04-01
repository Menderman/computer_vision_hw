#include<opencv2/opencv.hpp>

cv::Mat ColorToGray(cv::Mat img) {
	cv::Mat result = cv::Mat::zeros(img.rows, img.cols, CV_8UC1);
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			cv::Vec3b bgr = img.at<cv::Vec3b>(i, j);
			//bgr.val[0]:B, bgr.val[1]:G, bgr.val[2]:R
			result.at<uchar>(i, j) = (bgr.val[0] + bgr.val[1] + bgr.val[2]) / 3;
		}
	}
	return result;
}

cv::Mat EdgeDetection(cv::Mat img, int kernel[3][3]) {
	cv::Mat result = cv::Mat::zeros(img.rows, img.cols, CV_8UC1);
	for (int i = 1; i < img.rows - 1; i++) {
		for (int j = 1; j < img.cols - 1; j++) {
			//convolution
			int value = kernel[0][0] * img.at<uchar>(i - 1, j - 1) + kernel[0][1] * img.at<uchar>(i - 1, j) + kernel[0][2] * img.at<uchar>(i - 1, j + 1) +
				kernel[1][0] * img.at<uchar>(i, j - 1) + kernel[1][1] * img.at<uchar>(i, j) + kernel[1][2] * img.at<uchar>(i, j + 1) +
				kernel[2][0] * img.at<uchar>(i + 1, j - 1) + kernel[2][1] * img.at<uchar>(i + 1, j) + kernel[2][2] * img.at<uchar>(i + 1, j + 1);
			if (value < 0) value = 0;//Relu
			result.at<uchar>(i, j) = value;
		}
	}
	return result;
}

cv::Mat MaxPooling(cv::Mat img) {
	cv::Mat result = cv::Mat::zeros(img.rows, img.cols, CV_8UC1);
	for (int i = 0; i < img.rows; i += 2) {
		for (int j = 0; j < img.cols; j += 2) {
			int max_val = img.at<uchar>(i, j);
			if (max_val < img.at<uchar>(i, j + 1)) max_val = img.at<uchar>(i, j + 1);
			if (max_val < img.at<uchar>(i + 1, j)) max_val = img.at<uchar>(i + 1, j);
			if (max_val < img.at<uchar>(i + 1, j + 1)) max_val = img.at<uchar>(i + 1, j + 1);
			result.at<uchar>(i, j) = max_val;
			result.at<uchar>(i, j + 1) = max_val;
			result.at<uchar>(i + 1, j) = max_val;
			result.at<uchar>(i + 1, j + 1) = max_val;
		}
	}
	return result;
}

cv::Mat Binarization(cv::Mat img) {
	cv::Mat result = cv::Mat::zeros(img.rows, img.cols, CV_8UC1);
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			if (img.at<uchar>(i, j) < 128) result.at<uchar>(i, j) = 0;
			else result.at<uchar>(i, j) = 255;
		}
	}
	return result;
}

int main() {
	cv::Mat img = cv::imread("car.png");
	cv::Mat img2 = cv::imread("liberty.png");
	//pad to even rows, cols
	if (img.rows % 2 == 1 && img.cols % 2 == 1) cv::copyMakeBorder(img, img, 0, 1, 0, 1, cv::BORDER_CONSTANT);
	else if (img.rows % 2 == 1 && img.cols % 2 == 0) cv::copyMakeBorder(img, img , 0, 1, 0, 0, cv::BORDER_CONSTANT);
	else if (img.rows % 2 == 0 && img.cols % 2 == 1) cv::copyMakeBorder(img, img, 0, 0, 0, 1, cv::BORDER_CONSTANT);
	if (img2.rows % 2 == 1 && img2.cols % 2 == 1) cv::copyMakeBorder(img2, img2, 0, 1, 0, 1, cv::BORDER_CONSTANT);
	else if (img2.rows % 2 == 1 && img2.cols % 2 == 0) cv::copyMakeBorder(img2, img2, 0, 1, 0, 0, cv::BORDER_CONSTANT);
	else if (img2.rows % 2 == 0 && img2.cols % 2 == 1) cv::copyMakeBorder(img2, img2, 0, 0, 0, 1, cv::BORDER_CONSTANT);

	cv::Mat result1 = cv::Mat::zeros(img.rows, img.cols, CV_8UC1);
	cv::Mat result2 = cv::Mat::zeros(img.rows, img.cols, CV_8UC1);
	cv::Mat result3 = cv::Mat::zeros(img.rows, img.cols, CV_8UC1);
	cv::Mat result4 = cv::Mat::zeros(img.rows, img.cols, CV_8UC1);

	int edge_detect_kernel[3][3] = { { -1, -1, -1}, {-1, 8, -1}, {-1, -1, -1} };
	//Q1
	result1 = ColorToGray(img);
	//Q2
	result2 = EdgeDetection(result1, edge_detect_kernel);
	//Q3
	//max pooling(downsampling), 2x2 filter, stride 2
	result3 = MaxPooling(result2);
	//Q4
	result4 = Binarization(result3);

	cv::imwrite("car_1.jpg", result1);
	cv::imwrite("car_2.jpg", result2);
	cv::imwrite("car_3.jpg", result3);
	cv::imwrite("car_4.jpg", result4);
	/*cv::namedWindow("result1", cv::WINDOW_NORMAL);
	imshow("result1", result1);
	cv::namedWindow("result2", cv::WINDOW_NORMAL);
	imshow("result2", result2);
	cv::namedWindow("result3", cv::WINDOW_NORMAL);
	imshow("result3", result3);
	cv::namedWindow("result4", cv::WINDOW_NORMAL);
	imshow("result4", result4);*/


	result1 = ColorToGray(img2);
	result2 = EdgeDetection(result1, edge_detect_kernel);
	result3 = MaxPooling(result2);
	result4 = Binarization(result3);

	cv::imwrite("liberty_1.jpg", result1);
	cv::imwrite("liberty_2.jpg", result2);
	cv::imwrite("liberty_3.jpg", result3);
	cv::imwrite("liberty_4.jpg", result4);
	/*cv::namedWindow("result1_2", cv::WINDOW_NORMAL);
	imshow("result1_2", result1);
	cv::namedWindow("result2_2", cv::WINDOW_NORMAL);
	imshow("result2_2", result2);
	cv::namedWindow("result3_2", cv::WINDOW_NORMAL);
	imshow("result3_2", result3);
	cv::namedWindow("result4_2", cv::WINDOW_NORMAL);
	imshow("result4_2", result4);*/


	cv::waitKey(0);
	cv::destroyAllWindows();
	return 0;
}