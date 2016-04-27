/////////libraries//////////
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/videoio/videoio.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <fstream>

//////using namespace/////////
using namespace cv;
using namespace std;
void OLBP(cv::Mat *psrc, cv::Mat *pdst);	//picked from a TUT, and then modified
char *get_string_id(int minutecount, int faceid, int secondcount, char path[]);
char *get_string_id2(int minutecount, int faceid, char path[]);
char *get_array_string(int minutecount, int faceid, char path[]);
Point2f get_variance(std::vector<Point2f> *ppoints, Point2f centroid);
Point2f get_centroid(std::vector<Point2f> *ppoints);

//get LBP image of source in destination
void OLBP(cv::Mat *psrc, cv::Mat *pdst) {
	Mat src = *psrc;
	Mat dst = *pdst;
	for (int i = 1; i<src.rows - 1; i++) {
		for (int j = 1; j<src.cols - 1; j++) {
			unsigned char center = src.at<char>(i, j);
			unsigned char code = 0;
			code |= (src.at<unsigned char>(i - 1, j - 1) > center) << 7;
			code |= (src.at<unsigned char>(i - 1, j) > center) << 6;
			code |= (src.at<unsigned char>(i - 1, j + 1) > center) << 5;
			code |= (src.at<unsigned char>(i, j + 1) > center) << 4;
			code |= (src.at<unsigned char>(i + 1, j + 1) > center) << 3;
			code |= (src.at<unsigned char>(i + 1, j) > center) << 2;
			code |= (src.at<unsigned char>(i + 1, j - 1) > center) << 1;
			code |= (src.at<unsigned char>(i, j - 1) > center) << 0;
			dst.at<unsigned char>(i - 1, j - 1) = code;
		}
	}
}

//These three are just for encoding/decoding the imagenames
char *get_string_id(int minutecount, int faceid, int secondcount, char path[]) {
	// filename: mmmmmmiisss . jpg
	int minutesize = 6;
	int faceidsize = 2;
	int secondsize = 3;
	int idsize = minutesize + faceidsize + secondsize + 4 + 1;		//4 is for extension .jpg and 1 is for '\0'
	int i = 0;
	int length = 0;

	bool isCreated = false;
	while (path[i] != '\0') {
		length++;
		i++;
	}

	char *out = (char *)malloc(sizeof(char) * (idsize + length));

	for (i = 0; i < length; i++) {
		out[i] = path[i];
	}
	for (i = 0; i < minutesize; i++) {
		out[minutesize - i - 1 + length] = 48 + (minutecount % 10);
		minutecount = minutecount / 10;
	}
	for (i = 0; i < faceidsize; i++) {
		out[minutesize + faceidsize - i - 1 + length] = 48 + (faceid % 10);
		faceid = faceid / 10;
	}
	for (i = 0; i < secondsize; i++) {
		out[minutesize + faceidsize+ secondsize - i - 1 + length] = 48 + (secondcount % 10);
		secondcount = secondcount / 10;
	}

	out[length + idsize - 1] = '\0';
	out[length + idsize - 2] = 'g';
	out[length + idsize - 3] = 'p';
	out[length + idsize - 4] = 'j';
	out[length + idsize - 5] = '.';

	return out;
}
char *get_string_id2(int minutecount, int faceid, char path[]) {

	int minutesize = 6;
	int faceidsize = 2;
	int idsize = minutesize + faceidsize + 4 + 1;		//4 is for extension .jpg and 1 is for '\0'
	int i = 0;
	int length = 0;
	bool isCreated = false;
	while (path[i] != '\0') {
		length++;
		i++;
	}

	char *out = (char *)malloc(sizeof(char) * (idsize + length));

	for (i = 0; i < length; i++) {
		out[i] = path[i];
	}
	for (i = 0; i < minutesize; i++) {
		out[minutesize - i - 1 + length] = 48 + (minutecount % 10);
		minutecount = minutecount / 10;
	}
	for (i = 0; i < faceidsize; i++) {
		out[minutesize + faceidsize - i - 1 + length] = 48 + (faceid % 10);
		faceid = faceid / 10;
	}

	out[length + idsize - 1] = '\0';
	out[length + idsize - 2] = 'g';
	out[length + idsize - 3] = 'p';
	out[length + idsize - 4] = 'j';
	out[length + idsize - 5] = '.';

	return out;
}
char *get_array_string(int minutecount, int faceid, char path[]) {
	//same as get_string_id, just '.jpg' replaced with '.txt'
	int minutesize = 6;
	int faceidsize = 2;
	int idsize = minutesize + faceidsize + 4 + 1;		//4 is for extension .jpg and 1 is for '\0'
	int i = 0;
	int length = 0;
	bool isCreated = false;
	while (path[i] != '\0') {
		length++;
		i++;
	}

	char *out = (char *)malloc(sizeof(char) * (idsize + length));

	for (i = 0; i < length; i++) {
		out[i] = path[i];
	}
	for (i = 0; i < minutesize; i++) {
		out[minutesize - i - 1 + length] = 48 + (minutecount % 10);
		minutecount = minutecount / 10;
	}
	for (i = 0; i < faceidsize; i++) {
		out[minutesize + faceidsize - i - 1 + length] = 48 + (faceid % 10);
		faceid = faceid / 10;
	}

	out[length + idsize - 1] = '\0';
	out[length + idsize - 2] = 't';
	out[length + idsize - 3] = 'x';
	out[length + idsize - 4] = 't';
	out[length + idsize - 5] = '.';

	return out;
}

//function to calculate the variance of a Point2f vector, having a Point2f centroid
Point2f get_variance(std::vector<Point2f> *ppoints, Point2f centroid) {
	int j = 0;
	Point2f variance;
	variance.x = 0;
	variance.y = 0;
	if ((*ppoints).size() == 0) {
		printf("Variance: No points found.\n");
		return (Point2f)NULL;
	}
	for (j = 0; j < (*ppoints).size(); j++) {
		variance.x = variance.x + ((*ppoints)[j].x - centroid.x)*((*ppoints)[j].x - centroid.x);
		variance.y = variance.y + ((*ppoints)[j].y - centroid.y)*((*ppoints)[j].y - centroid.y);
	}
	variance.x = variance.x / (*ppoints).size();
	variance.y = variance.y / (*ppoints).size();
	return variance;
}

//function to calculate the centroid of a Point2f vector
Point2f get_centroid(std::vector<Point2f> *ppoints) {
	int p = 0;
	Point2f centroid;
	centroid.x = 0;
	centroid.y = 0;

	if ((*ppoints).size() == 0) {
		printf("Centroid: No points found.\n");
		return (Point2f)NULL;
	}

	for (p = 0; p < (*ppoints).size(); p++) {
		centroid.x = centroid.x + (*ppoints)[p].x;
		centroid.y = centroid.y + (*ppoints)[p].y;
	}
	centroid.x = centroid.x / (*ppoints).size(); //Mean cordinates of points[i]
	centroid.y = centroid.y / (*ppoints).size(); // Mean cordinates of points[i]

	return centroid;
}