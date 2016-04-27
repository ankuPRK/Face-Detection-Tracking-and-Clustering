// The below codes are not optimized. It is straightforward for easy understanding.
// Original code's Copyright 2009 by Guoying Zhao & Matti Pietikainen
//but we have modified it to a very high extent, so if you want original, you should visit their webpage.

#include <math.h>
#include <string.h>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/videoio/videoio.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>

#define	BOUND(x, lowerbound, upperbound)  { (x) = (x) > (lowerbound) ? (x) : (lowerbound); \
                                            (x) = (x) < (upperbound) ? (x) : (upperbound); };
#define PI 3.1415926535897932
#define POW(nBit)   (1 << (nBit))
#define FREE(ptr) 	{if (NULL!=(ptr)) {delete[] ptr;  ptr=NULL;}}

using namespace cv;
using namespace std;

// uniform pattern for LBP without roation invariance for neighboring points 8 (only for 8)
// If the number of neighboring points is 4, 16 or other numbers than 8, you need to change this lookup array accordingly.
int  UniformPattern59[256]={    
	     1,   2,   3,   4,   5,   0,   6,   7,   8,   0,   0,   0,   9,   0,  10,  11,
		12,   0,   0,   0,   0,   0,   0,   0,  13,   0,   0,   0,  14,   0,  15,  16,
		17,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
		18,   0,   0,   0,   0,   0,   0,   0,  19,   0,   0,   0,  20,   0,  21,  22,
		23,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
		0,    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
		24,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
		25,   0,   0,   0,   0,   0,   0,   0,  26,   0,   0,   0,  27,   0,  28,  29,
		30,  31,   0,  32,   0,   0,   0,  33,   0,   0,   0,   0,   0,   0,   0,  34,
		0,    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  35,
		0,    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
		0,    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  36,
		37,  38,   0,  39,   0,   0,   0,  40,   0,   0,   0,   0,   0,   0,   0,  41,
		0,    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  42,
		43,  44,   0,  45,   0,   0,   0,  46,   0,   0,   0,   0,   0,   0,   0,  47,
		48,  49,   0,  50,   0,   0,   0,  51,  52,  53,   0,  54,  55,  56,  57,  58
};


void LBP_TOP(std::vector<cv::Mat> vec, int Length, int height, int width,
			 int FxRadius,int FyRadius,int Tinterval,
			 int XYNeighborPoints,int XTNeighborPoints,int YTNeighborPoints, 
			 int TimeLength,int BoderLength, int Bincount,
			 bool bBilinearinterpolation,float **Histogram);
void volume_LBP(std::vector<cv::Mat> imVec, int Length, int height, int width,
	int FxRadius, int FyRadius, int Tinterval,
	int XYNeighborPoints, int XTNeighborPoints, int YTNeighborPoints,
	int TimeLength, int BoderLength, int Bincount,
	bool bBilinearinterpolation, Mat hist);
void HOG_TOP(std::vector<cv::Mat> imVec, int Length, int height, int width, int Tinterval,
	int TimeLength, int BoderLength, int Bincount, Mat hist);
void volume_HOG(std::vector<cv::Mat> imVec, int Length, int height, int width, int Tinterval,
	int TimeLength, int BoderLength, int Bincount, Mat hist);

int RotLBP(int LBPCode,int NeighborPoints);
/* This function is to compute the LBP-TOP features for a video sequence
 Reference: 
 Guoying Zhao, Matti Pietikainen, "Dynamic texture recognition using local binary patterns 
 with an application to facial expressions," IEEE Transactions on Pattern Analysis and Machine 
 intelligence, 2007, 29(6):915-928.

  Copyright 2009 by Guoying Zhao & Matti Pietikainen
*/

// =====================================================================================================//
// Function: Running this funciton each time to compute the LBP-TOP distribution of one video sequence.
// Inputs:
// "imVec" keeps the grey level of all the pixels in sequences with [Length][height][width]; 
//      please note, all the images in one sequnces should have same size (height and weight). 
//      But they don't have to be same for different sequences.
// "Length" is the length of the sequence;
// "height" is the height of the images;
// "width" is the width of the images;
// "FxRadius", "FyRadius" and "Tinterval" are the radii parameter along X, Y and T axis; They can be 1, 2, 3 and 4. 
//"1" and "3" are recommended.
// Pay attention to "Tinterval". "Tinterval * 2 + 1" should be smaller than the length of the input sequence "Length". 
//For example, if one sequence includes seven frames, and you set Tinterval to three, 
//only the pixels in the frame 4 would be considered as central pixel and computed to get the LBP-TOP feature.
// "XYNeighborPoints", "XTNeighborPoints" and "YTNeighborPoints" are the number of the neighboring points
//     in XY plane, XT plane and YT plane; They can be 4, 8, 16 and 24. "8" is a good option.
// "TimeLength" and "BoderLength" are the parameters for bodering parts in time and space which would not 
//     be computed for features. Usually they are same to Tinterval and the bigger one of "FxRadius" and "FyRadius";
// "bBilinearinterpolation": if use bilinear interpolation for computing a neighboring point in a circle: 1 (yes), 0 (no).
// "Bincount": For example, if XYNeighborPoints = XTNeighborPoints = YTNeighborPoints = 8, you can set "Bincount" as "0" 
// if you want to use basic LBP, or set "Bincount" as 59 if using uniform pattern of LBP, 
// If the number of Neighboring points is different than 8, you need to change it accordingly as well as 
//change the above "UniformPattern59".
// Output: 
// "Histogram": keeps LBP-TOP distribution of all the pixels in the current frame with [3][dim]; 
//     here, "3" deote the three planes of LBP-TOP, i.e., XY, XZ and YZ planes.
//     Each value of Histogram[i][j] is between [0,1]
// =====================================================================================================//

void volume_LBP(std::vector<cv::Mat> imVec, int Length, int height, int width,
			 int FxRadius,int FyRadius,int Tinterval,
			 int XYNeighborPoints,int XTNeighborPoints,int YTNeighborPoints, 
			 int TimeLength,int BoderLength, int Bincount,
			 bool bBilinearinterpolation,Mat hist)

{
	int i,j;
	int xc,yc;			char Centerchar;	char Currentchar;
	int BasicLBP = 0;	int FeaBin = 0;		int p;
	int X,Y,Z;			float x1,y1,z1;		float u,v;	
	int ltx,lty;		int lbx,lby; 		int	rtx,rty;		int rbx,rby;
	float **Histogram = (float **)malloc(sizeof(float *)*3);

	if (Bincount == 0){
		for (i = 0; i < 3; i++){
			Histogram[i] = (float *)malloc(sizeof(float)*256);
		}
	}
	else{
		for (i = 0; i < 3; i++){
			Histogram[i] = (float *)malloc(sizeof(float)*Bincount);
		}
	}


	if (bBilinearinterpolation == false)
	{
		for(i = TimeLength; i < Length - TimeLength; i++)
		{
			for(yc = BoderLength; yc < height - BoderLength; yc++)
			{
				for(xc = BoderLength; xc < width - BoderLength; xc++)
				{
					Centerchar = imVec[i].at<uchar>(yc,xc);
					BasicLBP = 0;	FeaBin = 0;
					
					// In XY plane
					for(p = 0; p < XYNeighborPoints; p++)
					{
						X = int (xc + FxRadius * cos((2 * PI * p) / XYNeighborPoints) + 0.5);
						Y = int (yc - FyRadius * sin((2 * PI * p) / XYNeighborPoints) + 0.5);
						BOUND(X,0,width-1); BOUND(Y,0,height-1);
						Currentchar = imVec[i].at<uchar>(Y, X);
						if(Currentchar >= Centerchar) BasicLBP += POW ( FeaBin); 
						FeaBin++;
					}

					//     if "Bincount" is "0", it means basic LBP-TOP will be computed 
					//         and "UniformPattern59" does not work in this case. 
					//     Otherwise it should be the number of the uniform patterns, 
					//         then "UniformPattern59" keeps the lookup-table of the 
					//         basic LBP and unifrom LBP.
					if(Bincount == 0)
						Histogram[0][BasicLBP]++;
					else // uniform patterns
						Histogram[0][UniformPattern59[BasicLBP]]++;
					

					// In XT-plane
					BasicLBP = 0;	FeaBin = 0;
					for(p = 0; p < XTNeighborPoints; p++)
					{
						X = int (xc + FxRadius * cos((2 * PI * p) / XTNeighborPoints) + 0.5);
						Z = int (i + Tinterval * sin((2 * PI * p) / XTNeighborPoints) + 0.5);
						BOUND(X,0,width-1); BOUND(Z,0,Length-1);
						
						//Currentchar = fg[Z][yc][X];
						Currentchar = imVec[Z].at<uchar>(yc,X);

						if(Currentchar >= Centerchar) BasicLBP += POW ( FeaBin);
						FeaBin++;
					}

                	if(Bincount == 0)
						Histogram[1][BasicLBP]++;
					else
						Histogram[1][UniformPattern59[BasicLBP]]++;
					
					// In YT-plane
					BasicLBP = 0;	FeaBin = 0;
					for(p = 0; p < YTNeighborPoints; p++)
					{
						Y = int (yc - FyRadius * sin((2 * PI * p) / YTNeighborPoints) + 0.5);
						Z = int (i + Tinterval * cos((2 * PI * p) / YTNeighborPoints) + 0.5);
						BOUND(Y,0,height-1); BOUND(Z,0,Length-1);

						Currentchar = imVec[Z].at<uchar>(Y, xc);	//fg[Z][Y][xc]

						if(Currentchar >= Centerchar) BasicLBP += POW ( FeaBin);
						FeaBin++;
					}

					if(Bincount == 0)
						Histogram[2][BasicLBP]++;
					else
						Histogram[2][UniformPattern59[BasicLBP]]++;
								
				}//for(xc = BoderLength; xc < width - BoderLength; xc++)
			}//for(yc = BoderLength; yc < height - BoderLength; yc++)
				
		}//for(i = TimeLength; i < Length - TimeLength; i++)

	} // if (true == bUniformPattern) 
		
	else 
	{
		for(i = TimeLength; i < Length - TimeLength; i++)
		{
			for(yc = BoderLength; yc < height - BoderLength; yc++)
			{
				for(xc = BoderLength; xc < width - BoderLength; xc++)
				{
					Centerchar = imVec[i].at<uchar>(yc, xc);
					BasicLBP = 0;	FeaBin = 0;
					
					// In XY plane
					for(p = 0; p < XYNeighborPoints; p++)
					{
						//		bilinear interpolation
						x1 = float(xc + FxRadius * cos((2 * PI * p) / XYNeighborPoints));
						y1 = float(yc - FyRadius * sin((2 * PI * p) / XYNeighborPoints));
						

						u = x1 - int(x1);
						v = y1 - int(y1);
						ltx = int(floor(x1)); lty = int(floor(y1));
						lbx = int(floor(x1)); lby = int(ceil(y1));
						rtx = int(ceil(x1)); rty = int(floor(y1));
						rbx = int(ceil(x1)); rby = int(ceil(y1));
						// values of neighbors that do not fall exactly on pixels are estimated
						// by bilinear interpolation of four corner points near to it.
							

						Currentchar = (char)(imVec[i].at<uchar>(lty, ltx) *(1 - u) * (1 - v) + \
												imVec[i].at<uchar>(lby, lbx) * (1 - u) * v + \
												imVec[i].at<uchar>(rty, rtx) * u * (1 - v) + \
												imVec[i].at<uchar>(rby, rbx) * u * v);
						if(Currentchar >= Centerchar) BasicLBP += POW ( FeaBin); 
						FeaBin++;
					
					}


					//     if "Bincount" is "0", it means basic LBP-TOP will be computed 
					//         and "UniformPattern59" does not work in this case. 
					//     Otherwise it should be the number of the uniform patterns, 
					//         then "UniformPattern59" keeps the lookup-table of the 
					//         basic LBP and unifrom LBP.
					if(Bincount == 0)
						Histogram[0][BasicLBP]++;
					else // uniform patterns
						Histogram[0][UniformPattern59[BasicLBP]]++;
					

					// In XT-plane
					BasicLBP = 0;
					FeaBin = 0;
					for(p = 0; p < XTNeighborPoints; p++)
					{
					//	bilinear interpolation
						x1 =float(xc + FxRadius * cos((2 * PI * p) / XTNeighborPoints)) ;
						z1 =float(i + Tinterval * sin((2 * PI * p) / XTNeighborPoints)) ;

						u = x1 - int(x1);
						v = z1 - int(z1);
						ltx = int(floor(x1)); lty = int(floor(z1));
						lbx = int(floor(x1)); lby = int(ceil(z1));
						rtx = int(ceil(x1)); rty = int(floor(z1));
						rbx = int(ceil(x1)); rby = int(ceil(z1));
						// values of neighbors that do not fall exactly on pixels are estimated
						// by bilinear interpolation of four corner points near to it.

						Currentchar = (char) (imVec[lty].at<uchar>(yc, ltx) * (1 - u) * (1-v) + \
											  imVec[lby].at<uchar>(yc, lbx) * (1 - u) * v + \
											  imVec[rty].at<uchar>(yc, rtx)* u * (1-v) + \
											  imVec[rby].at<uchar>(yc, rbx) * u * v);
						if(Currentchar >= Centerchar) BasicLBP += POW ( FeaBin);
						FeaBin++;
					}

                	if(Bincount == 0)
						Histogram[1][BasicLBP]++;
					else
						Histogram[1][UniformPattern59[BasicLBP]]++;
					
					// In YT-plane
					BasicLBP = 0;
					FeaBin = 0;

					
					for(p = 0; p < YTNeighborPoints; p++)
					{
					//	bilinear interpolation
						y1 = float (yc - FyRadius * sin((2 * PI * p) / YTNeighborPoints));
						z1 = float (i + Tinterval * cos((2 * PI * p) / YTNeighborPoints));
						
						u = y1 - int(y1);
						v = z1 - int(z1);
						
						ltx = int(floor(y1)); lty = int(floor(z1));
						lbx = int(floor(y1)); lby = int(ceil(z1));
						rtx = int(ceil(y1)); rty = int(floor(z1));
						rbx = int(ceil(y1)); rby = int(ceil(z1));
						// values of neighbors that do not fall exactly on pixels are estimated 
						// by bilinear interpolation of four corner points near to it.
						Currentchar = (char)(imVec[lty].at<uchar>(ltx, xc) * (1 - u) * (1 - v) + \
											 imVec[lby].at<uchar>(lbx, xc) * (1 - u) * v + \
											 imVec[rty].at<uchar>(rtx, xc) * u * (1 - v) + \
											 imVec[rby].at<uchar>(rbx, xc) * u * v);
						if(Currentchar >= Centerchar) BasicLBP += POW ( FeaBin);
						FeaBin++;
					}

					if(Bincount == 0)
						Histogram[2][BasicLBP]++;
					else
						Histogram[2][UniformPattern59[BasicLBP]]++;
								
				}//for(xc = BoderLength; xc < width - BoderLength; xc++)
			}//for(yc = BoderLength; yc < height - BoderLength; yc++)
		}//for(i = TimeLength; i < Length - TimeLength; i++)
	} // if (true == bUniformPattern) 

//-------------  Normalization ----------------------------//
	int binCount[3];
	if(Bincount == 0)
	{
		binCount[0]= POW(XYNeighborPoints); 
		binCount[1]= POW(XTNeighborPoints); 
		binCount[2]= POW(YTNeighborPoints); 
	}
	else 
	{
		// for case that XYNeighborPoints = XTNeighborPoints = XTNeighborPoints.
		// If they are not same, there should be three "Bincount".
		binCount[0] = Bincount;		binCount[1] = Bincount;		binCount[2] = Bincount;
	}

	// Normaliztion
	int Total = 0;
	for(j = 0; j < 3; j++)
	{
		Total = 0;
		for(i = 0; i < binCount[j]; i++)  
			Total += int (Histogram[j][i]);
		for(i = 0; i < binCount[j]; i++)
		{
			Histogram[j][i] /= (Total*1.0);
			hist.at<float>(j, i) = MAX(Histogram[j][i],0);
		}
	}
//-------------  Normalization ----------------------------//
}

void LBP_TOP(std::vector<cv::Mat> imVec, int Length, int height, int width,
	int FxRadius, int FyRadius, int Tinterval,
	int XYNeighborPoints, int XTNeighborPoints, int YTNeighborPoints,
	int TimeLength, int BoderLength, int Bincount,
	bool bBilinearinterpolation, Mat hist)

{
	int i, j;
	int xc, yc;			char Centerchar;	char Currentchar;
	int BasicLBP = 0;	int FeaBin = 0;		int p;
	int X, Y, Z;			float x1, y1, z1;		float u, v;
	int ltx, lty;		int lbx, lby; 		int	rtx, rty;		int rbx, rby;
	float **Histogram = (float **)malloc(sizeof(float *) * 3);

	if (Bincount == 0){
		for (i = 0; i < 3; i++){
			Histogram[i] = (float *)malloc(sizeof(float) * 256);
		}
	}
	else{
		for (i = 0; i < 3; i++){
			Histogram[i] = (float *)malloc(sizeof(float)*Bincount);
		}
	}
	if (bBilinearinterpolation == false)
	{
		for (yc = BoderLength; yc < height - BoderLength; yc++)
		{
			for (xc = BoderLength; xc < width - BoderLength; xc++)
			{
				//					chuco++;
				Centerchar = imVec[imVec.size() / 2].at<uchar>(yc, xc);
				BasicLBP = 0;	FeaBin = 0;
				// In XY plane
				for (p = 0; p < XYNeighborPoints; p++)
				{
					X = int(xc + FxRadius * cos((2 * PI * p) / XYNeighborPoints) + 0.5);
					Y = int(yc - FyRadius * sin((2 * PI * p) / XYNeighborPoints) + 0.5);
					BOUND(X, 0, width - 1); BOUND(Y, 0, height - 1);
					Currentchar = imVec[imVec.size() / 2].at<uchar>(Y, X);
					if (Currentchar >= Centerchar) BasicLBP += POW(FeaBin);
					FeaBin++;
				}
				//     if "Bincount" is "0", it means basic LBP-TOP will be computed 
				//         and "UniformPattern59" does not work in this case. 
				//     Otherwise it should be the number of the uniform patterns, 
				//         then "UniformPattern59" keeps the lookup-table of the 
				//         basic LBP and unifrom LBP.
				if (Bincount == 0)
					Histogram[0][BasicLBP]++;
				else // uniform patterns
					Histogram[0][UniformPattern59[BasicLBP]]++;
			}
		}

				for (i = TimeLength; i < Length - TimeLength; i++)
				{
					for (xc = BoderLength; xc < width - BoderLength; xc++)
					{
//						chuco++;

						// In XT-plane
						Centerchar = imVec[i].at<uchar>(height/2, xc);
						BasicLBP = 0;	FeaBin = 0;
						for (p = 0; p < XTNeighborPoints; p++)
						{
							X = int(xc + FxRadius * cos((2 * PI * p) / XTNeighborPoints) + 0.5);
							Z = int(i + Tinterval * sin((2 * PI * p) / XTNeighborPoints) + 0.5);
							BOUND(X, 0, width - 1); BOUND(Z, 0, Length - 1);

							//Currentchar = fg[Z][yc][X];
							Currentchar = imVec[Z].at<uchar>(height/2, X);
							if (Currentchar >= Centerchar) BasicLBP += POW(FeaBin);
							FeaBin++;
						}

						if (Bincount == 0)
							Histogram[1][BasicLBP]++;
						else
							Histogram[1][UniformPattern59[BasicLBP]]++;
					}
				}

				for (i = TimeLength; i < Length - TimeLength; i++)
				{
					for (yc = BoderLength; yc < height - BoderLength; yc++)
					{
					// In YT-plane
//						chuco++;

						Centerchar = imVec[i].at<uchar>(yc, width/2);
						BasicLBP = 0;	FeaBin = 0;
					for (p = 0; p < YTNeighborPoints; p++)
					{
						Y = int(yc - FyRadius * sin((2 * PI * p) / YTNeighborPoints) + 0.5);
						Z = int(i + Tinterval * cos((2 * PI * p) / YTNeighborPoints) + 0.5);
						BOUND(Y, 0, height - 1); BOUND(Z, 0, Length - 1);

						Currentchar = imVec[Z].at<uchar>(Y, width/2);	//fg[Z][Y][xc]

						if (Currentchar >= Centerchar) BasicLBP += POW(FeaBin);
						FeaBin++;
					}

					if (Bincount == 0)
						Histogram[2][BasicLBP]++;
					else
						Histogram[2][UniformPattern59[BasicLBP]]++;

				}//for(xc = BoderLength; xc < width - BoderLength; xc++)
			}//for(yc = BoderLength; yc < height - BoderLength; yc++)

	} // if (true == bUniformPattern) 


	//-------------  Normalization ----------------------------//
	int binCount[3];
	if (Bincount == 0)
	{
		binCount[0] = POW(XYNeighborPoints);
		binCount[1] = POW(XTNeighborPoints);
		binCount[2] = POW(YTNeighborPoints);
	}
	else
	{
		// for case that XYNeighborPoints = XTNeighborPoints = XTNeighborPoints.
		// If they are not same, there should be three "Bincount".
		binCount[0] = Bincount;		binCount[1] = Bincount;		binCount[2] = Bincount;
	}

	// Normaliztion
	int Total = 0;
	for (j = 0; j < 3; j++)
	{
		Total = 0;
		for (i = 0; i < binCount[j]; i++)
			Total += int(Histogram[j][i]);
		for (i = 0; i < binCount[j]; i++)
		{
			Histogram[j][i] /= (Total*1.0);
			hist.at<float>(j, i) = MAX(Histogram[j][i], 0);
		}
	}
	//-------------  Normalization ----------------------------//
//	cout << chuco << endl;
}


void HOG_TOP(std::vector<cv::Mat> imVec, int Length, int height, int width, int Tinterval,
	int TimeLength, int BoderLength, int Bincount,Mat hist)

{
	int i, j;
	int xc, yc;			char Centerchar;	char Currentchar;
	int BasicLBP = 0;	int FeaBin = 0;		int p;
	int X, Y, Z;			float x1, y1, z1;		float u, v;
	int ltx, lty;		int lbx, lby; 		int	rtx, rty;		int rbx, rby;
	float fx, fy, fz, mag, angle;


	float **Histogram = (float **)malloc(sizeof(float *) * 3);

	if (Bincount == 8){
		for (i = 0; i < 3; i++){
			Histogram[i] = (float *)malloc(sizeof(float) * 8);
		}
	}
	else{
		cout << "BinCount other than 8 is currently NOT supported." << endl;
	}

	for (yc = BoderLength; yc < height - BoderLength; yc++)
	{
		for (xc = BoderLength; xc < width - BoderLength; xc++)
		{
			// In XY plane
			fx = (imVec[Length/2].at<uchar>(yc + 1, xc + 1) + imVec[Length/2].at<uchar>(yc, xc + 1) + imVec[Length/2].at<uchar>(yc - 1, xc + 1)) -
				(imVec[Length/2].at<uchar>(yc + 1, xc - 1) + imVec[Length/2].at<uchar>(yc, xc - 1) + imVec[Length/2].at<uchar>(yc - 1, xc - 1));
			fy = (imVec[Length/2].at<uchar>(yc + 1, xc + 1) + imVec[Length/2].at<uchar>(yc + 1, xc) + imVec[Length/2].at<uchar>(yc + 1, xc - 1)) -
				(imVec[Length/2].at<uchar>(yc - 1, xc + 1) + imVec[Length/2].at<uchar>(yc - 1, xc) + imVec[Length/2].at<uchar>(yc - 1, xc - 1));
			mag = sqrt(fx*fx + fy*fy);
			if (fx == 0 && fy == 0){
				//do nothing
			}
			else if (fx == 0){
				//both 1st and last
				if (fy>0)
				Histogram[0][2] += mag / 2.0;
				else
				Histogram[0][6] += mag / 2.0;
			}
			else if (fy == 0){
				//both 1st and last
				if (fx>0)
				Histogram[0][0] += mag / 2.0;
				else
				Histogram[0][4] += mag / 2.0;
			}
			else{
				if (fx > 0 && fy > 0){
					angle = atan(fy / fx) * 180 / PI;
				}
				else if (fx < 0 && fy>0){
					angle = 180 + atan(fy / fx) * 180 / PI;
				}
				else if (fx < 0 && fy < 0){
					angle = 180 + atan(fy / fx) * 180 / PI;
				}
				else{
					angle = 360 + atan(fy / fx) * 180 / PI;
				}
				for (int qq = 0; qq < 8; qq++){
					int rr = (qq + 1) % 8;
					if (angle > 45 * qq && angle <= 45 * (qq + 1)){
						Histogram[0][qq] += mag*(45 * (qq + 1) - angle) / 45;
						Histogram[0][rr] += mag*(angle - 45 * qq) / 45;
						break;
					}
				}
			}

		}
	}

	for (i = TimeLength; i < Length - TimeLength; i++)
	{
		for (xc = BoderLength; xc < width - BoderLength; xc++)
		{

			// In XZ plane
			fx = (imVec[i + 1].at<uchar>(height / 2, xc + 1) + imVec[i].at<uchar>(height / 2, xc + 1) + imVec[i - 1].at<uchar>(height / 2, xc + 1)) -
				(imVec[i + 1].at<uchar>(height / 2, xc - 1) + imVec[i].at<uchar>(height / 2, xc - 1) + imVec[i - 1].at<uchar>(height / 2, xc - 1));
			fz = (imVec[i + 1].at<uchar>(height / 2, xc + 1) + imVec[i + 1].at<uchar>(height / 2, xc) + imVec[i + 1].at<uchar>(height / 2, xc - 1)) -
				(imVec[i - 1].at<uchar>(height / 2, xc + 1) + imVec[i - 1].at<uchar>(height / 2, xc) + imVec[i - 1].at<uchar>(height / 2, xc - 1));
			mag = sqrt(fx*fx + fz*fz);
			if (fx == 0 && fz == 0){
				//do nothing
			}
			else if (fx == 0){
				//both 1st and last
				if (fz>0)
					Histogram[0][2] += mag / 2.0;
				else
					Histogram[0][6] += mag / 2.0;
			}
			else if (fz == 0){
				//both 1st and last
				if (fx>0)
					Histogram[0][0] += mag / 2.0;
				else
					Histogram[0][4] += mag / 2.0;
			}
			else{
				if (fx > 0 && fz > 0){
					angle = atan(fz / fx) * 180 / PI;
				}
				else if (fx < 0 && fz>0){
					angle = 180 + atan(fz / fx) * 180 / PI;
				}
				else if (fx < 0 && fz < 0){
					angle = 180 + atan(fz / fx) * 180 / PI;
				}
				else{
					angle = 360 + atan(fz / fx) * 180 / PI;
				}
				for (int qq = 0; qq < 8; qq++){
					int rr = (qq + 1) % 8;
					if (angle > 45 * qq && angle <= 45 * (qq + 1)){
						Histogram[1][qq] += mag*(45 * (qq + 1) - angle) / 45;
						Histogram[1][rr] += mag*(angle - 45 * qq) / 45;
					}
				}
			}
		}
	}

	for (i = TimeLength; i < Length - TimeLength; i++)
	{
		for (yc = BoderLength; yc < height - BoderLength; yc++)
		{

					// In YZ plane
					fz = (imVec[i+1].at<uchar>(yc + 1, width/2 ) + imVec[i+1].at<uchar>(yc, width/2 ) + imVec[i+1].at<uchar>(yc - 1, width/2 )) -
						(imVec[i-1].at<uchar>(yc + 1, width/2 ) + imVec[i-1].at<uchar>(yc, width/2 ) + imVec[i-1].at<uchar>(yc - 1, width/2 ));
					fy = (imVec[i+1].at<uchar>(yc + 1, width/2 ) + imVec[i].at<uchar>(yc + 1, width/2 ) + imVec[i-1].at<uchar>(yc + 1, width/2 )) -
						(imVec[i+1].at<uchar>(yc - 1, width/2 ) + imVec[i+1].at<uchar>(yc - 1, width/2 ) + imVec[i-1].at<uchar>(yc - 1, width/2 ));
					mag = sqrt(fz*fz + fy*fy);
					if (fz == 0 && fy == 0){
						//do nothing
					}
					else if (fz == 0){
						//both 1st and last
						if (fy>0)
							Histogram[0][2] += mag / 2.0;
						else
							Histogram[0][6] += mag / 2.0;
					}
					else if (fy == 0){
						//both 1st and last
						if (fz>0)
							Histogram[0][0] += mag / 2.0;
						else
							Histogram[0][4] += mag / 2.0;
					}
					else{
						if (fz > 0 && fy > 0){
							angle = atan(fy / fz) * 180 / PI;
						}
						else if (fz < 0 && fy>0){
							angle = 180 + atan(fy / fz) * 180 / PI;
						}
						else if (fz < 0 && fy<0){
							angle = 180 + atan(fy / fz) * 180 / PI;
						}
						else{
							angle = 360 + atan(fy / fz) * 180 / PI;
						}
						for (int qq = 0; qq < 8; qq++){
							int rr = (qq + 1) % 8;
							if (angle > 45 * qq && angle <= 45 * (qq + 1)){
								Histogram[2][qq] += mag*(45 * (qq + 1) - angle) / 45;
								Histogram[2][rr] += mag*(angle - 45 * qq) / 45;
							}
						}
					}

				}//for(xc = BoderLength; xc < width - BoderLength; xc++)
			}//for(yc = BoderLength; yc < height - BoderLength; yc++)

	//-------------  Normalization ----------------------------//
	int binCount[3];
	int laxes[3] = { width - BoderLength * 2, height - BoderLength * 2, Length - TimeLength * 2 };
	int Total[3] = { laxes[0] * laxes[1], laxes[0] * laxes[2], laxes[1]*laxes[2] };
	// Normaliztion
	for (j = 0; j < 3; j++)
	{
		for (i = 0; i < Bincount; i++)
		{
			Histogram[j][i] /= (Total[j]*1.0);
			hist.at<float>(j, i) = MAX(Histogram[j][i], 0);
		}
	}
	//-------------  Normalization ----------------------------//

}

void volume_HOG(std::vector<cv::Mat> imVec, int Length, int height, int width, int Tinterval,
	int TimeLength, int BoderLength, int Bincount, Mat hist)

{
	int i, j;
	int xc, yc;			char Centerchar;	char Currentchar;
	int BasicLBP = 0;	int FeaBin = 0;		int p;
	int X, Y, Z;			float x1, y1, z1;		float u, v;
	int ltx, lty;		int lbx, lby; 		int	rtx, rty;		int rbx, rby;
	float fx, fy, fz, mag, angle;


	float **Histogram = (float **)malloc(sizeof(float *) * 3);

	if (Bincount == 8){
		for (i = 0; i < 3; i++){
			Histogram[i] = (float *)malloc(sizeof(float) * 8);
		}
	}
	else{
		cout << "BinCount other than 8 is currently NOT supported." << endl;
	}

	for (i = TimeLength; i < Length - TimeLength; i++)
	{
		for (yc = BoderLength; yc < height - BoderLength; yc++)
		{
			for (xc = BoderLength; xc < width - BoderLength; xc++)
			{
				// In XY plane
				fx = (imVec[i].at<uchar>(yc + 1, xc + 1) + imVec[i].at<uchar>(yc, xc + 1) + imVec[i].at<uchar>(yc - 1, xc + 1)) -
					(imVec[i].at<uchar>(yc + 1, xc - 1) + imVec[i].at<uchar>(yc, xc - 1) + imVec[i].at<uchar>(yc - 1, xc - 1));
				fy = (imVec[i].at<uchar>(yc + 1, xc + 1) + imVec[i].at<uchar>(yc + 1, xc) + imVec[i].at<uchar>(yc + 1, xc - 1)) -
					(imVec[i].at<uchar>(yc - 1, xc + 1) + imVec[i].at<uchar>(yc - 1, xc) + imVec[i].at<uchar>(yc - 1, xc - 1));
				mag = sqrt(fx*fx + fy*fy);
				if (fx == 0 && fy == 0){
					//do nothing
				}
				else if (fx == 0){
					//both 1st and last
					if (fy>0)
						Histogram[0][2] += mag / 2.0;
					else
						Histogram[0][6] += mag / 2.0;
				}
				else if (fy == 0){
					//both 1st and last
					if (fx>0)
						Histogram[0][0] += mag / 2.0;
					else
						Histogram[0][4] += mag / 2.0;
				}
				else{
					if (fx > 0 && fy > 0){
						angle = atan(fy / fx) * 180 / PI;
					}
					else if (fx < 0 && fy>0){
						angle = 180 + atan(fy / fx) * 180 / PI;
					}
					else if (fx < 0 && fy<0){
						angle = 180 + atan(fy / fx) * 180 / PI;
					}
					else{
						angle = 360 + atan(fy / fx) * 180 / PI;
					}
					for (int qq = 0; qq < 8; qq++){
						int rr = (qq + 1) % 8;
						if (angle > 45 * qq && angle <= 45 * (qq + 1)){
							Histogram[0][qq] += mag*(45 * (qq + 1) - angle) / 45;
							Histogram[0][rr] += mag*(angle - 45 * qq) / 45;
						}
					}
				}

				// In XZ plane
				fx = (imVec[i + 1].at<uchar>(yc, xc + 1) + imVec[i].at<uchar>(yc, xc + 1) + imVec[i - 1].at<uchar>(yc, xc + 1)) -
					(imVec[i + 1].at<uchar>(yc, xc - 1) + imVec[i].at<uchar>(yc, xc - 1) + imVec[i - 1].at<uchar>(yc, xc - 1));
				fz = (imVec[i + 1].at<uchar>(yc, xc + 1) + imVec[i + 1].at<uchar>(yc, xc) + imVec[i + 1].at<uchar>(yc, xc - 1)) -
					(imVec[i - 1].at<uchar>(yc, xc + 1) + imVec[i - 1].at<uchar>(yc, xc) + imVec[i - 1].at<uchar>(yc, xc - 1));
				mag = sqrt(fx*fx + fz*fz);
				if (fx == 0 && fz == 0){
					//do nothing
				}
				else if (fx == 0){
					//both 1st and last
					if (fz>0)
						Histogram[0][2] += mag / 2.0;
					else
						Histogram[0][6] += mag / 2.0;
				}
				else if (fz == 0){
					//both 1st and last
					if (fx>0)
						Histogram[0][0] += mag / 2.0;
					else
						Histogram[0][4] += mag / 2.0;
				}
				else{
					if (fx > 0 && fz > 0){
						angle = atan(fz / fx) * 180 / PI;
					}
					else if (fx < 0 && fz>0){
						angle = 180 + atan(fz / fx) * 180 / PI;
					}
					else if (fx < 0 && fz<0){
						angle = 180 + atan(fz / fx) * 180 / PI;
					}
					else{
						angle = 360 + atan(fz / fx) * 180 / PI;
					}
					for (int qq = 0; qq < 8; qq++){
						int rr = (qq + 1) % 8;
						if (angle > 45 * qq && angle <= 45 * (qq + 1)){
							Histogram[1][qq] += mag*(45 * (qq + 1) - angle) / 45;
							Histogram[1][rr] += mag*(angle - 45 * qq) / 45;
						}
					}
				}
				// In YZ plane
				fz = (imVec[i + 1].at<uchar>(yc + 1, xc) + imVec[i + 1].at<uchar>(yc, xc) + imVec[i + 1].at<uchar>(yc - 1, xc)) -
					(imVec[i - 1].at<uchar>(yc + 1, xc) + imVec[i - 1].at<uchar>(yc, xc) + imVec[i - 1].at<uchar>(yc - 1, xc));
				fy = (imVec[i + 1].at<uchar>(yc + 1, xc) + imVec[i].at<uchar>(yc + 1, xc) + imVec[i - 1].at<uchar>(yc + 1, xc)) -
					(imVec[i + 1].at<uchar>(yc - 1, xc) + imVec[i + 1].at<uchar>(yc - 1, xc) + imVec[i - 1].at<uchar>(yc - 1, xc));
				mag = sqrt(fz*fz + fy*fy);
				if (fz == 0 && fy == 0){
					//do nothing
				}
				else if (fz == 0){
					//both 1st and last
					if (fy>0)
						Histogram[0][2] += mag / 2.0;
					else
						Histogram[0][6] += mag / 2.0;
				}
				else if (fy == 0){
					//both 1st and last
					if (fz>0)
						Histogram[0][0] += mag / 2.0;
					else
						Histogram[0][4] += mag / 2.0;
				}
				else{
					if (fz > 0 && fy > 0){
						angle = atan(fy / fz) * 180 / PI;
					}
					else if (fz < 0 && fy>0){
						angle = 180 + atan(fy / fz) * 180 / PI;
					}
					else if (fz < 0 && fy<0){
						angle = 180 + atan(fy / fz) * 180 / PI;
					}
					else{
						angle = 360 + atan(fy / fz) * 180 / PI;
					}
					for (int qq = 0; qq < 8; qq++){
						int rr = (qq + 1) % 8;
						if (angle > 45 * qq && angle <= 45 * (qq + 1)){
							Histogram[2][qq] += mag*(45 * (qq + 1) - angle) / 45;
							Histogram[2][rr] += mag*(angle - 45 * qq) / 45;
						}
					}
				}

			}//for(xc = BoderLength; xc < width - BoderLength; xc++)
		}//for(yc = BoderLength; yc < height - BoderLength; yc++)

	}//for(i = TimeLength; i < Length - TimeLength; i++)

	//-------------  Normalization ----------------------------//
	int binCount[3];
	int laxes[3] = { width - BoderLength * 2, height - BoderLength * 2, Length - TimeLength * 2 };
	int Total = laxes[0] * laxes[1] * laxes[2];
	// Normaliztion
	for (j = 0; j < 3; j++)
	{
		for (i = 0; i < Bincount; i++)
		{
			Histogram[j][i] /= (Total * 1.0);
			hist.at<float>(j, i) = MAX(Histogram[j][i], 0);
		}
	}
	//-------------  Normalization ----------------------------//

}


// For a basic LBP code, this function is to get its rotation invariance corresponding code.
int RotLBP(int LBPCode,int NeighborPoints)
{
	int minLBP = LBPCode,	tempCode;
	for(int i = 1; i < NeighborPoints; i++)
   	{
		tempCode = (LBPCode>>i) | (((LBPCode & (int(pow((float)2,i)) - 1)) << (NeighborPoints - i)));
		if (tempCode < minLBP) minLBP = tempCode;
	}

    return minLBP;
}
