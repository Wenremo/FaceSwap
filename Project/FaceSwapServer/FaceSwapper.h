#pragma once

#include "stdafx.h"

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/video/video.hpp>
// #include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/photo.hpp>

#include <iostream>

using namespace std;
using namespace cv;

#define MAX_FACENUM			64
#define LANDMARK_COUNT		68

typedef struct CF_Rect {
	int x;
	int y;
	int width;
	int height;
} CF_Rect;

typedef struct CF_Point {
	int x;
	int y;
} CF_Point;

typedef struct SFaceStru {
	CF_Rect		rcFace;							// face region
	CF_Point    ptCord[LANDMARK_COUNT];
} SFaceStru;

typedef struct SFaceInfo {
	int			nFaces;							// face count
	SFaceStru	pFace[MAX_FACENUM];				// face structure
	int			nMaxFaceIdx;					// max face index
} SFaceInfo;

cv::Mat faceswap_main(cv::Mat imgModel, cv::Mat imgUser, vector<Point2f> pointsModel, vector<Point2f> pointsUser);
cv::Mat faceswap_main_part(cv::Mat &pImageModel, cv::Mat &pImageUser, SFaceInfo* pFaceInfoModel, SFaceInfo* pFaceInfoUser, vector<Point2f> &pointsModel, vector<Point2f> &pointsUser);

