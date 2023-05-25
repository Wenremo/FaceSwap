// FaceSwapServer.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"

#include "SwapFace.h"
#include "FaceSwapper.h"
#include <opencv2/highgui/highgui.hpp>

int main(int argc, char* argv[])
{
	if (cvDetectLandMarksInitialize() < 0) {
		printf("Internal error\n");

		return -1;
	}	
	cv::String strModel = "D:/background/1.jpg";
	cv::String strUser  = "D:/background/4.jpg";
	cv::String strSave  = "D:/swap.jpg";
	char*      strAttr  = "D:/attribute.dat";

	SFaceInfo sFaceInfoModel;
	SFaceInfo sFaceInfoUser;
	FaceSwapper face_swapper;

	cv::Mat pImageModel = imread(strModel);
	cv::Mat pImageUser  = imread(strUser);
		
	if (cvDetectLandMarks(pImageModel.data, pImageModel.cols, pImageModel.rows, pImageModel.channels(), &sFaceInfoModel) != 0) return -1;
	if (cvDetectLandMarks(pImageUser.data,  pImageUser.cols,  pImageUser.rows,  pImageUser.channels(),  &sFaceInfoUser) != 0) return -1;

	vector<Point2f> pointsModel, pointsUser;
	cv::Mat pImageSwap = faceswap_main_part(pImageModel, pImageUser, &sFaceInfoModel, &sFaceInfoUser, pointsModel, pointsUser);

	cv::imwrite(strSave, pImageSwap);

	//write attributes
	int i, nSizeAttr = 0, aryAttr[1024];

	//model
	aryAttr[nSizeAttr++] = pImageModel.cols;
	aryAttr[nSizeAttr++] = pImageModel.rows;
	aryAttr[nSizeAttr++] = sFaceInfoModel.pFace[sFaceInfoModel.nMaxFaceIdx].rcFace.x;
	aryAttr[nSizeAttr++] = sFaceInfoModel.pFace[sFaceInfoModel.nMaxFaceIdx].rcFace.y;
	aryAttr[nSizeAttr++] = sFaceInfoModel.pFace[sFaceInfoModel.nMaxFaceIdx].rcFace.width;
	aryAttr[nSizeAttr++] = sFaceInfoModel.pFace[sFaceInfoModel.nMaxFaceIdx].rcFace.height;

	for (i = 0; i < LANDMARK_COUNT; i++) {
		aryAttr[nSizeAttr++] = sFaceInfoModel.pFace[sFaceInfoModel.nMaxFaceIdx].ptCord[i].x;
		aryAttr[nSizeAttr++] = sFaceInfoModel.pFace[sFaceInfoModel.nMaxFaceIdx].ptCord[i].y;
	}
	
	for (i = 0; i < pointsModel.size(); i++) {
		aryAttr[nSizeAttr++] = pointsModel.at(i).x;
		aryAttr[nSizeAttr++] = pointsModel.at(i).y;
	}

	//user
	aryAttr[nSizeAttr++] = pImageUser.cols;
	aryAttr[nSizeAttr++] = pImageUser.rows;
	aryAttr[nSizeAttr++] = sFaceInfoUser.pFace[sFaceInfoUser.nMaxFaceIdx].rcFace.x;
	aryAttr[nSizeAttr++] = sFaceInfoUser.pFace[sFaceInfoUser.nMaxFaceIdx].rcFace.y;
	aryAttr[nSizeAttr++] = sFaceInfoUser.pFace[sFaceInfoUser.nMaxFaceIdx].rcFace.width;
	aryAttr[nSizeAttr++] = sFaceInfoUser.pFace[sFaceInfoUser.nMaxFaceIdx].rcFace.height;

	for (i = 0; i < LANDMARK_COUNT; i++) {
		aryAttr[nSizeAttr++] = sFaceInfoUser.pFace[sFaceInfoUser.nMaxFaceIdx].ptCord[i].x;
		aryAttr[nSizeAttr++] = sFaceInfoUser.pFace[sFaceInfoUser.nMaxFaceIdx].ptCord[i].y;
	}

	for (i = 0; i < pointsUser.size(); i++) {
		aryAttr[nSizeAttr++] = pointsUser.at(i).x;
		aryAttr[nSizeAttr++] = pointsUser.at(i).y;
	}

	FILE* fp = fopen(strAttr, "w");
	for (i = 0; i < 392; i++) {
		fprintf(fp, "%d;", aryAttr[i]);
	}
	//fwrite(aryAttr, sizeof(int), nSizeAttr, fp);
	fclose(fp);

	printf(argv[3]);

	exit(0);

	return 0;
}

