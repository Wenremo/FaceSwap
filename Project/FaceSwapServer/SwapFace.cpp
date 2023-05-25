/* Implementation of SwapFace class methods*/

#include "stdafx.h"
#include "SwapFace.h"

////////////////////////////////////////
// Brief description: 
// find faces usinng available cascades
///////////////////////////////////////
std::vector<cv::Rect> SwapFace::getFaces() {
	cv::CascadeClassifier face_cascade;
	face_cascade.load(CASCADE_PATH);
	
	std::vector<cv::Rect> faces;
	face_cascade.detectMultiScale(resizedFrame, faces, 
								  SCALE_FACTOR, MIN_NEIGHBOURS, 
								  2, cv::Size(WINDOW_SIZE, WINDOW_SIZE));

#if VISUALIZATION
	cv::Mat facesVisualization = resizedFrame.clone();
	for (int i = 0; i < faces.size(); i++) {
		cv::rectangle(facesVisualization, faces[i], cv::Scalar(255, 0, 255));
	}
	cv::imshow("detected faces", facesVisualization);
#endif 

	return faces;
}

////////////////////////////////////////
// Brief description: 
// function to build gaussian pyramid, 
// when each image is resized in x2
///////////////////////////////////////
std::vector<cv::Mat> SwapFace::buildGaussianPyr(cv::Mat img, int pyrLevel = PYRAMID_DEPTH) {
	std::vector<cv::Mat> gaussPyr(0, cv::Mat());
	cv::Mat downImg = img.clone();
	gaussPyr.push_back(downImg);

	for (int i = 0; i < pyrLevel; i++) {
		cv::pyrDown(downImg, downImg);
		gaussPyr.push_back(downImg);
	}

	return gaussPyr;
}

////////////////////////////////////////////
// Brief description: 
// function to build gaussian pyramid, 
// when each image is resized in x2.
// Overloaded function to get resized rects 
///////////////////////////////////////////
std::vector<std::pair<cv::Rect, cv::Mat>> SwapFace::buildGaussianPyr(cv::Rect rect, cv::Mat mask, int pyrLevel = PYRAMID_DEPTH) {
	std::vector<std::pair<cv::Rect, cv::Mat> > gaussPyr(0, std::make_pair(cv::Rect(), cv::Mat()));
	cv::Rect downRect = rect;
	cv::Mat maskRect;
	cv::resize(mask, maskRect, cv::Size(downRect.width, downRect.height));

	gaussPyr.push_back(std::make_pair(downRect, maskRect));

	for (int i = 0; i < pyrLevel; i++) {
		downRect.x = downRect.x / 2;
		downRect.y = downRect.y / 2;
		downRect.width = downRect.width / 2;
		downRect.height = downRect.height / 2;
		cv::resize(maskRect, maskRect, cv::Size(downRect.width, downRect.height));
		gaussPyr.push_back(std::make_pair(downRect, maskRect));
	}

	return gaussPyr;
}

//////////////////////////////////////////////////////
// Brief description: 
// function to build laplacian pyramid, 
// when each image is obtained by substraction of real
// image and its resized analog
//////////////////////////////////////////////////////
std::vector<cv::Mat> SwapFace::buildLaplPyr(std::vector<cv::Mat> gaussPyr, int pyrLevel = PYRAMID_DEPTH) {
	std::vector<cv::Mat> laplPyr(0, cv::Mat());
	laplPyr.push_back(gaussPyr[pyrLevel]);

	cv::Mat upImg;
	cv::Mat laplMat;

	for (int i = pyrLevel; i >= 1; i--) {
		cv::pyrUp(gaussPyr[i], upImg);
		laplMat = gaussPyr[i - 1] - upImg;
		laplPyr.push_back(laplMat);
	}

	return laplPyr;
}

///////////////////////////////////////
// Brief description: 
// KMeans using to extract skin region 
///////////////////////////////////////
cv::Mat SwapFace::segmentFace(cv::Mat src) {
	cv::Mat samples(src.rows * src.cols, 3, CV_32F);
	for (int y = 0; y < src.rows; y++) {

		for (int x = 0; x < src.cols; x++) {
			for (int z = 0; z < 3; z++) {
				samples.at<float>(y + x*src.rows, z) = src.at<cv::Vec3b>(y, x)[z];
			}
		}
	}

	cv::Mat centers;
	cv::Mat labels;
	kmeans(samples, CLUSTER_COUNT, labels, 
		   cv::TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 10000, 0.0001), 
		   CLUSTER_ATTEMPTS, cv::KMEANS_PP_CENTERS, centers);

	cv::Mat resMask(src.size(), CV_8UC1);
	for (int y = 0; y < src.rows; y++) {
		uchar* dataMask = resMask.data + resMask.step.buf[0] * y;

		for (int x = 0; x < src.cols; x++) {
			dataMask[x] = (uchar)labels.at<int>(y + x*src.rows, 0) * 255;
		}
	}

	// check if inverse needed
	int whites = 0;
	int blacks = 0;

	cv::Mat ellipse = cv::Mat(src.rows, src.cols, CV_8UC1, cv::Scalar(0));
	cv::ellipse(ellipse, cv::RotatedRect(cv::Point(ellipse.cols / 2, ellipse.rows / 2), cv::Size(3 * ellipse.cols / 4, ellipse.rows), 0), cv::Scalar(255), -1);

	for (int y = 0; y < src.rows; y++) {
		uchar* dataMask = resMask.data + resMask.step.buf[0] * y;
		uchar* dataEllipse = ellipse.data + ellipse.step.buf[0] * y;

		for (int x = 0; x < src.cols; x++) {
			if (dataEllipse[x] == 255) {
				if (dataMask[x] == 255) whites++;
				else blacks++;
			}
		}
	}

	if (blacks > whites) resMask = 255 - resMask;

	return resMask;
}

////////////////////////////////////
// Brief description: 
// use 3sigma rule for segmentation  
////////////////////////////////////
cv::Mat SwapFace::perform3SigmaRule(cv::Mat& src, cv::Mat mask) {
	cv::Scalar mean, std;
	cv::meanStdDev(src, mean, std, mask);

	cv::Mat maskChannel1, maskChannel2, maskUpd;
	cv::threshold(src, maskChannel1, mean[0] - 2 * std[0], 255, cv::THRESH_BINARY);
	cv::threshold(src, maskChannel2, mean[0] + 2 * std[0], 255, cv::THRESH_BINARY_INV);
	cv::bitwise_and(maskChannel1, maskChannel2, maskUpd);

	return maskUpd;
}

/////////////////////////////////////////
// Brief description: 
// split matrix and process each channel 
/////////////////////////////////////////
cv::Mat SwapFace::getStatisticsMask(cv::Mat src) {
	cv::Mat mask = cv::Mat(src.rows, src.cols, CV_8UC1, cv::Scalar(0));
	cv::ellipse(mask, cv::RotatedRect(cv::Point(mask.cols / 2, mask.rows / 2), 
				      cv::Size(mask.cols / 2, 3 * mask.rows / 4), 0), cv::Scalar(255), -1);

	cv::Mat hsvSRC;
	cv::cvtColor(src, hsvSRC, cv::COLOR_RGB2HSV);
	cv::Mat bgr[3];
	split(hsvSRC, bgr);

	cv::Mat mask1 = perform3SigmaRule(bgr[0], mask);
	cv::Mat mask2 = perform3SigmaRule(bgr[1], mask);
	cv::Mat mask3 = perform3SigmaRule(bgr[2], mask);

	cv::Mat combMask;
	cv::bitwise_and(mask1, mask2, mask2);
	cv::bitwise_and(mask2, mask3, combMask);

	return combMask;
}

/////////////////////////////////////////
// Brief description: 
// walk iteratively through mask and cut 
// borders out of statisctics 
/////////////////////////////////////////
void SwapFace::clarifyBorders(cv::Mat face, cv::Mat& mask, const int iterations = 7) {
	cv::Scalar mean, std;
	cv::meanStdDev(face, mean, std, mask);

	cv::Mat element = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));

	for (int i = 0; i < iterations; i++) {
		cv::Mat eroded;
		morphologyEx(mask, eroded, cv::MORPH_ERODE, element);
		cv::Mat diff = mask - eroded;

		for (int y = 0; y < mask.rows; y++) {
			uchar* dataMask = mask.data + mask.step.buf[0] * y;
			uchar* dataDiff = diff.data + diff.step.buf[0] * y;

			for (int x = 0; x < mask.cols; x++) {
				if (dataDiff[x] == 255) {
					bool goodPixel = true;

					for (int channel = 0; channel < 3; channel++) {
						const float leftBorder  = mean[channel] - 1.0 * std[channel];
						const float rightBorder = mean[channel] + 1.0 * std[channel];
						
						if (face.at<cv::Vec3b>(y, x)[channel] > rightBorder ||
							face.at<cv::Vec3b>(y, x)[channel] < leftBorder) {
							goodPixel = false;
							break;
						}
					}

					if (!goodPixel) 
						dataMask[x] = 0;
				}
			}
		}
	}
}

///////////////////////////////////////
// Brief description: 
// Find sking region and postprocess it 
///////////////////////////////////////
cv::Mat SwapFace::findMask(cv::Mat face) {
	cv::Mat faceResized;
	cv::resize(face, faceResized, cv::Size(face.cols / 4, face.rows / 4), 0, 0, cv::INTER_NEAREST);

	cv::Mat resMask = segmentFace(faceResized);
	cv::resize(resMask, resMask, cv::Size(face.cols, face.rows), 0, 0, cv::INTER_NEAREST);
	cv::GaussianBlur(resMask, resMask, cv::Size(11, 11), 5, 5);
	cv::threshold(resMask, resMask, 128, 255, cv::THRESH_BINARY);

	cv::Mat statisticsMask = getStatisticsMask(face);
	cv::bitwise_and(resMask, statisticsMask, resMask);

	int maxLength = 0;
	int maxInd = 0;

	std::vector<std::vector<cv::Point> > contours;
	cv::findContours(resMask, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);

	if (contours.size() == 0)
		return resMask;

	for (int i = 0; i < contours.size(); i++) {
		double length = contours[i].size();
		if (length > maxLength) {
			maxLength = length;
			maxInd = i;
		}
	}

	resMask = 0;
	drawContours(resMask, contours, maxInd, cv::Scalar(255), CV_FILLED);

	std::vector<cv::Point> hull;
	cv::convexHull(contours[maxInd], hull);

	if (hull.size() == 0) {
		return resMask;
	}

	resMask = 0;
	auto hullRes = std::vector<std::vector<cv::Point> >(1, hull);
	drawContours(resMask, hullRes, 0, cv::Scalar(255), -1);

	cv::Mat restrictionEllipse = cv::Mat(resMask.rows, resMask.cols, CV_8UC1, cv::Scalar(0));
	cv::ellipse(restrictionEllipse, cv::RotatedRect(cv::Point(resMask.cols / 2, resMask.rows / 2), cv::Size(resMask.cols - 6, resMask.rows - 6), 0), cv::Scalar(255), -1);
	cv::bitwise_and(restrictionEllipse, resMask, resMask);

	clarifyBorders(face, resMask);

	return resMask;
}

///////////////////////////////////////
// Brief description: 
// Function to copy one face to another 
// with some restrictions 
///////////////////////////////////////
std::pair<cv::Mat, cv::Mat> SwapFace::copySrcToDstUsingMask(cv::Mat imgSrc, cv::Mat imgDst, cv::Mat maskSrc, cv::Mat maskDst) {
	cv::Mat result = imgDst.clone();
	cv::Mat blackMask = cv::Mat(result.size(), CV_8UC1, cv::Scalar(0));

	for (int y = 0; y < imgSrc.rows; y++) {
		for (int x = 0; x < imgSrc.cols; x++) {
			if (maskDst.at<uchar>(y, x) == 255) {
				result.at<cv::Vec3b>(y, x) = cv::Vec3b(255, 255, 255);

				if (maskSrc.at<uchar>(y, x) == 255) {
					result.at<cv::Vec3b>(y, x) = imgSrc.at<cv::Vec3b>(y, x);
				}
				else {
					blackMask.at<uchar>(y, x) = 255;
				}
			}
		}
	}

	return std::make_pair(result, blackMask);
}

///////////////////////////////////////
// Brief description: 
// get distance between two points 
///////////////////////////////////////
float SwapFace::getDist(cv::Point& p1, cv::Point& p2) {
	return sqrt((float)(p1.x - p2.x) * (p1.x - p2.x) + (float)(p1.y - p2.y) * (p1.y - p2.y));
}

///////////////////////////////////////////
// Brief description: 
// get closest point to point P in contour 
///////////////////////////////////////////
cv::Point SwapFace::findClosesPoint(cv::Point& p, std::vector<cv::Point>& contour) {
	float mindist = 10000;
	int minInd = 0;

	for (int i = 0; i < contour.size(); i++) {
		cv::Point contourPoint = contour[i];

		float dist = getDist(contourPoint, p);
		if (dist < mindist) {
			mindist = dist;
			minInd = i;
		}
	}

	return contour[minInd];
}

//////////////////////
// Brief description: 
// inpainting 
//////////////////////
cv::Mat SwapFace::stretchFace(cv::Mat imgSrc, cv::Mat imgDst, cv::Mat maskSrc, cv::Mat maskDst) {
	cv::resize(imgSrc, imgSrc, imgDst.size(), cv::INTER_NEAREST);
	cv::resize(maskSrc, maskSrc, maskDst.size(), cv::INTER_NEAREST);
	maskSrc *= 255;

	cv::Mat cuttedFace = cv::Mat(imgSrc.size(), CV_8UC1);
	imgSrc.copyTo(cuttedFace, maskSrc);

	auto stretched = copySrcToDstUsingMask(imgSrc, imgDst, maskSrc, maskDst);
	cv::Mat stretchedFace = stretched.first;
	cv::Mat blackMask = stretched.second;

	cv::inpaint(stretchedFace, blackMask, stretchedFace, 3, cv::INPAINT_NS);

	return stretchedFace;
}

///////////////////////////////////////////
// Brief description: 
// fit left image to right with right mask 
// and the same for the right image 
//////////////////////////////////////////
std::pair<cv::Mat, cv::Mat> SwapFace::fitImagesToEachOther(cv::Mat leftImg, cv::Mat rightImg, cv::Mat leftMask, cv::Mat rightMask) {
	cv::Mat leftFaceFitted = stretchFace(leftImg, rightImg, leftMask, rightMask);
	cv::Mat rightFaceFitted = stretchFace(rightImg, leftImg, rightMask, leftMask);

	return std::make_pair(leftFaceFitted, rightFaceFitted);
}

////////////////////////////////
// Brief description: 
// main algorithm to swap faces 
////////////////////////////////
void SwapFace::swapFace(cv::Rect lFace, cv::Rect rFace) {
	cv::Mat leftFaceImg = cv::Mat(resizedFrame, lFace);
	cv::Mat rightFaceImg = cv::Mat(resizedFrame, rFace);

	//leftFaceImg = cv::imread("C:/Users/Alfred/Desktop/SwapFace/testData/l1.png");
	//rightFaceImg = cv::imread("C:/Users/Alfred/Desktop/SwapFace/testData/l2.png");
	cv::Mat leftMask = findMask(leftFaceImg);
	cv::Mat rightMask = findMask(rightFaceImg);

	auto fittedImages = fitImagesToEachOther(leftFaceImg, rightFaceImg, leftMask, rightMask);

	cv::Mat comb = resizedFrame.clone();
	fittedImages.first.copyTo(comb(rFace), rightMask);
	fittedImages.second.copyTo(comb(lFace), leftMask);

	cv::imshow("comb", comb);
	cv::imshow("resizedFrame", resizedFrame);

	resizedFrame.convertTo(resizedFrame, CV_32F, 1.0 / 255.0);
	comb.convertTo(comb, CV_32F, 1.0 / 255.0);

	auto gpSrc = buildGaussianPyr(resizedFrame);
	auto lpSrc = buildLaplPyr(gpSrc);

	auto gpComb = buildGaussianPyr(comb);
	auto lpComb = buildLaplPyr(gpComb);

	auto gpRectL = buildGaussianPyr(lFace, leftMask);
	auto gpRectR = buildGaussianPyr(rFace, rightMask);

	std::reverse(gpRectL.begin(), gpRectL.end());
	std::reverse(gpRectR.begin(), gpRectR.end());

	std::vector<cv::Mat> laplStitched;
	laplStitched.push_back(lpComb[0]);

	for (int i = 1; i < lpSrc.size(); i++) {
		cv::Rect rectFaceL = gpRectL[i].first;
		cv::Rect rectFaceR = gpRectR[i].first;

		cv::Mat maskFaceL = gpRectL[i].second;
		cv::Mat maskFaceR = gpRectR[i].second;
		maskFaceL *= 255;
		maskFaceR *= 255;

		cv::Mat leftFaceImg1 = cv::Mat(lpComb[i], rectFaceR);
		cv::Mat rightFaceImg1 = cv::Mat(lpComb[i], rectFaceL);
		if (i == 2) cv::imshow("leftFaceImg2", leftFaceImg1);
		if (i == 3) cv::imshow("leftFaceImg3", leftFaceImg1);
		if (i == 4) cv::imshow("leftFaceImg4", leftFaceImg1);

		cv::Mat combUn1 = lpSrc[i].clone();
		cv::Mat element2;
		if (i == 1) element2 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
		if (i == 2) element2 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7));
		if (i == 3) element2 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(11, 11));
		if (i == 4) element2 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(13, 13));

		morphologyEx(maskFaceR, maskFaceR, cv::MORPH_ERODE, element2);
		morphologyEx(maskFaceL, maskFaceL, cv::MORPH_ERODE, element2);

		leftFaceImg1.copyTo(combUn1(rectFaceR), maskFaceR);
		rightFaceImg1.copyTo(combUn1(rectFaceL), maskFaceL);

		if (i == 2) cv::imshow("combUn2", combUn1);
		if (i == 3) cv::imshow("combUn3", combUn1);
		if (i == 4) cv::imshow("combUn4", combUn1);

		laplStitched.push_back(combUn1);
	}

	cv::Mat usualStitched = laplStitched[0].clone();
	for (int i = 1; i <= 4; i++) {
		cv::Mat up;
		cv::pyrUp(usualStitched, up);
		usualStitched = up + laplStitched[i];

		if (i == 2) cv::imshow("usualStitched2", usualStitched);
		if (i == 3) cv::imshow("usualStitched3", usualStitched);
		if (i == 4) cv::imshow("usualStitched4", usualStitched);
	}
	usualStitched.convertTo(usualStitched, CV_8UC3, 255);
	cv::resize(usualStitched, resultFrame, cv::Size(fullFrame.cols, fullFrame.rows));	
}

#define TEST_DRAW		0
cv::Mat SwapFace::swapFace(cv::Mat &pImageModel, cv::Mat &pImageUser, SFaceInfo* pFaceInfoModel, SFaceInfo* pFaceInfoUser)
{
	Point2f pt;
	vector<Point2f> pointsModel;
	vector<Point2f> pointsUser;

	// Find convex hull
	vector<Point2f> hullUser;
	vector<Point2f> hullModel;
	vector<int> hullIndex;

#if 1 //use middle
	int i;
	int nGapTopModel = 1*((pFaceInfoModel->pFace[0].ptCord[38].y - pFaceInfoModel->pFace[0].ptCord[20].y) + (pFaceInfoModel->pFace[0].ptCord[43].y - pFaceInfoModel->pFace[0].ptCord[23].y)) / 4;
	int nGapTopUser  = 1*((pFaceInfoUser->pFace[0].ptCord[38].y  - pFaceInfoUser->pFace[0].ptCord[20].y)  + (pFaceInfoUser->pFace[0].ptCord[43].y  - pFaceInfoUser->pFace[0].ptCord[23].y))  / 4;

	for (i = 17; i <= 26; i++) {
		pt = Point2f((float)pFaceInfoModel->pFace[0].ptCord[i].x, (float)max(0, pFaceInfoModel->pFace[0].ptCord[i].y - nGapTopModel));
		pointsModel.push_back(pt);
		pt = Point2f((float)pFaceInfoUser->pFace[0].ptCord[i].x,  (float)max(0, pFaceInfoUser->pFace[0].ptCord[i].y  - nGapTopUser));
		pointsUser.push_back(pt);
	}

	//////////////////////////////////////////////////////////////////////////
	pt = Point2f((float)((pFaceInfoModel->pFace[0].ptCord[26].x + pFaceInfoModel->pFace[0].ptCord[16].x) / 2), (float)((pFaceInfoModel->pFace[0].ptCord[26].y + pFaceInfoModel->pFace[0].ptCord[16].y) / 2));
	pointsModel.push_back(pt);
	pt = Point2f((float)((pFaceInfoUser->pFace[0].ptCord[26].x  + pFaceInfoUser->pFace[0].ptCord[16].x) / 2),  (float)((pFaceInfoUser->pFace[0].ptCord[26].y  + pFaceInfoUser->pFace[0].ptCord[16].y) / 2));
	pointsUser.push_back(pt);

	pt = Point2f((float)((pFaceInfoModel->pFace[0].ptCord[45].x + pFaceInfoModel->pFace[0].ptCord[14].x) / 2), (float)((pFaceInfoModel->pFace[0].ptCord[45].y + pFaceInfoModel->pFace[0].ptCord[14].y) / 2));
	pointsModel.push_back(pt);
	pt = Point2f((float)((pFaceInfoUser->pFace[0].ptCord[45].x  + pFaceInfoUser->pFace[0].ptCord[14].x) / 2),  (float)((pFaceInfoUser->pFace[0].ptCord[45].y  + pFaceInfoUser->pFace[0].ptCord[14].y) / 2));
	pointsUser.push_back(pt);

	pt = Point2f((float)((pFaceInfoModel->pFace[0].ptCord[46].x + pFaceInfoModel->pFace[0].ptCord[13].x) / 2), (float)((pFaceInfoModel->pFace[0].ptCord[46].y + pFaceInfoModel->pFace[0].ptCord[13].y) / 2));
	pointsModel.push_back(pt);
	pt = Point2f((float)((pFaceInfoUser->pFace[0].ptCord[46].x  + pFaceInfoUser->pFace[0].ptCord[13].x) / 2),  (float)((pFaceInfoUser->pFace[0].ptCord[46].y  + pFaceInfoUser->pFace[0].ptCord[13].y) / 2));
	pointsUser.push_back(pt);

	pt = Point2f((float)((pFaceInfoModel->pFace[0].ptCord[35].x + pFaceInfoModel->pFace[0].ptCord[13].x) / 2), (float)((pFaceInfoModel->pFace[0].ptCord[35].y + pFaceInfoModel->pFace[0].ptCord[13].y) / 2));
	pointsModel.push_back(pt);
	pt = Point2f((float)((pFaceInfoUser->pFace[0].ptCord[35].x  + pFaceInfoUser->pFace[0].ptCord[13].x) / 2), (float)((pFaceInfoUser->pFace[0].ptCord[35].y  + pFaceInfoUser->pFace[0].ptCord[13].y) / 2));
	pointsUser.push_back(pt);

	pt = Point2f((float)((pFaceInfoModel->pFace[0].ptCord[12].x + pFaceInfoModel->pFace[0].ptCord[54].x) / 2), (float)((pFaceInfoModel->pFace[0].ptCord[12].y + pFaceInfoModel->pFace[0].ptCord[54].y) / 2));
	pointsModel.push_back(pt);
	pt = Point2f((float)((pFaceInfoUser->pFace[0].ptCord[12].x  + pFaceInfoUser->pFace[0].ptCord[54].x) / 2),  (float)((pFaceInfoUser->pFace[0].ptCord[12].y  + pFaceInfoUser->pFace[0].ptCord[54].y) / 2));
	pointsUser.push_back(pt);

	pt = Point2f((float)((pFaceInfoModel->pFace[0].ptCord[11].x + pFaceInfoModel->pFace[0].ptCord[54].x) / 2), (float)((pFaceInfoModel->pFace[0].ptCord[11].y + pFaceInfoModel->pFace[0].ptCord[54].y) / 2));
	pointsModel.push_back(pt);
	pt = Point2f((float)((pFaceInfoUser->pFace[0].ptCord[11].x  + pFaceInfoUser->pFace[0].ptCord[54].x) / 2),  (float)((pFaceInfoUser->pFace[0].ptCord[11].y  + pFaceInfoUser->pFace[0].ptCord[54].y) / 2));
	pointsUser.push_back(pt);

	pt = Point2f((float)((pFaceInfoModel->pFace[0].ptCord[10].x + pFaceInfoModel->pFace[0].ptCord[55].x) / 2), (float)((pFaceInfoModel->pFace[0].ptCord[10].y + pFaceInfoModel->pFace[0].ptCord[55].y) / 2));
	pointsModel.push_back(pt);
	pt = Point2f((float)((pFaceInfoUser->pFace[0].ptCord[10].x  + pFaceInfoUser->pFace[0].ptCord[55].x) / 2),  (float)((pFaceInfoUser->pFace[0].ptCord[10].y  + pFaceInfoUser->pFace[0].ptCord[55].y) / 2));
	pointsUser.push_back(pt);

	pt = Point2f((float)((pFaceInfoModel->pFace[0].ptCord[9].x + pFaceInfoModel->pFace[0].ptCord[56].x) / 2), (float)((pFaceInfoModel->pFace[0].ptCord[9].y + pFaceInfoModel->pFace[0].ptCord[56].y) / 2));
	pointsModel.push_back(pt);
	pt = Point2f((float)((pFaceInfoUser->pFace[0].ptCord[9].x  + pFaceInfoUser->pFace[0].ptCord[56].x) / 2),  (float)((pFaceInfoUser->pFace[0].ptCord[9].y  + pFaceInfoUser->pFace[0].ptCord[56].y) / 2));
	pointsUser.push_back(pt);

	pt = Point2f((float)((pFaceInfoModel->pFace[0].ptCord[8].x + pFaceInfoModel->pFace[0].ptCord[57].x) / 2), (float)((pFaceInfoModel->pFace[0].ptCord[8].y + pFaceInfoModel->pFace[0].ptCord[57].y) / 2));
	pointsModel.push_back(pt);
	pt = Point2f((float)((pFaceInfoUser->pFace[0].ptCord[8].x  + pFaceInfoUser->pFace[0].ptCord[57].x) / 2),  (float)((pFaceInfoUser->pFace[0].ptCord[8].y  + pFaceInfoUser->pFace[0].ptCord[57].y) / 2));
	pointsUser.push_back(pt);

	pt = Point2f((float)((pFaceInfoModel->pFace[0].ptCord[7].x + pFaceInfoModel->pFace[0].ptCord[58].x) / 2), (float)((pFaceInfoModel->pFace[0].ptCord[7].y + pFaceInfoModel->pFace[0].ptCord[58].y) / 2));
	pointsModel.push_back(pt);
	pt = Point2f((float)((pFaceInfoUser->pFace[0].ptCord[7].x  + pFaceInfoUser->pFace[0].ptCord[58].x) / 2),  (float)((pFaceInfoUser->pFace[0].ptCord[7].y  + pFaceInfoUser->pFace[0].ptCord[58].y) / 2));
	pointsUser.push_back(pt);

	pt = Point2f((float)((pFaceInfoModel->pFace[0].ptCord[6].x + pFaceInfoModel->pFace[0].ptCord[59].x) / 2), (float)((pFaceInfoModel->pFace[0].ptCord[6].y + pFaceInfoModel->pFace[0].ptCord[59].y) / 2));
	pointsModel.push_back(pt);
	pt = Point2f((float)((pFaceInfoUser->pFace[0].ptCord[6].x  + pFaceInfoUser->pFace[0].ptCord[59].x) / 2),  (float)((pFaceInfoUser->pFace[0].ptCord[6].y  + pFaceInfoUser->pFace[0].ptCord[59].y) / 2));
	pointsUser.push_back(pt);

	pt = Point2f((float)((pFaceInfoModel->pFace[0].ptCord[5].x + pFaceInfoModel->pFace[0].ptCord[48].x) / 2), (float)((pFaceInfoModel->pFace[0].ptCord[5].y + pFaceInfoModel->pFace[0].ptCord[48].y) / 2));
	pointsModel.push_back(pt);
	pt = Point2f((float)((pFaceInfoUser->pFace[0].ptCord[5].x  + pFaceInfoUser->pFace[0].ptCord[48].x) / 2),  (float)((pFaceInfoUser->pFace[0].ptCord[5].y  + pFaceInfoUser->pFace[0].ptCord[48].y) / 2));
	pointsUser.push_back(pt);

	pt = Point2f((float)((pFaceInfoModel->pFace[0].ptCord[4].x + pFaceInfoModel->pFace[0].ptCord[48].x) / 2), (float)((pFaceInfoModel->pFace[0].ptCord[4].y + pFaceInfoModel->pFace[0].ptCord[48].y) / 2));
	pointsModel.push_back(pt);
	pt = Point2f((float)((pFaceInfoUser->pFace[0].ptCord[4].x  + pFaceInfoUser->pFace[0].ptCord[48].x) / 2),  (float)((pFaceInfoUser->pFace[0].ptCord[4].y  + pFaceInfoUser->pFace[0].ptCord[48].y) / 2));
	pointsUser.push_back(pt);

	pt = Point2f((float)((pFaceInfoModel->pFace[0].ptCord[3].x + pFaceInfoModel->pFace[0].ptCord[31].x) / 2), (float)((pFaceInfoModel->pFace[0].ptCord[3].y + pFaceInfoModel->pFace[0].ptCord[31].y) / 2));
	pointsModel.push_back(pt);
	pt = Point2f((float)((pFaceInfoUser->pFace[0].ptCord[3].x  + pFaceInfoUser->pFace[0].ptCord[31].x) / 2),  (float)((pFaceInfoUser->pFace[0].ptCord[3].y  + pFaceInfoUser->pFace[0].ptCord[31].y) / 2));
	pointsUser.push_back(pt);

	pt = Point2f((float)((pFaceInfoModel->pFace[0].ptCord[3].x + pFaceInfoModel->pFace[0].ptCord[41].x) / 2), (float)((pFaceInfoModel->pFace[0].ptCord[3].y + pFaceInfoModel->pFace[0].ptCord[41].y) / 2));
	pointsModel.push_back(pt);
	pt = Point2f((float)((pFaceInfoUser->pFace[0].ptCord[3].x  + pFaceInfoUser->pFace[0].ptCord[41].x) / 2),  (float)((pFaceInfoUser->pFace[0].ptCord[3].y  + pFaceInfoUser->pFace[0].ptCord[41].y) / 2));
	pointsUser.push_back(pt);

	pt = Point2f((float)((pFaceInfoModel->pFace[0].ptCord[2].x + pFaceInfoModel->pFace[0].ptCord[36].x) / 2), (float)((pFaceInfoModel->pFace[0].ptCord[2].y + pFaceInfoModel->pFace[0].ptCord[36].y) / 2));
	pointsModel.push_back(pt);
	pt = Point2f((float)((pFaceInfoUser->pFace[0].ptCord[2].x  + pFaceInfoUser->pFace[0].ptCord[36].x) / 2),  (float)((pFaceInfoUser->pFace[0].ptCord[2].y  + pFaceInfoUser->pFace[0].ptCord[36].y) / 2));
	pointsUser.push_back(pt);

	pt = Point2f((float)((pFaceInfoModel->pFace[0].ptCord[0].x + pFaceInfoModel->pFace[0].ptCord[17].x) / 2), (float)((pFaceInfoModel->pFace[0].ptCord[0].y + pFaceInfoModel->pFace[0].ptCord[17].y) / 2));
	pointsModel.push_back(pt);
	pt = Point2f((float)((pFaceInfoUser->pFace[0].ptCord[0].x  + pFaceInfoUser->pFace[0].ptCord[17].x) / 2),  (float)((pFaceInfoUser->pFace[0].ptCord[0].y  + pFaceInfoUser->pFace[0].ptCord[17].y) / 2));
	pointsUser.push_back(pt);

	for (size_t i = 0; i < pointsModel.size(); i++) {
		hullModel.push_back(pointsModel[i]);
		hullUser.push_back(pointsUser[i]);
	}
// 	hullModel.push_back(pointsModel[0]);
// 	hullUser.push_back(pointsUser[0]);

#else //use landmark
	for (int i = 0; i < LANDMARK_COUNT; i++) {
		if (i < 17) continue;
		//if (i == 1 || i == 2 || i == 3 || i == 4 || i == 5 || i == 6 || i == 7 || i == 9 || i == 10 || i == 12 || i == 14 || i == 15) continue;

		pt = Point2f((float)pFaceInfoModel->pFace[0].ptCord[i].x, (float)pFaceInfoModel->pFace[0].ptCord[i].y);
		pointsModel.push_back(pt);

		pt = Point2f((float)pFaceInfoUser->pFace[0].ptCord[i].x, (float)pFaceInfoUser->pFace[0].ptCord[i].y);
		pointsUser.push_back(pt);
	}

	cv::convexHull(pointsUser, hullIndex, false, false);

	for (size_t i = 0; i < hullIndex.size(); i++) {
		hullModel.push_back(pointsModel[hullIndex[i]]);
		hullUser.push_back(pointsUser[hullIndex[i]]);
	}
#endif

#if TEST_DRAW //for draw
	for (int i = 0; i < /*hullIndex*/pointsModel.size(); i++) {
		//drawMarker(img1UserCn, cv::Point(pointsUser[hullIndex[i]].x, pointsUser[hullIndex[i]].y), Scalar(0, 0, 255), 1, 4);
		drawMarker(pImageModel, cv::Point(hullModel[i].x, hullModel[i].y), Scalar(0, 0, 255), 1, 4);
		drawMarker(pImageUser,  cv::Point(hullUser[i].x, hullUser[i].y), Scalar(0, 0, 255), 1, 4);
	}

	// 	Mat resize_imgUser;
	// 	cv::resize(img1UserCn, resize_imgUser, Size(), resize_x, resize_y);
	cv::imshow("imgModel", pImageModel);
	cv::imshow("imgUser",  pImageUser);
#endif

	// Calculate mask
	vector<Point> hull8UModel;
	vector<Point> hull8UUser;
	for (size_t i = 0; i < hullModel.size(); i++) {
		Point ptM(hullModel[i].x, hullModel[i].y);
		hull8UModel.push_back(ptM);
		Point ptU(hullUser[i].x, hullUser[i].y);
		hull8UUser.push_back(ptU);
	}

	Mat maskModel = Mat::zeros(pImageModel.rows, pImageModel.cols, pImageModel.depth());
	fillConvexPoly(maskModel, &hull8UModel[0], hull8UModel.size(), Scalar(255,255,255), 8);

	hull8UModel.clear();
	for (size_t i = 0; i < 5; i++) {
		Point ptM(hullModel[i].x, hullModel[i].y);
		hull8UModel.push_back(ptM);
	}
	Point ptM(hullModel[hullModel.size()-1].x, hullModel[hullModel.size()-1].y);
	hull8UModel.push_back(ptM);
	fillConvexPoly(maskModel, &hull8UModel[0], hull8UModel.size(), Scalar(255,255,255), 8);

	hull8UModel.clear();
	for (size_t i = 5; i < 11; i++) {
		Point ptM(hullModel[i].x, hullModel[i].y);
		hull8UModel.push_back(ptM);
	}
	fillConvexPoly(maskModel, &hull8UModel[0], hull8UModel.size(), Scalar(255,255,255), 8);

	cv::Size feather_amount;
	feather_amount.width = feather_amount.height = (int)cv::norm(pointsModel[0] - pointsModel[16]) / 8;
// 	cv::erode(maskModel, maskModel, getStructuringElement(cv::MORPH_RECT, feather_amount), cv::Point(-1, -1), 1, cv::BORDER_CONSTANT, cv::Scalar(0));
	//cv::blur(mask, mask, feather_amount, cv::Point(-1, -1), cv::BORDER_CONSTANT);

	Mat maskUser = Mat::zeros(pImageUser.rows, pImageUser.cols, pImageUser.depth());
	fillConvexPoly(maskUser, &hull8UUser[0], hull8UUser.size(), Scalar(255,255,255), 8);

	hull8UUser.clear();
	for (size_t i = 0; i < 5; i++) {
		Point ptU(hullUser[i].x, hullUser[i].y);
		hull8UUser.push_back(ptU);
	}
	Point ptU(hullUser[hullModel.size()-1].x, hullUser[hullModel.size()-1].y);
	hull8UUser.push_back(ptU);
	fillConvexPoly(maskUser, &hull8UUser[0], hull8UUser.size(), Scalar(255,255,255), 8);

	hull8UUser.clear();
	for (size_t i = 5; i < 11; i++) {
		Point ptU(hullUser[i].x, hullUser[i].y);
		hull8UUser.push_back(ptU);
	}
	fillConvexPoly(maskUser, &hull8UUser[0], hull8UUser.size(), Scalar(255,255,255), 8);

	feather_amount.width = feather_amount.height = (int)cv::norm(pointsUser[0] - pointsUser[16]) / 8;
// 	cv::erode(maskUser, maskUser, getStructuringElement(cv::MORPH_RECT, feather_amount), cv::Point(-1, -1), 1, cv::BORDER_CONSTANT, cv::Scalar(0));
	//cv::blur(mask, mask, feather_amount, cv::Point(-1, -1), cv::BORDER_CONSTANT);


	cv::Rect lFace = cv::Rect(pFaceInfoModel->pFace[0].rcFace.x, pFaceInfoModel->pFace[0].rcFace.y, pFaceInfoModel->pFace[0].rcFace.width, pFaceInfoModel->pFace[0].rcFace.height);
	cv::Rect rFace = cv::Rect(pFaceInfoUser->pFace[0].rcFace.x,  pFaceInfoUser->pFace[0].rcFace.y,  pFaceInfoUser->pFace[0].rcFace.width,  pFaceInfoUser->pFace[0].rcFace.height);
	
	cv::Mat leftFaceImg = cv::Mat(pImageModel, lFace);
	cv::Mat rightFaceImg = cv::Mat(pImageUser, rFace);

// 	Rect maskModel = boundingRect(hullModel);

	//leftFaceImg = cv::imread("C:/Users/Alfred/Desktop/SwapFace/testData/l1.png");
	//rightFaceImg = cv::imread("C:/Users/Alfred/Desktop/SwapFace/testData/l2.png");
	cv::Mat leftMask = cv::Mat(maskModel, lFace);// + findMask(leftFaceImg);
	cv::Mat rightMask = cv::Mat(maskUser, rFace);// + findMask(rightFaceImg);
// 	cv::Mat leftMask = findMask(leftFaceImg);
// 	cv::Mat rightMask = findMask(rightFaceImg);

	auto fittedImages = fitImagesToEachOther(leftFaceImg, rightFaceImg, leftMask, rightMask);

#if TEST_DRAW //for draw
	cv::imshow("leftMask", leftMask);
	cv::imshow("rightMask",  rightMask);
	cv::imshow("leftFaceImg", leftFaceImg);
	cv::imshow("rightFaceImg",  rightFaceImg);
#endif

	cv::Mat combModel = pImageModel.clone();
	cv::Mat combUser  = pImageUser.clone();
	fittedImages.first.copyTo(combUser(rFace), rightMask);
	fittedImages.second.copyTo(combModel(lFace), leftMask);

#if TEST_DRAW //for draw
	cv::imshow("combModel", combModel);
	cv::imshow("combUser",  combUser);
#endif




#if 0
	pImageUser.convertTo(pImageUser, CV_32F, 1.0 / 255.0);
	pImageModel.convertTo(pImageModel, CV_32F, 1.0 / 255.0);
	combUser.convertTo(combUser, CV_32F, 1.0 / 255.0);
	combModel.convertTo(combModel, CV_32F, 1.0 / 255.0);

	auto gpSrcUser = buildGaussianPyr(pImageUser);
	auto lpSrcUser = buildLaplPyr(gpSrcUser);
	auto gpSrcModel= buildGaussianPyr(pImageModel);
	auto lpSrcModel= buildLaplPyr(gpSrcModel);

	auto gpCombModel = buildGaussianPyr(combModel);
	auto lpCombModel = buildLaplPyr(gpCombModel);
	auto gpCombUser  = buildGaussianPyr(combUser);
	auto lpCombUser  = buildLaplPyr(gpCombUser);

#if TEST_DRAW //for draw
	cv::imshow("pImageModel", pImageModel);
	cv::imshow("pImageUser",  pImageUser);
#endif

	auto gpRectL = buildGaussianPyr(lFace, leftMask);
	auto gpRectR = buildGaussianPyr(rFace, rightMask);

	std::reverse(gpRectL.begin(), gpRectL.end());
	std::reverse(gpRectR.begin(), gpRectR.end());

	std::vector<cv::Mat> laplStitched;
	laplStitched.push_back(lpCombModel[0]);

	for (int i = 1; i < gpSrcUser.size(); i++)
	{
		cv::Rect rectFaceL = gpRectL[i].first;
		cv::Rect rectFaceR = gpRectR[i].first;

		cv::Mat maskFaceL = gpRectL[i].second;
		cv::Mat maskFaceR = gpRectR[i].second;
		maskFaceL *= 255;
		maskFaceR *= 255;

		cv::Mat leftFaceImg1 = cv::Mat(lpCombUser[i], rectFaceR); //model
		cv::Mat rightFaceImg1 = cv::Mat(lpCombModel[i], rectFaceL); //user
// 		cv::Mat leftFaceImg1 = cv::Mat(lpComb[i], rectFaceR); //model
// 		cv::Mat rightFaceImg1 = cv::Mat(lpSrc[i], rectFaceL); //user
// 		cv::imshow("leftFaceImg1", leftFaceImg1);
// 		cv::imshow("rightFaceImg1",  rightFaceImg1);

		cv::Mat combUn1 = lpSrcUser[i].clone();  //user
		cv::Mat combUn2 = lpSrcModel[i].clone(); //model
		cv::Mat element2;
		if (i == 1) element2 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
		if (i == 2) element2 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7));
		if (i == 3) element2 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(11, 11));
		if (i == 4) element2 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(13, 13));

		morphologyEx(maskFaceR, maskFaceR, cv::MORPH_ERODE, element2);
		morphologyEx(maskFaceL, maskFaceL, cv::MORPH_ERODE, element2);

		leftFaceImg1.copyTo(combUn1(rectFaceR), maskFaceR);
		rightFaceImg1.copyTo(combUn2(rectFaceL), maskFaceL);
// 		leftFaceImg1.copyTo(combUn2(rectFaceR), maskFaceR);
// 		rightFaceImg1.copyTo(combUn1(rectFaceL), maskFaceL);

		laplStitched.push_back(combUn2);
	}

	cv::Mat usualStitched = laplStitched[0].clone();
// 	cv::imshow("usualStitched", usualStitched);

	for (int i = 1; i <= 4; i++) {
		cv::Mat up;
		cv::pyrUp(usualStitched, up);
		usualStitched = up + laplStitched[i];

#if TEST_DRAW //for draw
		if (i == 2) cv::imshow("usualStitched2", usualStitched);
		if (i == 3) cv::imshow("usualStitched3", usualStitched);
		if (i == 4) cv::imshow("usualStitched4", usualStitched);
#endif
	}
	usualStitched.convertTo(usualStitched, CV_8UC3, 255);

	cv::resize(usualStitched, resultFrame, cv::Size(fullFrame.cols, fullFrame.rows));

// 	cv::imshow("resultFrame", resultFrame);

#else

	// Clone seamlessly.
	Rect r = boundingRect(hullModel);
	Mat img1WarpedSub = combModel(r);
	Mat img2Sub       = pImageModel(r);
	Mat maskSub       = maskModel(r);

	Point center(r.width/2, r.height/2);

	cv::imshow("img2Sub pre", img2Sub);

	Mat output;
	cv::seamlessClone(img1WarpedSub, img2Sub, maskSub, center, output, NORMAL_CLONE);
	output.copyTo(resultFrame(r));

	cv::imshow("img1WarpedSub", img1WarpedSub);
	cv::imshow("img2Sub aft", img2Sub);
	cv::imshow("maskSub", maskSub);
	cv::imshow("output", output);
#endif

	return resultFrame;
}

//////////////////////////////////
// Brief description: 
// public funcion to run full code
//////////////////////////////////
cv::Mat SwapFace::run(cv::Rect lFace, cv::Rect rFace) {
// 	auto faces = getFaces();
// 
// 	if (faces.size() != 2) 
// 		return resizedFrame;
// 	
// 	cv::Rect lFace = faces[0];
// 	cv::Rect rFace = faces[1];

	if (lFace.x > rFace.x) {
		swapFace(rFace, lFace);
	} else {
		swapFace(lFace, rFace);
	}

	return resultFrame;
}