/* SwapFace class declaration */

#include <opencv2\opencv.hpp>
#include "GlobalParams.h"
#include "FaceSwapper.h"

class SwapFace {
private:
	cv::Mat fullFrame;
	cv::Mat resizedFrame;
	cv::Mat resultFrame;

	std::vector<cv::Rect> getFaces();

	std::vector<cv::Mat> buildGaussianPyr(cv::Mat img, int pyrLevel);
	std::vector<std::pair<cv::Rect, cv::Mat>> buildGaussianPyr(cv::Rect rect, cv::Mat mask, int pyrLevel);
	std::vector<cv::Mat> buildLaplPyr(std::vector<cv::Mat> gaussPyr, int pyrLevel);
	
	cv::Mat segmentFace(cv::Mat src);
	cv::Mat findMask(cv::Mat face);
	cv::Mat getStatisticsMask(cv::Mat src);
	cv::Mat perform3SigmaRule(cv::Mat& src, cv::Mat mask);
	void clarifyBorders(cv::Mat face, cv::Mat& mask, int iterations);

	std::pair<cv::Mat, cv::Mat> copySrcToDstUsingMask(cv::Mat imgSrc, cv::Mat imgDst, cv::Mat maskSrc, cv::Mat maskDst);

	float getDist(cv::Point& p1, cv::Point& p2);
	cv::Point findClosesPoint(cv::Point& p, std::vector<cv::Point>& contour);
	cv::Mat stretchFace(cv::Mat imgSrc, cv::Mat imgDst, cv::Mat maskSrc, cv::Mat maskDst);

	std::pair<cv::Mat, cv::Mat> fitImagesToEachOther(cv::Mat leftImg, cv::Mat rightImg, cv::Mat leftMask, cv::Mat rightMask);

	void swapFace(cv::Rect lFace, cv::Rect rFace);

public:
	SwapFace(cv::Mat& inputFrame) 
		: fullFrame(inputFrame)
	    , resizedFrame(inputFrame)
	    , resultFrame(resizedFrame) {
		cv::resize(resizedFrame, resizedFrame, cv::Size(WIDTH, HEIGHT));
	};

	virtual ~SwapFace() {
		fullFrame.release();
		resizedFrame.release();
	};

	cv::Mat run(cv::Rect lFace, cv::Rect rFace);
	cv::Mat swapFace(cv::Mat &pImageModel, cv::Mat &pImageUser, SFaceInfo* pFaceInfoModel, SFaceInfo* pFaceInfoUser);
};
