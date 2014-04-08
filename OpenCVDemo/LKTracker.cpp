#include "LKTracker.h"
using namespace cv;

LKTracker::LKTracker(){
  term_criteria = TermCriteria( TermCriteria::COUNT+TermCriteria::EPS, 20, 0.03);
  window_size = Size(4,4);
  level = 5;
  lambda = 0.5;
}

void calcOpticalFlowPyrLKzp( InputArray _prevImg, InputArray _nextImg,
                           InputArray _prevPts, InputOutputArray _nextPts,
                           OutputArray _status, OutputArray _err,
                           Size winSize, int maxLevel,
                           TermCriteria criteria,
                           int flags, double minEigThreshold )
{
#ifdef HAVE_TEGRA_OPTIMIZATION
    if (tegra::calcOpticalFlowPyrLK(_prevImg, _nextImg, _prevPts, _nextPts, _status, _err, winSize, maxLevel, criteria, derivLambda, flags))
        return;
#endif
    Mat prevImg = _prevImg.getMat(), nextImg = _nextImg.getMat(), prevPtsMat = _prevPts.getMat();
//    derivLambda = std::min(std::max(derivLambda, 0.), 1.);
    const int derivDepth = 1;

   // CV_Assert( derivLambda >= 0 );
    CV_Assert( maxLevel >= 0 && winSize.width > 2 && winSize.height > 2 );
    CV_Assert( prevImg.size() == nextImg.size() &&
        prevImg.type() == nextImg.type() );

    int level=0, i, k, npoints, cn = prevImg.channels(), cn2 = cn*2;
    CV_Assert( (npoints = prevPtsMat.checkVector(2, CV_32F, true)) >= 0 );
    
    if( npoints == 0 )
    {
        _nextPts.release();
        _status.release();
        _err.release();
        return;
    }
    
    if( !(flags & OPTFLOW_USE_INITIAL_FLOW) )
        _nextPts.create(prevPtsMat.size(), prevPtsMat.type(), -1, true);
    
    Mat nextPtsMat = _nextPts.getMat();
    CV_Assert( nextPtsMat.checkVector(2, CV_32F, true) == npoints );
    
    const Point2f* prevPts = (const Point2f*)prevPtsMat.data;
    Point2f* nextPts = (Point2f*)nextPtsMat.data;
    
    _status.create((int)npoints, 1, CV_8U, -1, true);
    Mat statusMat = _status.getMat(), errMat;
    CV_Assert( statusMat.isContinuous() );
    uchar* status = statusMat.data;
    float* err = 0;
    
    for( i = 0; i < npoints; i++ )
        status[i] = true;
    
    if( _err.needed() )
    {
        _err.create((int)npoints, 1, CV_32F, -1, true);
        errMat = _err.getMat();
        CV_Assert( errMat.isContinuous() );
        err = (float*)errMat.data;
    }

    vector<Mat> prevPyr(maxLevel+1), nextPyr(maxLevel+1);
    
    // build the image pyramids.
    // we pad each level with +/-winSize.{width|height}
    // pixels to simplify the further patch extraction.
    // Thanks to the reference counting, "temp" mat (the pyramid layer + border)
    // will not be deallocated, since {prevPyr|nextPyr}[level] will be a ROI in "temp".
    for( k = 0; k < 2; k++ )
    {
        Size sz = prevImg.size();
        vector<Mat>& pyr = k == 0 ? prevPyr : nextPyr;
        Mat& img0 = k == 0 ? prevImg : nextImg;
        
        for( level = 0; level <= maxLevel; level++ )
        {
            Mat temp(sz.height + winSize.height*2,
                     sz.width + winSize.width*2,
                     img0.type());
            pyr[level] = temp(Rect(winSize.width, winSize.height, sz.width, sz.height));
            if( level == 0 )
                img0.copyTo(pyr[level]);
            else
                pyrDown(pyr[level-1], pyr[level], pyr[level].size());
            copyMakeBorder(pyr[level], temp, winSize.height, winSize.height,
                           winSize.width, winSize.width, BORDER_REFLECT_101);
            sz = Size((sz.width+1)/2, (sz.height+1)/2);
            if( sz.width <= winSize.width || sz.height <= winSize.height )
            {
                maxLevel = level;
                break;
            }
        }
    }
    // dI/dx ~ Ix, dI/dy ~ Iy
    Mat derivIBuf((prevImg.rows + winSize.height*2),
             (prevImg.cols + winSize.width*2),
             CV_MAKETYPE(derivDepth, cn2));

    if( (criteria.type & TermCriteria::COUNT) == 0 )
        criteria.maxCount = 30;
    else
        criteria.maxCount = std::min(std::max(criteria.maxCount, 0), 100);
    if( (criteria.type & TermCriteria::EPS) == 0 )
        criteria.epsilon = 0.01;
    else
        criteria.epsilon = std::min(std::max(criteria.epsilon, 0.), 10.);
    criteria.epsilon *= criteria.epsilon;

    for( level = maxLevel; level >= 0; level-- )
    {
        Size imgSize = prevPyr[level].size();
        Mat _derivI( imgSize.height + winSize.height*2,
            imgSize.width + winSize.width*2, derivIBuf.type(), derivIBuf.data );
        Mat derivI = _derivI(Rect(winSize.width, winSize.height, imgSize.width, imgSize.height));
      //  calcSharrDeriv(prevPyr[level], derivI);
        copyMakeBorder(derivI, _derivI, winSize.height, winSize.height, winSize.width, winSize.width, BORDER_CONSTANT);
        
        Mat I = prevPyr[level], J = nextPyr[level];
        
    //    parallel_for(BlockedRange(0, npoints), LKTrackerInvoker(prevPyr[level], derivI,
     //                                                           nextPyr[level], prevPts, nextPts,
    //                                                            status, err,
    //                                                            winSize, criteria, level, maxLevel,
    //                                                            flags, (float)minEigThreshold));
    }
}





bool LKTracker::trackf2f(const Mat& img1, const Mat& img2,vector<Point2f> &points1, vector<cv::Point2f> &points2){
  //TODO!:implement c function cvCalcOpticalFlowPyrLK() or Faster tracking function
  //Forward-Backward tracking
  calcOpticalFlowPyrLK( img1,img2, points1, points2, status,similarity, window_size, level, term_criteria, lambda, 0);
  calcOpticalFlowPyrLK( img2,img1, points2, pointsFB, FB_status,FB_error, window_size, level, term_criteria, lambda, 0);
  //Compute the real FB-error
  for( int i= 0; i<points1.size(); ++i ){
        FB_error[i] = norm(pointsFB[i]-points1[i]);
  }
  //Filter out points with FB_error[i] > median(FB_error) && points with sim_error[i] > median(sim_error)
  normCrossCorrelation(img1,img2,points1,points2);
  return filterPts(points1,points2);
}
void myNcc_CCORR_NORMEDlk(cv::Mat &src,const cv:: Mat &dst,float &ncc){	//这是归一化相关匹配法，method=CV_TM_CCORR_NORMED，自动判断是8U还是32F
	if(src.step[1]==1){//实际是指src.step.buf的两个值，第一个src.step[0]是矩阵的行宽（单位是字节），第二个是矩阵的数据类型的大小（字节单位）
		double srcCFH=0.0,dstCFH=0.0,nccSum=0.0;
		for (int i=0;i<src.rows;i++)
			for (int j=0;j<src.cols;j++)
			{
				srcCFH+=(src.at<uchar>(i,j))*(src.at<uchar>(i,j));
				dstCFH+=(dst.at<uchar>(i,j))*(dst.at<uchar>(i,j));
				nccSum+=(src.at<uchar>(i,j))*(dst.at<uchar>(i,j));
			}
			double CFH=sqrt((double)srcCFH)*sqrt((double)dstCFH);
			ncc=(double)nccSum/CFH;
	}
	else{

		double srcCFH=0.0,dstCFH=0.0,nccSum=0.0;
		for (int i=0;i<src.rows;i++)
			for (int j=0;j<src.cols;j++)
			{
				srcCFH+=(src.at<float>(i,j))*(src.at<float>(i,j));
				dstCFH+=(dst.at<float>(i,j))*(dst.at<float>(i,j));
				nccSum+=(src.at<float>(i,j))*(dst.at<float>(i,j));
			}
			double CFH=sqrt((double)srcCFH)*sqrt((double)dstCFH);
			ncc=(double)nccSum/CFH;
	}


}
void LKTracker::normCrossCorrelation(const Mat& img1,const Mat& img2, vector<Point2f>& points1, vector<Point2f>& points2) {
        Mat rec0(10,10,CV_8U);
        Mat rec1(10,10,CV_8U);
        Mat res(1,1,CV_32F);
		float myncc =0;
        for (int i = 0; i < points1.size(); i++) {
                if (status[i] == 1) {
                        getRectSubPix( img1, Size(10,10), points1[i],rec0 );
                        getRectSubPix( img2, Size(10,10), points2[i],rec1);
                        //matchTemplate( rec0,rec1, res, CV_TM_CCOEFF_NORMED);						
                       // similarity[i] = ((float *)(res.data))[0];
						myNcc_CCORR_NORMEDlk(rec0,rec1,myncc);
						similarity[i] = myncc;

                } else {
                        similarity[i] = 0.0;
                }
        }
        rec0.release();
        rec1.release();
        res.release();
}


bool LKTracker::filterPts(vector<Point2f>& points1,vector<Point2f>& points2){
  //Get Error Medians
  simmed = median(similarity);
  size_t i, k;
  for( i=k = 0; i<points2.size(); ++i ){
        if( !status[i])
          continue;
        if(similarity[i]> simmed){
          points1[k] = points1[i];
          points2[k] = points2[i];
          FB_error[k] = FB_error[i];
          k++;
        }
    }
  if (k==0)
    return false;
  points1.resize(k);
  points2.resize(k);
  FB_error.resize(k);

  fbmed = median(FB_error);
  for( i=k = 0; i<points2.size(); ++i ){
      if( !status[i])
        continue;
      if(FB_error[i] <= fbmed){
        points1[k] = points1[i];
        points2[k] = points2[i];
        k++;
      }
  }
  points1.resize(k);
  points2.resize(k);
  if (k>0)
    return true;
  else
    return false;
}




/*
 * old OpenCV style
void LKTracker::init(Mat img0, vector<Point2f> &points){
  //Preallocate
  //pyr1 = cvCreateImage(Size(img1.width+8,img1.height/3),IPL_DEPTH_32F,1);
  //pyr2 = cvCreateImage(Size(img1.width+8,img1.height/3),IPL_DEPTH_32F,1);
  //const int NUM_PTS = points.size();
  //status = new char[NUM_PTS];
  //track_error = new float[NUM_PTS];
  //FB_error = new float[NUM_PTS];
}


void LKTracker::trackf2f(..){
  cvCalcOpticalFlowPyrLK( &img1, &img2, pyr1, pyr1, points1, points2, points1.size(), window_size, level, status, track_error, term_criteria, CV_LKFLOW_INITIAL_GUESSES);
  cvCalcOpticalFlowPyrLK( &img2, &img1, pyr2, pyr1, points2, pointsFB, points2.size(),window_size, level, 0, 0, term_criteria, CV_LKFLOW_INITIAL_GUESSES | CV_LKFLOW_PYR_A_READY | CV_LKFLOW_PYR_B_READY );
}
*/

