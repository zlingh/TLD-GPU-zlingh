#include <opencv2/opencv.hpp>

class CV_EXPORTS PatchGenerator
{
public:
    PatchGenerator();
    PatchGenerator(double _backgroundMin, double _backgroundMax,
                   double _noiseRange, bool _randomBlur=true,
                   double _lambdaMin=0.6, double _lambdaMax=1.5,
                   double _thetaMin=-CV_PI, double _thetaMax=CV_PI,
                   double _phiMin=-CV_PI, double _phiMax=CV_PI );
    void operator()(const cv::Mat& image, cv::Point2f pt, cv::Mat& patch, cv::Size patchSize, cv::RNG& rng) const;
    void operator()(const cv::Mat& image, const cv::Mat& transform, cv::Mat& patch,
                    cv::Size patchSize, cv::RNG& rng) const;
    void warpWholeImage(const cv::Mat& image, cv::Mat& matT, cv::Mat& buf,
                        CV_OUT cv::Mat& warped, int border, cv::RNG& rng) const;
    void generateRandomTransform(cv::Point2f srcCenter, cv::Point2f dstCenter,
                                 CV_OUT cv::Mat& transform, cv::RNG& rng,
                                 bool inverse=false) const;
    void setAffineParam(double lambda, double theta, double phi);
    
    double backgroundMin, backgroundMax;
    double noiseRange;
    bool randomBlur;
    double lambdaMin, lambdaMax;
    double thetaMin, thetaMax;
    double phiMin, phiMax;
};
