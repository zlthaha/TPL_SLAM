
#pragma once

#include <iostream>
#include <queue>

#include "camodocal/camera_models/CameraFactory.h"
#include "camodocal/camera_models/CataCamera.h"
#include "camodocal/camera_models/PinholeCamera.h"
#include "camodocal/camera_models/EquidistantCamera.h"
#include "elsed/src/ELSED.h"// llg modify 添加elsed算法
#include "parameters.h"
#include "tic_toc.h"

// #include <opencv2/line_descriptor.hpp>
#include <opencv2/features2d.hpp>

#include "line_descriptor_custom.hpp"
#include <opencv2/opencv.hpp>// llg add for line flow的并行计算

#define _LAYER_ 4// llg add for line flow 
#define _POINTNUM_ 5

using namespace cv::line_descriptor;
using namespace std;
using namespace cv;
using namespace camodocal;

struct Line
{
	Point2f StartPt;
	Point2f EndPt;
	float lineWidth;
	Point2f Vp;

	Point2f Center;
	Point2f unitDir; // [cos(theta), sin(theta)]
	float length;
	float theta;

	// para_a * x + para_b * y + c = 0
	float para_a;
	float para_b;
	float para_c;

	float image_dx;
	float image_dy;
    float line_grad_avg;

	float xMin;
	float xMax;
	float yMin;
	float yMax;
	unsigned short id;
	int colorIdx;

    ///////////////// llg add for line flow
    Point2f MidPt;// 光流跟踪线上的几个分位点
    Point2f MidPt1;
    Point2f MidPt2;
    Point2f MidPt5;
    Point2f MidPt6;
    vector<Point2f> keyPoint;
    /////////////////
};

struct LineST
{
    Point2f start;
    Point2f end;
};
typedef vector<LineST> LineRecord;

class FrameLines
{
public:
    int frame_id;
    Mat img;

    vector<Mat> img_pyr;

    vector<Line> vecLine;
    vector< int > lineID;
    vector<Vec4f> ELSED;// llg add for 2D-3D
    vector<LineRecord> lineRec;
    vector<int> success;

    // opencv3 lsd+lbd
    std::vector<KeyLine> keylsd;
    Mat lbd_descr;
};
typedef shared_ptr< FrameLines > FrameLinesPtr;

class LineFeatureTracker
{
  public:
    LineFeatureTracker();

    void readIntrinsicParameter(const string &calib_file);
    void NearbyLineTracking(const vector<Line> forw_lines, const vector<Line> cur_lines, vector<pair<int, int> >& lineMatches);

    vector<Line> undistortedLineEndPoints();

    void readImage(const cv::Mat &_img);

    FrameLinesPtr curframe_, forwframe_;

    cv::Mat undist_map1_, undist_map2_ , K_;

    camodocal::CameraPtr m_camera;       // pinhole camera

    int frame_cnt;
    vector<int> ids;                     // 每个特征点的id
    vector<int> linetrack_cnt;           // 记录某个特征已经跟踪多少帧了，即被多少帧看到了
    int allfeature_cnt;                  // 用来统计整个地图中有了多少条线，它将用来赋值

    double sum_time;
    double mean_time;
};



////////////////////////////// llg add for line flow
//光流部分
//与原版不同取消了magnitude和angle 在我这个方法里应该不需要这个
class OpticalFlowTracker
{
public:
    OpticalFlowTracker(
        //const Mat &magnitude_,
        const Mat &img1_,
        const Mat &img2_,
        const vector<Line> &kp1_,
        vector<Line> &kp2_,
        vector<int> &success_,
        bool inverse_ = true, bool has_initial_ = false, int layer_ = 1) : img1(img1_), img2(img2_), kp1(kp1_), kp2(kp2_), success(success_), inverse(inverse_),
                                                                           has_initial(has_initial_), layer(layer_) {}

    // vector<vector<Point2f>> kp_2;

    void calculateOpticalFlow(const Range &range);
    void SVDcalculateOpticalFlow(const Range &range);

private:
    const Mat &img1;
    const Mat &img2;
    const vector<Line> &kp1;
    //const Mat &magnitude;
    vector<Line> &kp2;
    vector<int> &success;
    bool inverse = true;
    bool has_initial = false;
    int layer;
};



void OpticalFlowSingleLevel(
    //const Mat &magnitude,
    const Mat &img1,
    const Mat &img2,
    const vector<Line> &kp1,
    vector<Line> &kp2,
    vector<int> &success,
    bool inverse = false,
    bool has_initial_guess = false, int layer_ = 0);



void OpticalFlowMultiLevel(
    //const Mat &magnitude,
    //const Mat &angle,
    vector<Mat> &img1_pyr,
    vector<Mat> &img2_pyr,
    const vector<Line> &kp1,
    vector<Line> &kp2,
    vector<int> &success,
    bool inverse = false);

void LKmatch(
    vector<KeyLine> &line1,
    vector<KeyLine> &line2,
    vector<Line> &lk_line,
    vector<int> &succ,
    vector<DMatch> &result);

void getDistancePL(vector<Point2f> c, vector<KeyLine> l, vector<int> &id, vector<double> &value);