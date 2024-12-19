#include "linefeature_tracker.h"
#include <math.h>
// #include "line_descriptor/src/precomp_custom.hpp"

LineFeatureTracker::LineFeatureTracker()
{
    allfeature_cnt = 0;
    frame_cnt = 0;
    sum_time = 0.0;
}

void LineFeatureTracker::readIntrinsicParameter(const string &calib_file)
{
    ROS_INFO("reading paramerter of camera %s", calib_file.c_str());

    m_camera = CameraFactory::instance()->generateCameraFromYamlFile(calib_file);
    K_ = m_camera->initUndistortRectifyMap(undist_map1_,undist_map2_);    

}

vector<Line> LineFeatureTracker::undistortedLineEndPoints()
{
    vector<Line> un_lines;
    un_lines = curframe_->vecLine;
    float fx = K_.at<float>(0, 0);
    float fy = K_.at<float>(1, 1);
    float cx = K_.at<float>(0, 2);
    float cy = K_.at<float>(1, 2);
    //ROS_WARN_STREAM("NODE LINE " << curframe_->vecLine.size());
    for (unsigned int i = 0; i <curframe_->vecLine.size(); i++)
    {
        un_lines[i].StartPt.x = (curframe_->vecLine[i].StartPt.x - cx)/fx;
        un_lines[i].StartPt.y = (curframe_->vecLine[i].StartPt.y - cy)/fy;
        un_lines[i].EndPt.x = (curframe_->vecLine[i].EndPt.x - cx)/fx;
        un_lines[i].EndPt.y = (curframe_->vecLine[i].EndPt.y - cy)/fy;
    }
    return un_lines;
}

void LineFeatureTracker::NearbyLineTracking(const vector<Line> forw_lines, const vector<Line> cur_lines,
                                            vector<pair<int, int> > &lineMatches) {

    float th = 3.1415926/9;
    float dth = 30 * 30;
    for (size_t i = 0; i < forw_lines.size(); ++i) {
        Line lf = forw_lines.at(i);
        Line best_match;
        size_t best_j = 100000;
        size_t best_i = 100000;
        float grad_err_min_j = 100000;
        float grad_err_min_i = 100000;
        vector<Line> candidate;

        // 从 forw --> cur 查找
        for(size_t j = 0; j < cur_lines.size(); ++j) {
            Line lc = cur_lines.at(j);
            // condition 1 距离判断
            Point2f d = lf.Center - lc.Center;
            float dist = d.dot(d);
            if( dist > dth) continue;  //
            // condition 2 角度判断
            float delta_theta1 = fabs(lf.theta - lc.theta);
            float delta_theta2 = 3.1415926 - delta_theta1;
            if( delta_theta1 < th || delta_theta2 < th)// 距离和角度均满足
            {
                //std::cout << "theta: "<< lf.theta * 180 / 3.14259 <<" "<< lc.theta * 180 / 3.14259<<" "<<delta_theta1<<" "<<delta_theta2<<std::endl;
                candidate.push_back(lc);
                //float cost = fabs(lf.image_dx - lc.image_dx) + fabs( lf.image_dy - lc.image_dy) + 0.1 * dist;
                float cost = fabs(lf.line_grad_avg - lc.line_grad_avg) + dist/10.0;

                //std::cout<< "line match cost: "<< cost <<" "<< cost - sqrt( dist )<<" "<< sqrt( dist ) <<"\n\n";
                if(cost < grad_err_min_j)
                {
                    best_match = lc;
                    grad_err_min_j = cost;
                    best_j = j;
                }
            }

        }
        if(grad_err_min_j > 50) continue;  // 没找到

        //std::cout<< "!!!!!!!!! minimal cost: "<<grad_err_min_j <<"\n\n";

        // 如果 forw --> cur 找到了 best, 那我们反过来再验证下
        if(best_j < cur_lines.size())
        {
            // 反过来，从 cur --> forw 查找
            Line lc = cur_lines.at(best_j);
            for (int k = 0; k < forw_lines.size(); ++k)
            {
                Line lk = forw_lines.at(k);

                // condition 1
                Point2f d = lk.Center - lc.Center;
                float dist = d.dot(d);
                if( dist > dth) continue;  //
                // condition 2
                float delta_theta1 = fabs(lk.theta - lc.theta);
                float delta_theta2 = 3.1415926 - delta_theta1;
                if( delta_theta1 < th || delta_theta2 < th)
                {
                    //std::cout << "theta: "<< lf.theta * 180 / 3.14259 <<" "<< lc.theta * 180 / 3.14259<<" "<<delta_theta1<<" "<<delta_theta2<<std::endl;
                    //candidate.push_back(lk);
                    //float cost = fabs(lk.image_dx - lc.image_dx) + fabs( lk.image_dy - lc.image_dy) + dist;
                    float cost = fabs(lk.line_grad_avg - lc.line_grad_avg) + dist/10.0;

                    if(cost < grad_err_min_i)
                    {
                        grad_err_min_i = cost;
                        best_i = k;
                    }
                }

            }
        }

        if( grad_err_min_i < 50 && best_i == i){

            //std::cout<< "line match cost: "<<grad_err_min_j<<" "<<grad_err_min_i <<"\n\n";
            lineMatches.push_back(make_pair(best_j,i));
        }
        /*
        vector<Line> l;
        l.push_back(lf);
        vector<Line> best;
        best.push_back(best_match);
        visualizeLineTrackCandidate(l,forwframe_->img,"forwframe_");
        visualizeLineTrackCandidate(best,curframe_->img,"curframe_best");
        visualizeLineTrackCandidate(candidate,curframe_->img,"curframe_");
        cv::waitKey(0);
        */
    }

}

//#define NLT
#ifdef  NLT
void LineFeatureTracker::readImage(const cv::Mat &_img)
{
    cv::Mat img;
    TicToc t_p;
    frame_cnt++;
    cv::remap(_img, img, undist_map1_, undist_map2_, cv::INTER_LINEAR);
    //ROS_INFO("undistortImage costs: %fms", t_p.toc());
    if (EQUALIZE)   // 直方图均衡化
    {
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
        clahe->apply(img, img);
    }

    bool first_img = false;
    if (forwframe_ == nullptr) // 系统初始化的第一帧图像
    {
        forwframe_.reset(new FrameLines);
        curframe_.reset(new FrameLines);
        forwframe_->img = img;
        curframe_->img = img;
        first_img = true;
    }
    else
    {
        forwframe_.reset(new FrameLines);  // 初始化一个新的帧
        forwframe_->img = img;
    }

    // step 1: line extraction
    TicToc t_li;
    int lineMethod = 2;
    bool isROI = false;
    lineDetector ld(lineMethod, isROI, 0, (float)img.cols, 0, (float)img.rows);
    //ROS_INFO("ld inition costs: %fms", t_li.toc());
    TicToc t_ld;
    forwframe_->vecLine = ld.detect(img);

    for (size_t i = 0; i < forwframe_->vecLine.size(); ++i) {
        if(first_img)
            forwframe_->lineID.push_back(allfeature_cnt++);
        else
            forwframe_->lineID.push_back(-1);   // give a negative id
    }
    ROS_INFO("line detect costs: %fms", t_ld.toc());

    // step 3: junction & line matching
    if(curframe_->vecLine.size() > 0)
    {
        TicToc t_nlt;
        vector<pair<int, int> > linetracker;
        NearbyLineTracking(forwframe_->vecLine, curframe_->vecLine, linetracker);
        ROS_INFO("line match costs: %fms", t_nlt.toc());

        // 对新图像上的line赋予id值
        for(int j = 0; j < linetracker.size(); j++)
        {
            forwframe_->lineID[linetracker[j].second] = curframe_->lineID[linetracker[j].first];
        }

        // show NLT match
        //visualizeLineMatch(curframe_->vecLine, forwframe_->vecLine, linetracker,
                           curframe_->img, forwframe_->img, "NLT Line Matches", 10, true,
                           "frame");
        //visualizeLinewithID(forwframe_->vecLine,forwframe_->lineID,forwframe_->img,"forwframe_");
        //visualizeLinewithID(curframe_->vecLine,curframe_->lineID,curframe_->img,"curframe_");
        stringstream ss;
        ss <<"/home/hyj/datasets/line/" <<frame_cnt<<".jpg";
        // SaveFrameLinewithID(forwframe_->vecLine,forwframe_->lineID,forwframe_->img,ss.str().c_str());
        waitKey(5);


        vector<Line> vecLine_tracked, vecLine_new;
        vector< int > lineID_tracked, lineID_new;
        // 将跟踪的线和没跟踪上的线进行区分
        for (size_t i = 0; i < forwframe_->vecLine.size(); ++i)
        {
            if( forwframe_->lineID[i] == -1)
            {
                forwframe_->lineID[i] = allfeature_cnt++;
                vecLine_new.push_back(forwframe_->vecLine[i]);
                lineID_new.push_back(forwframe_->lineID[i]);
            }
            else
            {
                vecLine_tracked.push_back(forwframe_->vecLine[i]);
                lineID_tracked.push_back(forwframe_->lineID[i]);
            }
        }
        int diff_n = 30 - vecLine_tracked.size();  // 跟踪的线特征少于50了，那就补充新的线特征, 还差多少条线
        if( diff_n > 0)    // 补充线条
        {
            for (int k = 0; k < vecLine_new.size(); ++k) {
                vecLine_tracked.push_back(vecLine_new[k]);
                lineID_tracked.push_back(lineID_new[k]);
            }
        }

        forwframe_->vecLine = vecLine_tracked;
        forwframe_->lineID = lineID_tracked;

    }
    curframe_ = forwframe_;
}
#endif
int frame_num = 0;
#define MATCHES_DIST_THRESHOLD 30
void visualize_line_match(Mat imageMat1, Mat imageMat2,
                          std::vector<KeyLine> octave0_1, std::vector<KeyLine>octave0_2,
                          std::vector<DMatch> good_matches)
{
    //	Mat img_1;
    cv::Mat img1,img2;
    if (imageMat1.channels() != 3){
        cv::cvtColor(imageMat1, img1, cv::COLOR_GRAY2BGR);
    }
    else{
        img1 = imageMat1;
    }

    if (imageMat2.channels() != 3){
        cv::cvtColor(imageMat2, img2, cv::COLOR_GRAY2BGR);
    }
    else{
        img2 = imageMat2;
    }

    cv::Mat lsd_outImg;
    std::vector<char> lsd_mask( good_matches.size(), 1 );
    drawLineMatches( img1, octave0_1, img2, octave0_2, good_matches, lsd_outImg, Scalar::all( -1 ),Scalar::all( -1 ), lsd_mask,DrawLinesMatchesFlags::DEFAULT );
    //    srand(time(NULL));
    int lowest = 0, highest = 255;
    int range = (highest - lowest) + 1;
    for (int k = 0; k < good_matches.size(); ++k) {
        DMatch mt = good_matches[k];

        KeyLine line1 = octave0_1[mt.queryIdx];  // trainIdx
        KeyLine line2 = octave0_2[mt.trainIdx];  //queryIdx


        unsigned int r = lowest + int(rand() % range);
        unsigned int g = lowest + int(rand() % range);
        unsigned int b = lowest + int(rand() % range);
        cv::Point startPoint = cv::Point(int(line1.startPointX), int(line1.startPointY));
        cv::Point endPoint = cv::Point(int(line1.endPointX), int(line1.endPointY));
        cv::line(img1, startPoint, endPoint, cv::Scalar(r, g, b),2 ,8);

        cv::Point startPoint2 = cv::Point(int(line2.startPointX), int(line2.startPointY));
        cv::Point endPoint2 = cv::Point(int(line2.endPointX), int(line2.endPointY));
        cv::line(img2, startPoint2, endPoint2, cv::Scalar(r, g, b),2, 8);
        cv::line(img2, startPoint, startPoint2, cv::Scalar(0, 0, 255),1, 8);
        cv::line(img2, endPoint, endPoint2, cv::Scalar(0, 0, 255),1, 8);

    }
    /* plot matches */
    // cv::cvtColor(imageMat2, img2, cv::COLOR_GRAY2BGR);

    namedWindow("LSD matches", cv::WINDOW_NORMAL);
    imshow( "LSD matches", lsd_outImg );
    string name = to_string(frame_num);
    string path = "/home//llg/PL_VINS/LBD/";
    name = path + name + ".jpg";
    ofstream foutD("/home/llg/PL_VINS/lbd.txt", ofstream::app);
    foutD<< frame_num << "  " << good_matches.size() << endl;
    foutD.close();
    imwrite(name, lsd_outImg);
    
    // namedWindow("LSD matches1", cv::WINDOW_NORMAL);
    namedWindow("LSD matches2", cv::WINDOW_NORMAL);
    // imshow("LSD matches1", img1);
    imshow("LSD matches2", img2);
    waitKey(1);
}

/*////////////////////////////// 画不同方法的图对比用 不用的时候注释
void visualize_line_match2(Mat imageMat1, Mat imageMat2,
                          std::vector<KeyLine> octave0_1, std::vector<KeyLine>octave0_2,
                          std::vector<DMatch> good_matches)
{
    //	Mat img_1;
    cv::Mat img1,img2;
    if (imageMat1.channels() != 3){
        cv::cvtColor(imageMat1, img1, cv::COLOR_GRAY2BGR);
    }
    else{
        img1 = imageMat1;
    }

    if (imageMat2.channels() != 3){
        cv::cvtColor(imageMat2, img2, cv::COLOR_GRAY2BGR);
    }
    else{
        img2 = imageMat2;
    }

    cv::Mat lsd_outImg;
    std::vector<char> lsd_mask( good_matches.size(), 1 );
    drawLineMatches( img1, octave0_1, img2, octave0_2, good_matches, lsd_outImg, Scalar::all( -1 ),Scalar::all( -1 ), lsd_mask,DrawLinesMatchesFlags::DEFAULT );
    //    srand(time(NULL));
    int lowest = 0, highest = 255;
    int range = (highest - lowest) + 1;
    for (int k = 0; k < good_matches.size(); ++k) {
        DMatch mt = good_matches[k];

        KeyLine line1 = octave0_1[mt.queryIdx];  // trainIdx
        KeyLine line2 = octave0_2[mt.trainIdx];  //queryIdx


        unsigned int r = lowest + int(rand() % range);
        unsigned int g = lowest + int(rand() % range);
        unsigned int b = lowest + int(rand() % range);
        cv::Point startPoint = cv::Point(int(line1.startPointX), int(line1.startPointY));
        cv::Point endPoint = cv::Point(int(line1.endPointX), int(line1.endPointY));
        cv::line(img1, startPoint, endPoint, cv::Scalar(r, g, b),2 ,8);

        cv::Point startPoint2 = cv::Point(int(line2.startPointX), int(line2.startPointY));
        cv::Point endPoint2 = cv::Point(int(line2.endPointX), int(line2.endPointY));
        cv::line(img2, startPoint2, endPoint2, cv::Scalar(r, g, b),2, 8);
        cv::line(img2, startPoint, startPoint2, cv::Scalar(0, 0, 255),1, 8);
        cv::line(img2, endPoint, endPoint2, cv::Scalar(0, 0, 255),1, 8);

    }
    /* plot matches */
    /*// cv::cvtColor(imageMat2, img2, cv::COLOR_GRAY2BGR);

    namedWindow("LSD matches", cv::WINDOW_NORMAL);
    imshow( "LSD matches", lsd_outImg );
    string name = to_string(frame_num);
    string path = "/home//llg/PL_VINS/LF/";
    name = path + name + ".jpg";
    ofstream foutD("/home/llg/PL_VINS/lf.txt", ofstream::app);
    foutD<< frame_num  << "  " << good_matches.size() <<endl;
    frame_num ++;
    foutD.close();
    imwrite(name, lsd_outImg);
    
    // namedWindow("LSD matches1", cv::WINDOW_NORMAL);
    namedWindow("LSD matches2", cv::WINDOW_NORMAL);
    // imshow("LSD matches1", img1);
    imshow("LSD matches2", img2);
    waitKey(1);
}*/





//////////////////////////////







void visualize_line_match(Mat imageMat1, Mat imageMat2,
                          std::vector<KeyLine> octave0_1, std::vector<KeyLine>octave0_2,
                          std::vector<bool> good_matches)
{
    //	Mat img_1;
    cv::Mat img1,img2;
    if (imageMat1.channels() != 3){
        cv::cvtColor(imageMat1, img1, cv::COLOR_GRAY2BGR);
    }
    else{
        img1 = imageMat1;
    }

    if (imageMat2.channels() != 3){
        cv::cvtColor(imageMat2, img2, cv::COLOR_GRAY2BGR);
    }
    else{
        img2 = imageMat2;
    }

    //    srand(time(NULL));
    int lowest = 0, highest = 255;
    int range = (highest - lowest) + 1;
    for (int k = 0; k < good_matches.size(); ++k) {

        if(!good_matches[k]) continue;

        KeyLine line1 = octave0_1[k];  // trainIdx
        KeyLine line2 = octave0_2[k];  //queryIdx

        unsigned int r = lowest + int(rand() % range);
        unsigned int g = lowest + int(rand() % range);
        unsigned int b = lowest + int(rand() % range);
        cv::Point startPoint = cv::Point(int(line1.startPointX), int(line1.startPointY));
        cv::Point endPoint = cv::Point(int(line1.endPointX), int(line1.endPointY));
        cv::line(img1, startPoint, endPoint, cv::Scalar(r, g, b),2 ,8);

        cv::Point startPoint2 = cv::Point(int(line2.startPointX), int(line2.startPointY));
        cv::Point endPoint2 = cv::Point(int(line2.endPointX), int(line2.endPointY));
        cv::line(img2, startPoint2, endPoint2, cv::Scalar(r, g, b),2, 8);
        cv::line(img2, startPoint, startPoint2, cv::Scalar(0, 0, 255),1, 8);
        cv::line(img2, endPoint, endPoint2, cv::Scalar(0, 0, 255),1, 8);

    }
    /* plot matches */
    /*
    cv::Mat lsd_outImg;
    std::vector<char> lsd_mask( lsd_matches.size(), 1 );
    drawLineMatches( imageMat1, octave0_1, imageMat2, octave0_2, good_matches, lsd_outImg, Scalar::all( -1 ), Scalar::all( -1 ), lsd_mask,
    DrawLinesMatchesFlags::DEFAULT );

    imshow( "LSD matches", lsd_outImg );
    */
   namedWindow("LSD matches1", cv::WINDOW_NORMAL);
   namedWindow("LSD matches2", cv::WINDOW_NORMAL);
    imshow("LSD matches1", img1);
    imshow("LSD matches2", img2);
    waitKey(1);
}
void visualize_line(Mat imageMat1,std::vector<KeyLine> octave0_1)
{
    //	Mat img_1;
    cv::Mat img1;
    if (imageMat1.channels() != 3){
        cv::cvtColor(imageMat1, img1, cv::COLOR_GRAY2BGR);
    }
    else{
        img1 = imageMat1;
    }

    //    srand(time(NULL));
    int lowest = 0, highest = 255;
    int range = (highest - lowest) + 1;
    for (int k = 0; k < octave0_1.size(); ++k) {

        unsigned int r = 255; //lowest + int(rand() % range);
        unsigned int g = 255; //lowest + int(rand() % range);
        unsigned int b = 0;  //lowest + int(rand() % range);
        cv::Point startPoint = cv::Point(int(octave0_1[k].startPointX), int(octave0_1[k].startPointY));
        cv::Point endPoint = cv::Point(int(octave0_1[k].endPointX), int(octave0_1[k].endPointY));
        cv::line(img1, startPoint, endPoint, cv::Scalar(r, g, b),2 ,8);
        // cv::circle(img1, startPoint, 2, cv::Scalar(255, 0, 0), 5);
        // cv::circle(img1, endPoint, 2, cv::Scalar(0, 255, 0), 5);


    }
    string name = to_string(frame_num);
    string path = "/home//llg/PL_VINS/LSD/";
    name = path + name + ".jpg";
    imwrite(name, img1);

    ofstream foutD("/home/llg/PL_VINS/LSD.txt", ofstream::app);
    foutD<< frame_num << "  " << octave0_1.size() << endl;
    foutD.close();
    /* plot matches */
    /*
    cv::Mat lsd_outImg;
    std::vector<char> lsd_mask( lsd_matches.size(), 1 );
    drawLineMatches( imageMat1, octave0_1, imageMat2, octave0_2, good_matches, lsd_outImg, Scalar::all( -1 ), Scalar::all( -1 ), lsd_mask,
    DrawLinesMatchesFlags::DEFAULT );

    imshow( "LSD matches", lsd_outImg );
    */
    //namedWindow("LSD_C", cv::WINDOW_NORMAL);
    //imshow("LSD_C", img1);
    //waitKey(1);
}

/////////////////////////////////// 画线特征提取对比
void visualize_line2(Mat imageMat1,std::vector<KeyLine> octave0_1)
{
    //	Mat img_1;
    cv::Mat img1;
    if (imageMat1.channels() != 3){
        cv::cvtColor(imageMat1, img1, cv::COLOR_GRAY2BGR);
    }
    else{
        img1 = imageMat1;
    }

    //    srand(time(NULL));
    int lowest = 0, highest = 255;
    int range = (highest - lowest) + 1;
    for (int k = 0; k < octave0_1.size(); ++k) {

        unsigned int r = 255; //lowest + int(rand() % range);
        unsigned int g = 255; //lowest + int(rand() % range);
        unsigned int b = 0;  //lowest + int(rand() % range);
        cv::Point startPoint = cv::Point(int(octave0_1[k].startPointX), int(octave0_1[k].startPointY));
        cv::Point endPoint = cv::Point(int(octave0_1[k].endPointX), int(octave0_1[k].endPointY));
        cv::line(img1, startPoint, endPoint, cv::Scalar(r, g, b),2 ,8);
        // cv::circle(img1, startPoint, 2, cv::Scalar(255, 0, 0), 5);
        // cv::circle(img1, endPoint, 2, cv::Scalar(0, 255, 0), 5);


    }
    string name = to_string(frame_num);
    string path = "/home//llg/PL_VINS/ELSED/";
    name = path + name + ".jpg";
    imwrite(name, img1);

    ofstream foutD("/home/llg/PL_VINS/ELSED.txt", ofstream::app);
    foutD<< frame_num << "  " << octave0_1.size() << endl;
    foutD.close();
    /* plot matches */
    /*
    cv::Mat lsd_outImg;
    std::vector<char> lsd_mask( lsd_matches.size(), 1 );
    drawLineMatches( imageMat1, octave0_1, imageMat2, octave0_2, good_matches, lsd_outImg, Scalar::all( -1 ), Scalar::all( -1 ), lsd_mask,
    DrawLinesMatchesFlags::DEFAULT );

    imshow( "LSD matches", lsd_outImg );
    */
    //namedWindow("LSD_C", cv::WINDOW_NORMAL);
    //imshow("LSD_C", img1);
    //waitKey(1);
}





cv::Mat last_unsuccess_image;
vector< KeyLine > last_unsuccess_keylsd;
vector< int >  last_unsuccess_id;
Mat last_unsuccess_lbd_descr;
void LineFeatureTracker::readImage(const cv::Mat &_img)
{
    cv::Mat img;
    TicToc t_p;
    frame_cnt++;
    
    cv::remap(_img, img, undist_map1_, undist_map2_, cv::INTER_LINEAR);



//    cv::imshow("lineimg",img);
//    cv::waitKey(1);
    //ROS_INFO("undistortImage costs: %fms", t_p.toc());
    
    if (EQUALIZE)   // 直方图均衡化
    {
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
        clahe->apply(img, img);
    }
    

    bool first_img = false;
    if (forwframe_ == nullptr) // 系统初始化的第一帧图像
    {
        forwframe_.reset(new FrameLines);
        curframe_.reset(new FrameLines);
        forwframe_->img = img;
        curframe_->img = img;
        first_img = true;
        ////////////////// llg add for line flow
        forwframe_->img_pyr.push_back(img);
        curframe_->img_pyr.push_back(img);
    }
    else
    {
        forwframe_.reset(new FrameLines);  // 初始化一个新的帧
        forwframe_->img = img;// forw是追踪的帧
        forwframe_->img_pyr.push_back(img);// llg add for line flow
    }
    TicToc t_li;
    Ptr<line_descriptor::LSDDetectorC> lsd_ = line_descriptor::LSDDetectorC::createLSDDetectorC();
    // lsd parameters
    line_descriptor::LSDDetectorC::LSDOptions opts;
    opts.refine       = 1;     //1     	The way found lines will be refined
    opts.scale        = 0.5;   //0.8   	The scale of the image that will be used to find the lines. Range (0..1].
    opts.sigma_scale  = 0.6;	//0.6  	Sigma for Gaussian filter. It is computed as sigma = _sigma_scale/_scale.
    opts.quant        = 2.0;	//2.0   Bound to the quantization error on the gradient norm
    opts.ang_th       = 22.5;	//22.5	Gradient angle tolerance in degrees
    opts.log_eps      = 1.0;	//0		Detection threshold: -log10(NFA) > log_eps. Used only when advance refinement is chosen
    opts.density_th   = 0.6;	//0.7	Minimal density of aligned region points in the enclosing rectangle.
    opts.n_bins       = 1024;	//1024 	Number of bins in pseudo-ordering of gradient modulus.
    double min_line_length = 0.125;  // Line segments shorter than that are rejected
    // opts.refine       = 1;
    // opts.scale        = 0.5;
    // opts.sigma_scale  = 0.6;
    // opts.quant        = 2.0;
    // opts.ang_th       = 22.5;
    // opts.log_eps      = 1.0;
    // opts.density_th   = 0.6;
    // opts.n_bins       = 1024;
    // double min_line_length = 0.125;
    opts.min_length   = min_line_length*(std::min(img.cols,img.rows));

    std::vector<KeyLine> lsd, keylsd;
    //std::vector<KeyLine> lsd2;// llg add 对比不同线提取方法 不画图就没用处
    
	//void LSDDetectorC::detect( const std::vector<Mat>& images, std::vector<std::vector<KeyLine> >& keylines, int scale, int numOctaves, const std::vector<Mat>& masks ) const
    lsd_->detect( img, lsd, 2, 1, opts);// llg modify 注释原本LSD提取

    /*//////////////////////// llg modify 定义elsed
    upm::ELSEDParams elsed_params;// elsed的默认参数，可以修改
    upm::ELSED elsed(elsed_params);
    /////////////////////////
    vector<Vec4f> e_lines;
    e_lines = elsed.detect(img);// llg modify elsed  */

    /*ofstream foutC1("/home/llg/PL_VINS/output/line_detect.csv", ios::app);// 线特征提取用时
    foutC1 << t_li.toc() << endl;
    foutC1.close();*/


    /*//////////////////////// 把vec4f转成keyline
    int class_counter = -1;
    for(int i = 0; i < (int) e_lines.size(); i++)
    {
        KeyLine kl;
        Vec4f e_line = e_lines[i];
        kl.startPointX = e_line[0];// 这个是和图像金字塔缩放层数有关系的
        kl.startPointY = e_line[1];
        kl.endPointX = e_line[2];
        kl.endPointY = e_line[3];
        kl.sPointInOctaveX = e_line[0];
        kl.sPointInOctaveY = e_line[1];
        kl.ePointInOctaveX = e_line[2];
        kl.ePointInOctaveY = e_line[3];
        kl.lineLength = (float) sqrt( pow( e_line[0] - e_line[2], 2 ) + pow( e_line[1] - e_line[3], 2 ) );
        // compute number of pixels covered by line 
        LineIterator li( img, Point2f( e_line[0], e_line[1] ), Point2f( e_line[2], e_line[3] ) );
        kl.numOfPixels = li.count;

        kl.angle = atan2( ( kl.endPointY - kl.startPointY ), ( kl.endPointX - kl.startPointX ) );
        kl.class_id = ++class_counter;
        kl.octave = 0;
        kl.size = ( kl.endPointX - kl.startPointX ) * ( kl.endPointY - kl.startPointY );
        kl.response = kl.lineLength / max( img.cols, img.rows );
        kl.pt = Point2f( ( kl.endPointX + kl.startPointX ) / 2, ( kl.endPointY + kl.startPointY ) / 2 );

        lsd.push_back( kl );
    }
    /////////////////////////////*/


    //visualize_line(img,lsd2);
    //visualize_line2(img, lsd);// llg add 画不同线特征提取对比图

    // step 1: line extraction
    // TicToc t_li;
    // std::vector<KeyLine> lsd, keylsd;
    // Ptr<LSDDetector> lsd_;
    // lsd_ = cv::line_descriptor::LSDDetector::createLSDDetector();
    // lsd_->detect( img, lsd, 2, 2 );


    sum_time += t_li.toc();
   ROS_INFO("line detect costs: %fms", t_li.toc());

    Mat lbd_descr, keylbd_descr;
    // step 2: lbd descriptor
    // llg modify 这里计算LBD线描述符，若用光流这里不计算
    TicToc t_lbd;
    Ptr<BinaryDescriptor> bd_ = BinaryDescriptor::createBinaryDescriptor(  );// llg modify 注释LBD
    

    bd_->compute( img, lsd, lbd_descr );// llg modify 注释LBD

    /*ofstream foutC2("/home/llg/PL_VINS/output/line_descriptor.csv", ios::app);// 线特征描述用时
    foutC2 << t_lbd.toc() << endl;
    foutC2.close();*/

    // std::cout<<"lbd_descr = "<<lbd_descr.size()<<std::endl;
//////////////////////////
    for ( int i = 0; i < (int) lsd.size(); i++ )// keylsd是lsd的线长度进行了一个筛选  后续全是基于keylsd的匹配等
    {
        if( lsd[i].octave == 0 && lsd[i].lineLength >= 60)
        {
            keylsd.push_back( lsd[i] );
            keylbd_descr.push_back( lbd_descr.row( i ) );// 从这里看，lbd_descr这个矩阵里每行是一个线的描述符
        }
    }
    // std::cout<<"lbd_descr = "<<lbd_descr.size()<<std::endl;
//    ROS_INFO("lbd_descr detect costs: %fms", keylsd.size() * t_lbd.toc() / lsd.size() );
    sum_time += keylsd.size() * t_lbd.toc() / lsd.size();// 线特征提取+描述总时间 llg add
///////////////

    forwframe_->keylsd = keylsd;
    forwframe_->lbd_descr = keylbd_descr;

    for (size_t i = 0; i < forwframe_->keylsd.size(); ++i) {
        if(first_img)
            forwframe_->lineID.push_back(allfeature_cnt++);
        else
            forwframe_->lineID.push_back(-1);   // give a negative id
    }
    //ROS_WARN_STREAM("NUM OF FORW  "<< forwframe_->lineID.size());
    // if(!first_img)
    // {
    //     std::vector<DMatch> lsd_matches;
    //     Ptr<BinaryDescriptorMatcher> bdm_;
    //     bdm_ = BinaryDescriptorMatcher::createBinaryDescriptorMatcher();
    //     bdm_->match(forwframe_->lbd_descr, curframe_->lbd_descr, lsd_matches);
    //     visualize_line_match(forwframe_->img.clone(), curframe_->img.clone(), forwframe_->keylsd, curframe_->keylsd, lsd_matches);
    //     // std::cout<<"lsd_matches = "<<lsd_matches.size()<<" forwframe_->keylsd = "<<keylbd_descr.size()<<" curframe_->keylsd = "<<keylbd_descr.size()<<std::endl;
    // }


    if(curframe_->keylsd.size() > 0)// 遍历lsd 进行匹配 llg add 
    {
        /////////////////////// llg add keylsd转换成Line进行线流跟踪
        vector<Line> cur_lines;
        /*for(size_t i = 0; i < forwframe_->keylsd.size(); ++i)// 这个好像不需要  下一帧的线不需要转
        {
            Line forw;
            forw.StartPt = forwframe_->keylsd[i].getStartPoint();
            forw.EndPt = forwframe_->keylsd[i].getEndPoint();
            forw.MidPt = 0.5 * (forw.StartPt + forw.EndPt);
            forw.length = sqrt((forw.StartPt.x - forw.EndPt.x) * (forw.StartPt.x - forw.EndPt.x) + (forw.StartPt.y - forw.EndPt.y) * (forw.StartPt.y - forw.EndPt.y));
            forw.MidPt1 = 0.5 * Point2f(forw.StartPt.x + forw.MidPt.x, forw.StartPt.y + forw.MidPt.y);
            forw.MidPt2 = 0.5 * Point2f(forw.EndPt.x + forw.MidPt.x, forw.EndPt.y + forw.MidPt.y);
            forw.MidPt5 = 0.7 * forw.StartPt + 0.3 * forw.MidPt1;
            forw.MidPt6 = 0.7 * forw.EndPt + 0.3 * forw.MidPt2;
            forw.keyPoint.push_back(forw.MidPt);
            forw.keyPoint.push_back(forw.MidPt5);
            forw.keyPoint.push_back(forw.MidPt1);
            forw.keyPoint.push_back(forw.MidPt2);
            forw.keyPoint.push_back(forw.MidPt6);
            forw_lines.push_back(forw);
        }*/
        /*for(size_t i = 0; i < curframe_->keylsd.size(); ++i)
        {
            Line cur;
            cur.StartPt = curframe_->keylsd[i].getStartPoint();
            cur.EndPt = curframe_->keylsd[i].getEndPoint();
            cur.MidPt = 0.5 * (cur.StartPt + cur.EndPt);
            cur.length = sqrt((cur.StartPt.x - cur.EndPt.x) * (cur.StartPt.x - cur.EndPt.x) + (cur.StartPt.y - cur.EndPt.y) * (cur.StartPt.y - cur.EndPt.y));
            cur.MidPt1 = 0.5 * Point2f(cur.StartPt.x + cur.MidPt.x, cur.StartPt.y + cur.MidPt.y);
            cur.MidPt2 = 0.5 * Point2f(cur.EndPt.x + cur.MidPt.x, cur.EndPt.y + cur.MidPt.y);
            cur.MidPt5 = 0.7 * cur.StartPt + 0.3 * cur.MidPt1;
            cur.MidPt6 = 0.7 * cur.EndPt + 0.3 * cur.MidPt2;
            cur.keyPoint.push_back(cur.MidPt);
            cur.keyPoint.push_back(cur.MidPt5);
            cur.keyPoint.push_back(cur.MidPt1);
            cur.keyPoint.push_back(cur.MidPt2);
            cur.keyPoint.push_back(cur.MidPt6);
            cur_lines.push_back(cur);
        }
        ///////////////////////*/

        /* compute matches */
        TicToc t_match;// 匹配线用时 llg add
        std::vector<DMatch> lsd_matches;
        //std::vector<DMatch> lsd_matches2;// llg add 画出不同方法跟踪示例图  不画的时候注释
        Ptr<BinaryDescriptorMatcher> bdm_;
        bdm_ = BinaryDescriptorMatcher::createBinaryDescriptorMatcher();
        bdm_->match(forwframe_->lbd_descr, curframe_->lbd_descr, lsd_matches);// llg modify 这里可更改为线光流匹配 这部分的上下相关注释，然后把光流跟踪结果存进lsd_matches*/
//        ROS_INFO("lbd_macht costs: %fms", t_match.toc());

        
        //////////////////////llg add for line optical flow
        /*vector<Line> lf_lines;// 跟踪到的线的keypoint和上面的forw_lines判断匹配
        OpticalFlowMultiLevel(curframe_->img_pyr, forwframe_->img_pyr, cur_lines, lf_lines, curframe_->success);
        LKmatch(curframe_->keylsd, forwframe_->keylsd, lf_lines, curframe_->success, lsd_matches);
        //ROS_WARN_STREAM("tracked line num = "<< lsd_matches.size());
        //////////////////////*/

        /* select best matches */
        std::vector<DMatch> good_matches;
        //std::vector<DMatch> good_matches2;
        std::vector<KeyLine> good_Keylines;
        good_matches.clear();
        for ( int i = 0; i < (int) lsd_matches.size(); i++ )
        {
            if( lsd_matches[i].distance < 30 ){// llg modify 最优与次优比例限制 未完成.......

                DMatch mt = lsd_matches[i];
                KeyLine line1 =  forwframe_->keylsd[mt.queryIdx] ;// query是向前跟踪待匹配的线
                KeyLine line2 =  curframe_->keylsd[mt.trainIdx] ;
                Point2f serr = line1.getStartPoint() - line2.getStartPoint();// 
                Point2f eerr = line1.getEndPoint() - line2.getEndPoint();
                // std::cout<<"11111111111111111 = "<<abs(line1.angle-line2.angle)<<std::endl;
                if((serr.dot(serr) < 60 * 60) && (eerr.dot(eerr) < 60 * 60)&&abs(line1.angle-line2.angle)<0.1)   // 线段在图像里不会跑得特别远，用向量点乘做的判断
                    good_matches.push_back( lsd_matches[i] );// 这个good_matches给下面的id赋值，good才会认为是跟踪到的tracked，否则是new线
            }
        }

        //////////////////////////
        /*ofstream foutC3("/home/llg/PL_VINS/output/line_track.csv", ios::app);// 线特征匹配用时
        foutC3 << t_match.toc() << endl;
        foutC3.close();*/
        //////////////////////////

        /*good_matches2.clear();
        for ( int i = 0; i < (int) lsd_matches2.size(); i++ )
        {
            if( lsd_matches2[i].distance < 30 ){// llg modify 最优与次优比例限制 未完成.......

                DMatch mt = lsd_matches2[i];
                KeyLine line1 =  forwframe_->keylsd[mt.queryIdx] ;// query是向前跟踪待匹配的线
                KeyLine line2 =  curframe_->keylsd[mt.trainIdx] ;
                Point2f serr = line1.getStartPoint() - line2.getStartPoint();// 
                Point2f eerr = line1.getEndPoint() - line2.getEndPoint();
                // std::cout<<"11111111111111111 = "<<abs(line1.angle-line2.angle)<<std::endl;
                if((serr.dot(serr) < 60 * 60) && (eerr.dot(eerr) < 60 * 60)&&abs(line1.angle-line2.angle)<0.1)   // 线段在图像里不会跑得特别远，用向量点乘做的判断
                    good_matches2.push_back( lsd_matches2[i] );// 这个good_matches给下面的id赋值，good才会认为是跟踪到的tracked，否则是new线
            }
        }*/

        //ROS_WARN_STREAM("good line num = "<< good_matches.size());
        //ROS_WARN_STREAM("lsd MATCH "<< lsd_matches.size());

        // for(int k = 0; k < curframe_->lineID.size(); k++)
        // {
        //     ROS_WARN_STREAM("cur id  "<< curframe_->lineID[k]);
        // }

        ///////////////// llg add 下面是光流跟踪不需要改的地方，存成good_match后直接进行下面即可

        vector< int > success_id;
        // std::cout << forwframe_->lineID.size() <<" " <<curframe_->lineID.size();
        for (int k = 0; k < good_matches.size(); ++k) {
            DMatch mt = good_matches[k];
            forwframe_->lineID[mt.queryIdx] = curframe_->lineID[mt.trainIdx];
            success_id.push_back(curframe_->lineID[mt.trainIdx]);
            
        }

        /////////////////////// llg modify 只保留good的line
        // curframe_->lineID = success_id;
        // curframe_->vecLine = success_line;
        ///////////////////////



        //visualize_line_match(forwframe_->img.clone(), curframe_->img.clone(), forwframe_->keylsd, curframe_->keylsd, good_matches2);// 可视化elsed线特征LBD匹配情况
        //visualize_line_match2(forwframe_->img.clone(), curframe_->img.clone(), forwframe_->keylsd, curframe_->keylsd, good_matches);// ELSED线特征的光流跟踪

        //把没追踪到的线存起来

        vector<KeyLine> vecLine_tracked, vecLine_new;
        vector< int > lineID_tracked, lineID_new;
        Mat DEscr_tracked, Descr_new;
        // 将跟踪的线和没跟踪上的线进行区分
        for (size_t i = 0; i < forwframe_->keylsd.size(); ++i)
        {
            
            if( forwframe_->lineID[i] == -1)
            {
                forwframe_->lineID[i] = allfeature_cnt++;// 新线赋值ID
                vecLine_new.push_back(forwframe_->keylsd[i]);
                lineID_new.push_back(forwframe_->lineID[i]);
                Descr_new.push_back( forwframe_->lbd_descr.row( i ) );
            }
            
            else
            {
                
                vecLine_tracked.push_back(forwframe_->keylsd[i]);//tracked到的线继承了ID
                lineID_tracked.push_back(forwframe_->lineID[i]);
                DEscr_tracked.push_back( forwframe_->lbd_descr.row( i ) );

            }
        }
        // for(int k = 0; k < forwframe_->lineID.size(); k++)
        // {
        //     ROS_WARN_STREAM("forw id  "<< forwframe_->lineID[k]);
        // }
        // ROS_WARN_STREAM("good  "<< good_matches.size());
        // ROS_WARN_STREAM("tracked  "<< lineID_tracked.size());
        // ROS_WARN_STREAM("new  "<< lineID_new.size());

        // 对新提取的线分类水平线、垂直线
        vector<KeyLine> h_Line_new, v_Line_new;
        vector< int > h_lineID_new,v_lineID_new;
        Mat h_Descr_new,v_Descr_new;
        for (size_t i = 0; i < vecLine_new.size(); ++i)
        {
            if((((vecLine_new[i].angle >= 3.14/4 && vecLine_new[i].angle <= 3*3.14/4))||(vecLine_new[i].angle <= -3.14/4 && vecLine_new[i].angle >= -3*3.14/4)))
            {
                h_Line_new.push_back(vecLine_new[i]);
                h_lineID_new.push_back(lineID_new[i]);
                h_Descr_new.push_back(Descr_new.row( i ));
            }
            else
            {
                v_Line_new.push_back(vecLine_new[i]);
                v_lineID_new.push_back(lineID_new[i]);
                v_Descr_new.push_back(Descr_new.row( i ));
            }      
        }

        // 已跟踪线进行水平和垂直线分类
        int h_line,v_line;
        h_line = v_line =0;
        for (size_t i = 0; i < vecLine_tracked.size(); ++i)
        {
            if((((vecLine_tracked[i].angle >= 3.14/4 && vecLine_tracked[i].angle <= 3*3.14/4))||(vecLine_tracked[i].angle <= -3.14/4 && vecLine_tracked[i].angle >= -3*3.14/4)))
            {
                h_line ++;
            }
            else
            {
                v_line ++;
            }
        }
        int diff_h = 35 - h_line;
        int diff_v = 35 - v_line;

        // std::cout<<"h_line = "<<h_line<<" v_line = "<<v_line<<std::endl;
        if( diff_h > 0)    // 补充线条
        {
            int kkk = 1;
            if(diff_h > h_Line_new.size())
                diff_h = h_Line_new.size();
            else 
                kkk = int(h_Line_new.size()/diff_h);
            for (int k = 0; k < diff_h; ++k) 
            {
                vecLine_tracked.push_back(h_Line_new[k]);
                lineID_tracked.push_back(h_lineID_new[k]);
                DEscr_tracked.push_back(h_Descr_new.row(k));
            }
            // std::cout  <<"h_kkk = " <<kkk<<" diff_h = "<<diff_h<<" h_Line_new.size() = "<<h_Line_new.size()<<std::endl;
        }
        if( diff_v > 0)    // 补充线条
        {
            int kkk = 1;
            if(diff_v > v_Line_new.size())
                diff_v = v_Line_new.size();
            else 
                kkk = int(v_Line_new.size()/diff_v);
            for (int k = 0; k < diff_v; ++k)  
            {
                vecLine_tracked.push_back(v_Line_new[k]);
                lineID_tracked.push_back(v_lineID_new[k]);
                DEscr_tracked.push_back(v_Descr_new.row(k));
            }            // std::cout  <<"v_kkk = " <<kkk<<" diff_v = "<<diff_v<<" v_Line_new.size() = "<<v_Line_new.size()<<std::endl;
        }
        // int diff_n = 50 - vecLine_tracked.size();  // 跟踪的线特征少于50了，那就补充新的线特征, 还差多少条线
        // if( diff_n > 0)    // 补充线条
        // {
        //     for (int k = 0; k < vecLine_new.size(); ++k) {
        //         vecLine_tracked.push_back(vecLine_new[k]);
        //         lineID_tracked.push_back(lineID_new[k]);
        //         DEscr_tracked.push_back(Descr_new.row(k));
        //     }
        // }
        
        forwframe_->keylsd = vecLine_tracked;
        forwframe_->lineID = lineID_tracked;
        forwframe_->lbd_descr = DEscr_tracked;

    }

    // 将opencv的KeyLine数据转为季哥的Line
    for (int j = 0; j < forwframe_->keylsd.size(); ++j) {
        Line l;
        KeyLine lsd = forwframe_->keylsd[j];
        l.StartPt = lsd.getStartPoint();
        l.EndPt = lsd.getEndPoint();
        l.length = lsd.lineLength;
        forwframe_->vecLine.push_back(l);
    }
    curframe_ = forwframe_;
    //ROS_WARN_STREAM("final line " << curframe_->vecLine.size());


}




/////////////////////// llg add for line flow
vector<vector<double>> param;

inline float GetPixelValue(const cv::Mat &img, float x, float y)// line flow 中用到
{
    // boundary check
    if (x < 0)
        x = 0;
    if (y < 0)
        y = 0;
    if (x >= img.cols - 1)
        x = img.cols - 2;
    if (y >= img.rows - 1)
        y = img.rows - 2;

    float xx = x - floor(x);
    float yy = y - floor(y);
    int x_a1 = std::min(img.cols - 1, int(x) + 1);
    int y_a1 = std::min(img.rows - 1, int(y) + 1);

    return (1 - xx) * (1 - yy) * img.at<uchar>(y, x) + xx * (1 - yy) * img.at<uchar>(y, x_a1) + (1 - xx) * yy * img.at<uchar>(y_a1, x) + xx * yy * img.at<uchar>(y_a1, x_a1);
}


void OpticalFlowMultiLevel(
    vector<Mat> &img1_pyr,
    vector<Mat> &img2_pyr,
    const vector<Line> &kp1,
    vector<Line> &kp2,
    vector<int> &success,
    bool inverse)
{
    // parameters
    int pyramids = _LAYER_;
    double pyramid_scale = 0.5;
    double scales[] = {1.0, 0.5, 0.25, 0.125};// llg add 在建立0.5倍缩放关系的图像金字塔
    vector<int> successd{0};
    param.resize(kp1.size());// 存储的待估计的线流三个参数G  这里先给line的size  后面再给每个定义成3

    // create pyramids
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    if (img1_pyr.size() == 1)
        for (int i = 1; i < pyramids; i++)
        {

            Mat img1;
            cv::resize(img1_pyr[i - 1], img1,
                       cv::Size(img1_pyr[i - 1].cols * pyramid_scale, img1_pyr[i - 1].rows * pyramid_scale));
            img1_pyr.push_back(img1);
        }
    if (img2_pyr.size() == 1)
        for (int i = 1; i < pyramids; i++)
        {
            Mat img2;
            cv::resize(img2_pyr[i - 1], img2,
                       cv::Size(img2_pyr[i - 1].cols * pyramid_scale, img2_pyr[i - 1].rows * pyramid_scale));
            img2_pyr.push_back(img2);
        }
    if (img1_pyr.size() <= 1 || img2_pyr.size() <= 1)
        ROS_WARN("error img_pyr imput\n");
    // coarse-to-fine LK tracking in pyramids
    vector<Line> kp1_pyr, kp2_pyr;// 把线上采样的几个点在各层金字塔位置确定下来，kp2_py2是待跟踪的新图像
    for (auto &kp : kp1)
    {
        auto kp_top = kp;
        for (int m = 0; m < kp_top.keyPoint.size(); m++)
        {
            kp_top.keyPoint[m] *= scales[pyramids - 1];// 这里是最粗略的金字塔的点位
        }
        kp1_pyr.push_back(kp_top);
        // kp2_pyr.push_back(kp_top);
    }
    // ROS_WARN("multi layer img\n");
    for (int level = pyramids - 1; level >= 0; level--)
    {
        // from coarse to fine
        successd.clear();// 为什么每次都要清空size呢

        // ROS_WARN("begin single level(%d)\n", level);
        chrono::steady_clock::time_point t3 = chrono::steady_clock::now();
        if (level == 0)
            OpticalFlowSingleLevel(img1_pyr[level], img2_pyr[level], kp1_pyr, kp2_pyr, successd, false, true, level);
        else
            OpticalFlowSingleLevel(img1_pyr[level], img2_pyr[level], kp1_pyr, kp2_pyr, successd, false, true, level);
        // ROS_WARN("success single level(%d)\n", level);
        chrono::steady_clock::time_point t4 = chrono::steady_clock::now();
        auto time_used = chrono::duration_cast<chrono::duration<double>>(t4 - t3);
        // ROS_WARN("layer : %d  use %f s\n", level, time_used);

        if (level > 0)// kp1_pyr点位从循环最开始的coarse to fine
        {
            for (auto &kp : kp1_pyr)
                // cout << "size:" << kp.size() << endl;
                for (int m = 0; m < kp.keyPoint.size(); m++)
                {
                    kp.keyPoint[m] /= pyramid_scale;
                }
        }
    }
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    auto time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    // ROS_WARN("line : %d  use %f s\n", kp1.size(), time_used);
    // ROS_WARN("success multi level, %d lines\n", kp2_pyr.size());
    //目前已经跟踪到了kp2_pyr

    ////////////////////// 经过上面得到了下一帧的kp2_pyr和successd 下面和原版EPLF不同了
    /*ROS_WARN_STREAM("kp1 = "<< kp1_pyr.size());
    ROS_WARN_STREAM("kp2_pyr = "<< kp2_pyr.size());
    for(int i = 0; i < successd.size(); i++)
    {
        ROS_WARN_STREAM("success  "<< successd[i]);
    }*/
    success = successd;// 跟踪完转到外面进行判断距离吧
    kp2 = kp2_pyr;
    //ROS_WARN_STREAM("kp2 = "<< kp2.size());


    //////////////////////

    /*//在kp2中，真正留下的是追踪成功的点
    //这些点对应的序号在success中保存
    cv::Mat mergel = cv::Mat(ROW, COL, CV_8UC1, 255);
    cv::Mat maskl = cv::Mat(ROW, COL, CV_8UC1, 255);
    int lineIndex = 0;
    for (int m = 0; m < kp2_pyr.size(); m++)
    {
        if (successd[m] != -1)
        {
            kp2_pyr[m].recoverLine(successd[m], magnitude, angle, 5);

            bool ifMerge = false;
            // ROS_WARN("adding line ...");
            if (mergel.at<uchar>(kp2_pyr[m].MidPt) != 255 ||
                mergel.at<uchar>(kp2_pyr[m].StartPt) != 255 ||
                mergel.at<uchar>(kp2_pyr[m].EndPt) != 255)
            {
                // ROS_WARN("merge tracked line!!!\n");
                ifMerge = mergeLine(kp2_pyr[m], mergel, kp2, maskl);
                // ROS_WARN("success add line\n");
            }
            if (!ifMerge)
            {
                kp2.push_back(kp2_pyr[m]);
                success.push_back(m);
                drawLine(kp2_pyr[m], mergel, lineIndex++);
            }
        }
    }*/
}


void OpticalFlowSingleLevel(
    //const Mat &magnitude,
    const Mat &img1,
    const Mat &img2,
    const vector<Line> &kp1,
    vector<Line> &kp2,
    vector<int> &success,
    bool inverse, bool has_initial, int layer)
{

    kp2.resize(kp1.size());
    success.resize(kp1.size());// success和待跟踪的line向量的size一样
    OpticalFlowTracker tracker(img1, img2, kp1, kp2, success, inverse, has_initial, layer);
    parallel_for_(Range(0, kp1.size()),
                  std::bind(&OpticalFlowTracker::calculateOpticalFlow, &tracker, placeholders::_1));// parallel_for_是opencv的一个并行计算  也就是说可以同时并行计算不同线特征
}


void OpticalFlowTracker::calculateOpticalFlow(const Range &range)
{
    // 这里设置了一些阈值，但是后来发现不太好用
    int half_patch_size = 4;
    int iterations = 5;
    int door = 40500; //最大cost
    double Graydoor = 5;
    for (size_t i = range.start; i < range.end; i++)// 按待跟踪的线数进行循环
    {
        
        if (success[i] != -1)
        {

            auto kp = kp1[i].keyPoint;// 可以看出，点只用了keypoint里的5个点，端点并未参加跟踪。可能因为端点是不稳定的，下一帧可能消失了。最后跟踪完有个extend函数再判定端点
            if (param[i].empty())
            {
                param[i].resize(3);
                param[i][0] = param[i][1] = param[i][2] = 0;
                //ROS_WARN_STREAM("123");
            }
            else if (layer == _LAYER_ - 1)
                param[i][0] = param[i][1] = param[i][2] = 0;// param存贮的每条线的三个参数
            double &g1 = param[i][0], &g2 = param[i][1], &g3 = param[i][2];
            g1 *= 2;// 每层金字塔缩放的倍数
            g2 *= 2;
            //ROS_WARN_STREAM("1231111");// 到这没问题
            double cost = 0, lastCost = 0;
            int succ = i; // indicate if this point succeeded
            double errorGray = 0;

            // Gauss-Newton iterations
            Eigen::Matrix3d H = Eigen::Matrix3d::Zero(); // hessian
            Eigen::Vector3d b = Eigen::Vector3d::Zero(); // bias
            Eigen::Vector3d J;
            Eigen::Vector2d J0; // jacobian
            for (int iter = 0; iter < iterations; iter++)
            {
                chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
                if (inverse == false)
                {
                    H = Eigen::Matrix3d::Zero();
                    b = Eigen::Vector3d::Zero();
                }
                else
                {
                    // only reset b
                    b = Eigen::Vector3d::Zero();
                }

                cost = 0;
                errorGray = 0;

                // compute cost and jacobian
                for (int m = 0; m < kp.size(); m++)// 一个线上的点数
                {
                    double gray1 = 0, gray2 = 0;
                    // std::cout << kp.size();
                    double dealtx = (kp[m].x - kp[0].x);
                    double dealty = (kp[m].y - kp[0].y);
                    
                    double dx, dy, dx0, dy0;
                    dx0 = dealty * g3;
                    dy0 = dealtx * g3;
                    dx = g1 + dx0;
                    dy = g2 - dy0;

                    for (int x = -half_patch_size; x <= half_patch_size; x++)// 光流法中的开窗
                        for (int y = -half_patch_size; y <= half_patch_size; y++)
                        {

                            double gray01 = GetPixelValue(img1, kp[m].x + x, kp[m].y + y);
                            double gray02 = GetPixelValue(img2, kp[m].x + x + dx, kp[m].y + y + dy);
                            // double error = GetPixelValue(img1, kp[m].x + x, kp[m].y + y) -
                            //                GetPixelValue(img2, kp[m].x + x + dx, kp[m].y + y + dy);
                            double error = gray01 - gray02;
                            // if (iter == 0)
                            // {
                            gray1 += gray01;
                            gray2 += gray02;
                            // }
                            // Jacobian
                            if (inverse == false)
                            {
                                // double dx, dy;
                                // dx = g1 + (kp[m].y - kp[0].y) * g3;
                                // dy = g2 - (kp[m].x - kp[0].x) * g3;
                                J0 = -1.0 * Eigen::Vector2d(
                                                0.5 * (GetPixelValue(img2, kp[m].x + dx + x + 1, kp[m].y + dy + y) -
                                                       GetPixelValue(img2, kp[m].x + dx + x - 1, kp[m].y + dy + y)),
                                                0.5 * (GetPixelValue(img2, kp[m].x + dx + x, kp[m].y + dy + y + 1) -
                                                       GetPixelValue(img2, kp[m].x + dx + x, kp[m].y + dy + y - 1)));
                                Eigen::Matrix<double, 2, 3> J1;
                                J1 << 1.0, 0.0, dealty - dy0,
                                    0.0, 1.0, -dealtx - dx0;
                                J = J0.transpose() * J1;
                            }
                            else if (iter == 0)
                            {
                                // double dx, dy;
                                // dx = g1 + (kp[m].y - kp[0].y) * g3 + (kp[m].x - kp[0].x) * g4;
                                // dy = g2 - (kp[m].x - kp[0].x) * g3 + (kp[m].y - kp[0].y) * g4;

                                J0 = -1.0 * Eigen::Vector2d(
                                                0.5 * (GetPixelValue(img2, kp[m].x + dx + x + 1, kp[m].y + dy + y) -
                                                       GetPixelValue(img2, kp[m].x + dx + x - 1, kp[m].y + dy + y)),
                                                0.5 * (GetPixelValue(img2, kp[m].x + dx + x, kp[m].y + dy + y + 1) -
                                                       GetPixelValue(img2, kp[m].x + dx + x, kp[m].y + dy + y - 1)));
                                Eigen::Matrix<double, 2, 3> J1;
                                J1 << 1.0, 0.0, dealty - dy0,
                                    0.0, 1.0, -dealtx - dx0;
                                J = J0.transpose() * J1;
                            }

                            // compute H, b and set cost;
                            b += -error * J;
                            // cost += huberloss(error, 1) * 2;
                            cost += error * error;
                            if (inverse == false || iter == 0)
                            {
                                // also update H
                                H += J * J.transpose();
                            }
                        }
                    // if (iter == 0)
                    errorGray += abs(gray1 - gray2) / ((2 * half_patch_size + 1) * (2 * half_patch_size + 1));
                }
                chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
                auto time_used1 = chrono::duration_cast<chrono::duration<double>>(t2 - t1);

                chrono::steady_clock::time_point t3 = chrono::steady_clock::now();
                // compute update
                Eigen::Vector3d update = H.ldlt().solve(b);
                chrono::steady_clock::time_point t4 = chrono::steady_clock::now();

                auto time_used2 = chrono::duration_cast<chrono::duration<double>>(t4 - t3);
                // ROS_WARN("compute time DIV solve time : %f", float(time_used1.count() / time_used2.count()));

                if (std::isnan(update[0]) || std::isnan(update[1]) || std::isnan(update[2]) ||
                    std::isinf(update[0]) || std::isinf(update[1]) || std::isinf(update[2]))
                {
                    succ = -1;
                    ROS_WARN("fail line \n");
                    ROS_WARN_STREAM("break0");
                    break;
                }
                if (iter > 0 && cost > lastCost)
                {
                    //ROS_WARN_STREAM("break1");// 现在都是在这里中断的  不过原版EPLF也是经常在这里  这个可能因为不是角点的缺陷
                    break;
                }// 上面这两种属于异常的中断  理论上cost是逐渐减小的 不会说变大

                // update dx, dy
                g1 += update[0];
                g2 += update[1];
                g3 += update[2];

                lastCost = cost;
                succ = i;

                if (update.norm() < 1e-2)
                {
                    // converge
                    // ROS_WARN("converge\n");
                    //ROS_WARN_STREAM("break2");
                    break;
                }
            }// int iter = 0; iter < iterations; iter++
            cost /= ((2 * half_patch_size + 1) * (2 * half_patch_size + 1));

            // ROS_WARN("Error:%f  AvrError:%f", cost, errorGray);
            if (layer == 0 && errorGray >= Graydoor * _POINTNUM_)

                succ = -1;
            // else if (layer == 0)
            // {
            //     ofstream fout("/home/shitong/study/slam/lfvins_catkin/src/linefeature_tracker/src/error.csv", ios::out | ios::app);
            //     fout << cost << ',' << errorGray << endl;
            //     fout.close();
            // }

            // set kp2
            // if (layer == 0)
            for (int m = 0; m < kp.size(); m++)
            {
                //挨个存入kp2
                float dx = g1 + (kp[m].y - kp[0].y) * g3;
                float dy = g2 - (kp[m].x - kp[0].x) * g3;
                float x = kp[m].x + dx;
                float y = kp[m].y + dy;

                if (m == 0 && (x <= 0 || x >= img1.cols - 1 ||
                               y <= 0 || y >= img1.rows - 1))
                    succ = -1;

                if (kp2[i].keyPoint.size() == _POINTNUM_)// 估计需要先弄清这里为啥有的是第一种有的是第二种  才能判断到底是哪里出错了
                {   

                    kp2[i].keyPoint[m] = (Point2f(x, y));
                }
                else
                {    
                    kp2[i].keyPoint.push_back(Point2f(x, y));// 
                }
            }
            // if (layer == 0 && checkgoodLine(img2, kp2[i].keyPoint, 10, 1) == false)
            //     succ = -1;
            /*if (layer == 0 && checkgoodLine(magnitude, kp2[i].keyPoint) == false)
                succ = -1;*/

            success[i] = succ;
        }// if (success[i] != -1)
    }// for loop
}


// llg add for line flow 判断光流跟踪到的keypoint和提取的哪个线匹配
void LKmatch(vector<KeyLine> &line1, vector<KeyLine> &line2, vector<Line> &lk_line, vector<int> &succ, vector<DMatch> &result)
{

    for(int i = 0; i < line1.size(); i++)
    {
        
        if(succ[i] != -1)// 光流跟踪成功了 才能继续进行判断匹配
        {
            //map<int, double> min_distance;// 记录5个keypoint指向最小距离的线的ID  不能用map map中线ID一样就会覆盖 就没法计数了

            vector<int> min_id;// 跟踪到下一帧的线距离最近
            vector<double> min_dis;// 最近距离大小  和id一一对应
            vector<int> cut_num;
            int cut;
            getDistancePL(lk_line[i].keyPoint, line2, min_id, min_dis);
            for(int k = 0; k < min_id.size(); k++)
            {
                //ROS_WARN_STREAM("min id  "<< min_id[k]);
                //ROS_WARN_STREAM("min value  "<< min_dis[k]);
                cut = count(min_id.begin(), min_id.end(), min_id[k]);
                cut_num.push_back(cut);
            }
            int max_count_index = max_element(cut_num.begin(), cut_num.end()) - cut_num.begin();
            int max_count = cut_num[max_count_index];// 离哪个线最近的最多计数
            DMatch lk_match;
            if( min_dis[max_count_index] < 2 && max_count > 1)
            {
                lk_match.queryIdx = min_id[max_count_index];
                lk_match.trainIdx = i;
                lk_match.distance = 10; // PL-VINS中阈值为30 基于光流法不根据这个剔除  全部设置为小于30
                lk_match.imgIdx = 0;

                result.push_back(lk_match);
            }
        }
    }
}


void getDistancePL(vector<Point2f> c, vector<KeyLine> l, vector<int> &id, vector<double> &value)// 一个keypoint到line2的哪个线最近 min_index是下一帧线的索引
{
    for(int m = 0; m < c.size(); m++)
    {
        vector<double> distance;
        for(int i = 0; i < l.size(); i++)
        {
            Point2f a = l[i].getStartPoint();
            Point2f b = l[i].getEndPoint();
            int A = 0, B = 0, C = 0;
            A = a.y - b.y;
            B = b.x - a.x;
            C = a.x * b.y - a.y * b.x;
            double dis = 0.0;
            dis = ((double) abs(A * c[m].x + B * c[m].y + C)) / ((double)sqrt(A*A + B*B));
            distance.push_back(dis);
        }
        int min_index = min_element(distance.begin(), distance.end()) - distance.begin();
        double min_value = distance[min_index];
        id.push_back(min_index);
        value.push_back(min_value);
    }
    
}