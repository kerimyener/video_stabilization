/*!
 * \author	Kerim Yener Yurtdas<yurtdask@uni-bremen.de> 3020920
 * \date	01-November-2016
 */



#include <iostream>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include <opencv2/opencv.hpp>
#include <cmath>
#include "ros/ros.h"
#include "std_msgs/String.h"
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/Image.h>

using namespace std;
using namespace cv;

const int HORIZONTAL_BORDER_CROP = 20;
const int SMOOTHING_RADIUS = 30;

struct TransformParam
{
    TransformParam() {}
    TransformParam(double _dx, double _dy, double _da) {
        dx = _dx;
        dy = _dy;
        da = _da;
    }

    double dx;
    double dy;
    double da; // angle
};

struct Trajectory
{
    Trajectory() {}
    Trajectory(double _x, double _y, double _a) {
        x = _x;
        y = _y;
        a = _a;
    }
    // "+"
    friend Trajectory operator+(const Trajectory &c1,const Trajectory  &c2){
        return Trajectory(c1.x+c2.x,c1.y+c2.y,c1.a+c2.a);
    }
    //"-"
    friend Trajectory operator-(const Trajectory &c1,const Trajectory  &c2){
        return Trajectory(c1.x-c2.x,c1.y-c2.y,c1.a-c2.a);
    }
    //"*"
    friend Trajectory operator*(const Trajectory &c1,const Trajectory  &c2){
        return Trajectory(c1.x*c2.x,c1.y*c2.y,c1.a*c2.a);
    }
    //"/"
    friend Trajectory operator/(const Trajectory &c1,const Trajectory  &c2){
        return Trajectory(c1.x/c2.x,c1.y/c2.y,c1.a/c2.a);
    }
    //"="
    Trajectory operator =(const Trajectory &rx){
        x = rx.x;
        y = rx.y;
        a = rx.a;
        return Trajectory(x,y,a);
    }

    double x;
    double y;
    double a; // angle
};
//
int main(int argc, char **argv)
{
    ros::init(argc, argv, "main");
    ros::NodeHandle nh;
    image_transport::ImageTransport it(nh);
    image_transport::Publisher pub = it.advertise("camera/image", 1);

    VideoCapture cap("/home/kerim/Desktop/stabil13.avi");  //"/home/kerim/Desktop/MAH04128.MP4"
    assert(cap.isOpened());
    Size winSize(50,50);
    TermCriteria criteria(TermCriteria::COUNT|TermCriteria::EPS,80,0.01);

    Mat cur, cur_grey, cur_grey2;
    Mat prev, prev_grey, prev_grey2;
    Mat T(2,3,CV_64F);
    Mat last_T(2,3,CV_64F);
    Mat descriptors_1, descriptors_2;
    Mat img_matches;

    double a = 0;
    double x = 0;
    double y = 0;
    int counter = 0;
    char file [30000];

    Trajectory X;//posterior state estimate
    Trajectory X_;//prior estimate
    Trajectory P;// posterior estimate error covariance
    Trajectory P_;// prior estimate error covariance
    Trajectory K;//gain
    Trajectory z;//actual measurement
    double pstd = 4e-3;//can be changed
    double cstd = 0.35;//can be changed
    Trajectory Q(pstd,pstd,pstd);// process noise covariance
    Trajectory R(cstd,cstd,cstd);// measurement noise covariance

    SurfFeatureDetector detector( 400 );
    SurfDescriptorExtractor extractor;
    FlannBasedMatcher matcher;

    vector<KeyPoint>previousKey;
    cap >> prev;//get the first frame.ch
    resize(prev, prev, Size(420,280));
    cout << prev.size();
    cvtColor(prev, prev_grey, COLOR_BGR2GRAY);
    detector.detect(prev_grey, previousKey);
    extractor.compute(prev_grey, previousKey, descriptors_1);

    int vert_border = HORIZONTAL_BORDER_CROP * prev.rows / prev.cols; // get the aspect ratio correct
    int k=1;

    VideoWriter outputVideo;
    outputVideo.open("/home/kerim/Desktop/stabil14.avi" , CV_FOURCC('D','I','V','3'), 15, Size(420, 280)  );

    if ( !outputVideo.isOpened() ) //if not initialize the VideoWriter successfully, exit the program
    {
        cout << "ERROR: Failed to write the video" << endl;
        return -1;
    }

    while (nh.ok()) {

        /*drawKeypoints(prev_grey, previousKey, prev_grey2, Scalar(0, 0, 255), DrawMatchesFlags::DEFAULT);
        sprintf(file, "/home/kerim/Desktop/stabil/features_prev %d.jpeg", counter);
        imwrite(file, prev_grey2);
        counter ++;*/

        cap >> cur;
        resize(cur, cur, Size(420,280));
        /*sprintf(file, "/home/kerim/Desktop/stabil/cur_normal %d.jpeg", counter);
        imwrite(file, cur);*/

        if(cur.data == NULL) {
            break;
        }
        vector <Point2f> prev_corner, cur_corner;
        vector <Point2f> prev_corner2, cur_corner2;
        vector <uchar>   status;
        vector <float>   err;
        vector<KeyPoint> currentKey;
        vector< DMatch > matches, good_matches;

        cvtColor(cur, cur_grey, COLOR_BGR2GRAY);
        detector.detect( cur_grey, currentKey );
        extractor.compute(cur_grey, currentKey, descriptors_2);

        /*drawKeypoints(cur_grey, currentKey, cur_grey2, Scalar(0, 0, 255), DrawMatchesFlags::DEFAULT);
        sprintf(file, "/home/kerim/Desktop/stabil/features_cur %d.jpeg", counter);
        imwrite(file, cur_grey2);*/

        if((descriptors_1.empty())||(descriptors_2.empty()))
            continue;

        matcher.match( descriptors_1, descriptors_2, matches );

        double max_dist = 0; double min_dist = 100;

        for( int i = 0; i < descriptors_1.rows; i++ )
        { double dist = matches[i].distance;
            if( dist < min_dist ) min_dist = dist;
            if( dist > max_dist ) max_dist = dist;
        }

        for( int i = 0; i < descriptors_1.rows; i++ )
        { if( matches[i].distance < 5*min_dist )
            { good_matches.push_back( matches[i]); }
        }

        for( int i = 0; i < good_matches.size(); i++ )
        {
            //-- Get the keypoints from the good matches
            prev_corner.push_back( previousKey[ good_matches[i].queryIdx ].pt );
        }

        /*drawMatches( prev_grey, previousKey, cur_grey, currentKey,
                     good_matches, img_matches, Scalar(0,255,0), Scalar(0,0,255),
                     vector<char>(), DrawMatchesFlags::DEFAULT );
        sprintf(file, "/home/kerim/Desktop/stabil/matches %d.jpeg", counter);
        imwrite(file, img_matches);*/

        int new_transformation_found = 1;
        try
        {
            calcOpticalFlowPyrLK(prev_grey, cur_grey, prev_corner, cur_corner, status, err, winSize, 3,criteria, 0, 0.0001);
        }
        catch(Exception& ){
            new_transformation_found = 0;
        }
        // weed out bad matches
        if (new_transformation_found){
            for(size_t i=0; i < status.size(); i++) {
                if(status[i]) {
                    prev_corner2.push_back(prev_corner[i]);
                    cur_corner2.push_back(cur_corner[i]);
                    //Point p0( ceil( prev_corner2[i].x ), ceil( prev_corner2[i].y ) );
                    //Point p1( ceil( cur_corner2[i].x ), ceil( cur_corner2[i].y ) );
                    //line( cur, p0, p1, CV_RGB(0,0,255), 2 );
                }
            }
        }

        // translation + rotation only
        try{
            T = estimateRigidTransform(prev_corner2, cur_corner2, false); // false = rigid transform, no scaling/shearing
        }
        catch(Exception& ){
            last_T.copyTo(T);
        }
        // in rare cases no transform is found. We'll just use the last known good transform.
        if(T.data == NULL) {
            last_T.copyTo(T);
        }

        T.copyTo(last_T);

        // decompose T
        double dx = T.at<double>(0,2);
        double dy = T.at<double>(1,2);
        double da = atan2(T.at<double>(1,0), T.at<double>(0,0));
        //
        x += dx;
        y += dy;
        a += da;
        //

        z = Trajectory(x,y,a);//measurement value
        //
        if(k==1){
            // intial guesses
            X = Trajectory(0,0,0); //Initial estimate,  set 0
            P = Trajectory(1,1,1); //set error variance,set 1
        }
        else
        {
            //time update（prediction）
            X_ = X;
            P_ = P+Q;
            // measurement update（correction）
            K = P_/( P_+R );
            X = X_+K*(z-X_);
            P = (Trajectory(1,1,1)-K)*P_;
        }
        // target - current
        double diff_x = X.x - x;
        double diff_y = X.y - y;
        double diff_a = X.a - a;

        dx = dx + diff_x;
        dy = dy + diff_y;
        da = da + diff_a;
        //
        T.at<double>(0,0) = cos(da);
        T.at<double>(0,1) = -sin(da);
        T.at<double>(1,0) = sin(da);
        T.at<double>(1,1) = cos(da);
        T.at<double>(0,2) = dx;
        T.at<double>(1,2) = dy;

        Mat cur2, cur2_sharp;

        warpAffine(prev, cur2_sharp, T, cur.size());
        cv::GaussianBlur(cur2_sharp, cur2, cv::Size(0, 0), 3);
        cv::addWeighted(cur2_sharp, 1.5, cur2, -0.5, 0, cur2);

        cur2 = cur2(Range(vert_border, cur2.rows-vert_border),
                    Range(HORIZONTAL_BORDER_CROP, cur2.cols-HORIZONTAL_BORDER_CROP));

        // Resize cur2 back to cur size, for better side by side comparison
        resize(cur2, cur2, cur.size());

        // Now draw the original and stablised side by side for coolness
        Mat canvas = Mat::zeros(cur.rows, cur.cols, cur.type());///cur.cols*2+10

        //prev.copyTo(canvas(Range::all(), Range(0, cur2.cols)));
        cur2.copyTo(canvas(Range::all(), Range(0, cur2.cols)));///cur2.cols+10, cur2.cols*2+10

        // If too big to fit on the screen, then scale it down by 2, hopefully it'll fit :)
        if(canvas.cols > 1920) {
            resize(canvas, canvas, Size(canvas.cols/2, canvas.rows/2));
        }
        outputVideo << canvas;

        imshow("before and after", canvas);
        //imshow("matches", img_matches);

        //sprintf(file, "/home/kerim/Desktop/stabil/optic %d.jpeg", counter);
        //imwrite(file, canvas);
        /*sprintf(file, "/home/kerim/Desktop/stabil/cur_stabil %d.jpeg", counter);
        imwrite(file, cur2);*/

        waitKey(1);
        //
        cur.copyTo(prev);
        cur_grey.copyTo(prev_grey);
        descriptors_2.copyTo(descriptors_1);
        previousKey = currentKey ;

        cout << "Frame: " << k << "/" << " - good optical flow: " << prev_corner2.size() << endl;
        k++;
        sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", cur2).toImageMsg();
        pub.publish(msg);
        ros::spinOnce();

    }

    outputVideo.release();
    return 0;
}

