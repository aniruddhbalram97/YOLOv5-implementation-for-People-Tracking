// Constants.h
#include <opencv2/opencv.hpp>
#include <fstream>

#ifndef MYLIB_CONSTANTS_H
#define MYLIB_CONSTANTS_H

// using namespaces
using namespace std;
using namespace cv;
using namespace cv::dnn;

namespace ODConstants {
    // Declaring constants for the input blob size
    // Using 640x640 input images 
    const double WIDTH_OF_INPUT = 640.0;
    const double HEIGHT_OF_INPUT = 640.0;
    // Defining the thresholds for filtering
    const double THRES_SCORE = 0.5; // To filter low probability class score
    const double THRES_NMS = 0.5; // To filter out overlapping boxes using NMS
    const double THRES_CONF = 0.5; // Confidence threshold which filters out low confidence detections
    // Defining colors 
    Scalar R = Scalar(255, 0, 0);
    Scalar G = Scalar(0, 255, 0);
    Scalar B = Scalar(0, 0, 255);
    Scalar BLACK = Scalar(0, 0, 0);
    Scalar WHITE = Scalar(255,255,255);
    // Defining font properties
    const double F_SCALE = 1;
    const int F_STYLE = FONT_HERSHEY_COMPLEX;
    const int F_THICKNESS = 2;
}
#endif
