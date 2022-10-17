#pragma once
#include <opencv2/opencv.hpp>
#include <fstream>
#include <vector>

using namespace std;
using namespace cv;
using namespace cv::dnn;

class BlobGenerator {
    private:
        Mat blob;
    public:
        void generateBlobFromImage(Mat &image_in);
        Mat getBlob();
};

class HumanObjectDetector: public BlobGenerator {
    private:
        vector<int> ids;
        vector<float> confidence_values;
        vector<Rect> bounding_boxes;
        Net net;
        vector<Mat> detections;
    public:
        // function to draw a label around the 'class-text'
        void labelBox(Mat& image_in, string label_value, int posTop, int posLeft);
        // function to convert an input image to a blob
        // function to preprocess the image blob : forward propagate the input blob into a model 
        // trained on COCO 2017 dataset to obtain properties such as confidence and class prediction
        vector<Mat> preProcessAlgorithm(Mat blob, Net &net);
        // function to get the valid class from the preprocessed blob
        vector<Rect> postProcessAlgorithm(vector<Mat>& preprocessed_data, Mat& image_in, const vector<string>& name_of_class);
        Mat applyNMSAndAppendRectanglesToImage(Mat &image_in, vector<Rect> &bounding_boxes, const vector<string> &name_of_class);
};

