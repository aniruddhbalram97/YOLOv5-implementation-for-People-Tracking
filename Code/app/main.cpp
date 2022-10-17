#pragma once
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <fstream>
#include <vector>
//#include <constants.hpp>
#include <object_detection.hpp>

// using namespaces
using namespace std;
using namespace cv;
using namespace cv::dnn;

int main() {
vector<string> class_list;
ifstream ifs;
string line;
BlobGenerator Blob;
HumanObjectDetector HOD;
string file_name = "./../app/coco.names";
// reading coco.names file -> contains the classes defined within the coco dataset
ifs.open(file_name.c_str());
if(ifs.is_open()) {
    cout << "file " << file_name << " is open" << endl;
    while (getline(ifs,line)) {
    class_list.push_back(line);
}
}
else {
    cout << "error with file opening" << endl;
    exit(1);
}
ifs.close();
Mat image_in;
image_in = imread("./../app/traffic.jpg");
// generate blob from image
Blob.generateBlobFromImage(image_in);
Mat blob = Blob.getBlob();
// loading the model
Net yolo_model;
yolo_model = readNet("./../app/models/YOLOv5s.onnx");
vector<Mat> preprocessed_data;
preprocessed_data = HOD.preProcessAlgorithm(blob, yolo_model);
vector<Rect> bounding_boxes = HOD.postProcessAlgorithm(preprocessed_data, image_in, class_list);
Mat img = HOD.applyNMSAndAppendRectanglesToImage(image_in, bounding_boxes, class_list);
imshow("output", img);
waitKey(0);
return 0;
}



