#include "opencv2/objdetect.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

void detectAndDisplay(Mat frame);
CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;
String window_name = "Capture - Face detection";

int main(int argc, const char** argv)
{
    VideoCapture capture;
    Mat frame;

    if (!face_cascade.load("haarcascade_frontalface_alt.xml"))     { printf("(!) Error loading face cascade\n"); return -1; };
    if (!eyes_cascade.load("haarcascade_eye_tree_eyeglasses.xml")) { printf("(!) Error loading eyes cascade\n"); return -1; };

    capture.open(0);
    if (!capture.isOpened()) { printf("(!) Error opening video capture\n"); return -1; }

    while (capture.read(frame))
    {
        if (frame.empty()) { printf("(!) No captured frame -- Break!"); break; }

        detectAndDisplay(frame);

        char c = (char)waitKey(10);
        if (c == 27) { break; } // escape
    }
}

Mat gray(Mat frame)
{
    Mat grame;
    cvtColor(frame, grame, COLOR_BGR2GRAY);
    equalizeHist(grame, grame);
    return grame;
}

std::vector<Rect> detectFaces(Mat frame)
{
    std::vector<Rect> faces;
    face_cascade.detectMultiScale(frame, faces, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));
    return faces;
}

std::vector<Rect> detectEyes(Mat frame)
{
    std::vector<Rect> eyes;
    eyes_cascade.detectMultiScale(frame, eyes, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));
    return eyes;
}

void drawFaceRegion(Mat frame, const Rect& face)
{
    Point center(face.x + face.width / 2, face.y + face.height / 2);
    ellipse(frame, center, Size(face.width / 2, face.height / 2), 0, 0, 360, Scalar(255, 0, 255), 4, 8, 0);
}

void drawEyesRegion(Mat frame, const Rect& face, const Rect& eyes)
{
    Point eye_center(face.x + eyes.x + eyes.width / 2, face.y + eyes.y + eyes.height / 2);
    int radius = cvRound((eyes.width + eyes.height) * 0.25);
    circle(frame, eye_center, radius, Scalar(255, 0, 0), 4, 8, 0);
}

void detectAndDisplay(Mat frame)
{
    Mat grame = gray(frame);

    const std::vector<Rect> faces = detectFaces(grame);

    for (const auto& face : faces)
    {
        drawFaceRegion(frame, face);        
        const std::vector<Rect>     eyesPairs = detectEyes(grame(face));
        for (const auto& eyesPair : eyesPairs)
        {
            drawEyesRegion(frame, face, eyesPair);
        }
    }
    imshow(window_name, frame);
}
