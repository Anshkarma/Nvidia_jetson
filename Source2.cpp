#include <opencv2/opencv.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudabgsegm.hpp>
#include <iostream>

int main() {
    // Open video stream (0 for default camera)
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Error opening video stream\n";
        return -1;
    }

    // Try to use CUDA background subtractor
    cv::Ptr<cv::cuda::BackgroundSubtractorMOG2> pBackSub;
    try {
        pBackSub = cv::cuda::createBackgroundSubtractorMOG2();
    } catch (const cv::Exception&) {
        std::cerr << "CUDA MOG2 not available, falling back to CPU.\n";
        // Fallback to CPU if CUDA is not available
        cv::Ptr<cv::BackgroundSubtractor> cpuBackSub = cv::createBackgroundSubtractorMOG2();
        cv::Mat frame, fgMask;
        std::vector<std::vector<cv::Point>> contours;
        while (true) {
            cap >> frame;
            if (frame.empty())
                break;
            cpuBackSub->apply(frame, fgMask);
            cv::findContours(fgMask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
            for (auto& contour : contours) {
                double area = cv::contourArea(contour);
                if (area > 500) {
                    cv::Rect boundingBox = cv::boundingRect(contour);
                    cv::rectangle(frame, boundingBox, cv::Scalar(0, 255, 0), 2);
                }
            }
            cv::imshow("Frame", frame);
            cv::imshow("FG Mask", fgMask);
            char c = (char)cv::waitKey(30);
            if (c == 27)
                break;
        }
        cap.release();
        cv::destroyAllWindows();
        return 0;
    }

    // CUDA path
    cv::Mat frame;
    cv::cuda::GpuMat d_frame, d_fgMask;
    std::vector<std::vector<cv::Point>> contours;
    // CUDA path
    cv::Mat frame;
    cv::cuda::GpuMat d_frame, d_fgMask;
    std::vector<std::vector<cv::Point>> contours;
    while (true) {
        cap >> frame;
        if (frame.empty())
            break;

        // Apply background subtraction
        pBackSub->apply(frame, fgMask);

        // Find contours on the foreground mask
        cv::findContours(fgMask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        // Filter contours and draw bounding boxes
        for (auto& contour : contours) {
            double area = cv::contourArea(contour);
            if (area > 500) { // Filter small contours
                cv::Rect boundingBox = cv::boundingRect(contour);
                cv::rectangle(frame, boundingBox, cv::Scalar(0, 255, 0), 2);
            }
        }

        // Show results
        cv::imshow("Frame", frame);
        cv::imshow("FG Mask", fgMask);

        char c = (char)cv::waitKey(30);
        if (c == 27) // ESC key to exit
            break;
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}       