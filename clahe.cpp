#include "clahe.h"

#include <opencv2/imgproc.hpp>
#include <iostream>

cv::Mat applyCLAHE(const cv::Mat& _bgr_image, double const clipLimit, int const tilesGridSize) {
    cv::Mat bgr_image;
    cv::normalize(_bgr_image, bgr_image, 0, 1, cv::NORM_MINMAX, CV_32FC3);

    cv::Mat lab_image;
    cv::cvtColor(bgr_image, lab_image, CV_BGR2Lab);

    // Extract the L channel
    std::vector<cv::Mat> lab_planes(3);
    cv::split(lab_image, lab_planes);  // now we have the L image in lab_planes[0]

    // apply the CLAHE algorithm to the L channel
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
    clahe->setClipLimit(clipLimit);
    clahe->setTilesGridSize(cv::Size(tilesGridSize, tilesGridSize));
    cv::Mat dst;
    double min = 0, max = 0;
    cv::minMaxLoc(lab_planes[0], &min, &max);
    std::cout << "Min / max luminance: " << min << ", " << max << std::endl;
    cv::minMaxLoc(lab_planes[1], &min, &max);
    std::cout << "Min / max a: " << min << ", " << max << std::endl;
    cv::minMaxLoc(lab_planes[2], &min, &max);
    std::cout << "Min / max b: " << min << ", " << max << std::endl;
    cv::normalize(lab_planes[0], lab_planes[0], 0, 65535, cv::NORM_MINMAX, CV_16UC1);
    clahe->apply(lab_planes[0], dst);

    // Merge the the color planes back into an Lab image
    //dst.convertTo(lab_planes[0], CV_32FC1);
    //lab_planes[0] *= max / 65535.0;
    //lab_planes[0] += min;
    cv::normalize(dst, lab_planes[0], 0, 100, cv::NORM_MINMAX, CV_32FC1);
    cv::merge(lab_planes, lab_image);

   // convert back to RGB
   cv::Mat image_clahe;
   cv::cvtColor(lab_image, image_clahe, CV_Lab2BGR);
   return image_clahe;
}
