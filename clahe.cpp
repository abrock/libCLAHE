#include "clahe.h"

#include <opencv2/imgproc.hpp>
#include <iostream>

std::string type2str(int type) {
  std::string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
  }

  r += "C";
  r += (chans+'0');

  return r;
}

cv::Mat applyCLAHE(const cv::Mat& _bgr_image, double const clipLimit, int const tilesGridSize) {
    cv::Mat bgr_image;
    cv::normalize(_bgr_image, bgr_image, 0, 1, cv::NORM_MINMAX, CV_32FC3);

    if (bgr_image.channels() == 1) {
        cv::Mat tmp;
        cv::cvtColor(bgr_image, tmp, CV_GRAY2BGR, 3);
        bgr_image = tmp;
    }

    if (bgr_image.channels() != 3 && bgr_image.channels() != 4) {
        std::cout << "Unexpected number of channesl (" << bgr_image.channels() << "), expected 3 or 4" << std::endl;
        std::cout << "Original image type: " << type2str(_bgr_image.type()) << std::endl;
        std::cout << "Converted image type: " << type2str(bgr_image.type()) << std::endl;
    }

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
    cv::normalize(image_clahe, image_clahe, 0, 255, cv::NORM_MINMAX, CV_8UC3);
   return image_clahe;
}
