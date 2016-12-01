#ifndef APPLYCLAHE_H
#define APPLYCLAHE_H

#include <opencv2/core.hpp>

cv::Mat applyCLAHE(const cv::Mat& src, double const clipLimit = 4, int const tilesGridSize = 4);

#endif // APPLYCLAHE_H
