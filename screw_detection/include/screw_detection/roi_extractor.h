/**
 * @file
 * @brief This file contains the declaration of the ROIExtractor class.
 *
 * @author Sebastian Döbler
 * @version 1.0
 */

#ifndef SCREW_DETECTION_ROI_EXTRATOR_H
#define SCREW_DETECTION_ROI_EXTRATOR_H

//OpenCV
#include <opencv2/opencv.hpp>

//STD
#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

//Self
#include "extractor_parameters.h"

namespace screw_detection {

namespace roi_utils {

    const Pixel BLUR_SIDE_LENGTH = 3; //Side length of the cv::blur used in getCirclesHough

    /**
     * @brief Get all circles in format cv::Vec3f(x,y,r).
     * @param  circles Returns all circles detected
     * @param  image Image used to detect circles in
     * @param  remaining Parameters used for cv::HoughCircles
    */
    void getCirclesHough(std::vector<cv::Vec3f>& circles, const cv::Mat& image, const int contour_threshold,
        const int accumulator_threshold, const int min_circle_size, const int max_circle_size,
        const int min_circle_distance);
    /**
     * @brief Calls getCirclesHough() with HoughParameter Object
    */
    void getCirclesHough(std::vector<cv::Vec3f>& circles, const cv::Mat& image, const HoughParameters& parameters);
}

/**
     * @brief ROI Extractor
     * has the purpose of extracting the ROI from an image containing screws for further processing.
     */
class ROIExtractor {
public:
    /**
     * @brief Expects ExtractorParameters to already be initiated
    */
    ROIExtractor(const ExtractorParameters& params);

    /**
     * @brief Returns the single images of each screw in an image.
     * @param circles Returns all circles in the original image as (x,y,r)
     * @param roi_reference Returns the roi_reference from CustomROI.getROI()
     * @param roi Returns the extracted roi from CustomROI.getROI()
     * @param image Unmasked image containing screws
     * @param extend_screw_image Used to extend the extracted images by a flat amount. Usefull if the images
     * are supposed to be visualized, to preserve some image context
    */
    std::vector<cv::Mat> getScrewImages(std::vector<cv::Vec3f>& circles, cv::Vec2f& roi_reference,
        cv::Mat& roi, const cv::Mat& image, const int extend_screw_image = 0);
    /**
     * @brief Proxy for getScrewImages() without returning the roi
    */
    std::vector<cv::Mat> getScrewImages(std::vector<cv::Vec3f>& circles, cv::Vec2f& roi_reference,
        const cv::Mat& image, const int extend_screw_image = 0);

    /**
     * @brief Returns all circles inside image using the HoughParameters for Screws.
     * @param circles Return value
     * @param image which is already masked to only contain the ROI
    */
    void getScrewCircles(std::vector<cv::Vec3f>& circles, const cv::Mat& image);

    //Misc

    const ExtractorParameters& extractor_parameters() const { return this->parameters_; }

private:
    //Parameters
    ExtractorParameters parameters_;

    /**
     * @brief Internal function for getScrewImages. Returns ROI, roi reference and circles inside ROI
     */
    void getScrewsInROI(std::vector<cv::Vec3f>& circles, cv::Vec2f& roi_reference, cv::Mat& roi,
        const cv::Mat& image);
};
}

#endif
