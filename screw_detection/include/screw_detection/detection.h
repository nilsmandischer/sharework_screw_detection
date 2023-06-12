/**
 * @file
 * @brief This file contains the declaration of the ScrewDetector class.
 *
 * @author Sebastian Döbler
 * @version 1.0
 */

#ifndef SCREW_DETECTION_H
#define SCREW_DETECTION_H

//OpenCV
#include <opencv2/opencv.hpp>

//STD
#include <string>
#include <vector>

//Self
#include "extractor_parameters.h"
#include "model.h"
#include "roi_extractor.h"

namespace screw_detection {
/**
     * @brief ScrewDetector
     * has one main pipeline (processImage) which takes an image and returns found circles and 
     * wether the circles represents a screw. Also returns some processing data.
     */
class ScrewDetector {
public:
    /**
     * @brief Constructor that takes extractor parameters and the path from which to load the model.
     * @param  params Extractor parameters to use, Expects the parameters to be initiated before being passed
     * @param  model_data_path Abolsute path from which to load model_data.yaml
     */
    ScrewDetector(const ExtractorParameters& params, const std::string& model_data_path);

    /**
         * @brief Image entry point. Calls detection on received image.
     * @param  circles Returns circle which represent screws or holes
     * @param  roi_reference Returns the roi_reference from CustomROI.getROI()
     * @param  roi Returns the roi from CustomROI.getROI()
     * @param  are_screws Vector representing one circle per element. True := is screw
     * @param  image Unprocessed image to use for detection
         */
    bool processImage(std::vector<cv::Vec3f>& circles, cv::Vec2f& roi_reference, cv::Mat& roi,
        std::vector<bool>& are_screws, const cv::Mat& image);
    /**
     * @brief Proxy for processImage without returning roi image
    */
    bool processImage(std::vector<cv::Vec3f>& circles, cv::Vec2f& roi_reference,
        std::vector<bool>& are_screws, const cv::Mat& image);

    /**
     * @brief Resets the ROIExtractor object  using the passed parameters. Expects the parameters to be initiated
     * before being passed
    */
    void setParameters(const ExtractorParameters& parameters);

private:
    //Prediction model
    ScrewDetectorModel ml_model_;
    ROIExtractor roi_extractor_;
};
}

#endif
