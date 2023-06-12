/**
 * @file
 * @brief This file contains the declaration of the ScrewTrainer class.
 *
 * @author Sebastian Döbler
 * @version 1.0
 */

#ifndef SCREW_DETECTION_TRAINING_H
#define SCREW_DETECTION_TRAINING_H

// OpenCV
#include <opencv2/imgproc.hpp> //For cv::Filled
#include <opencv2/opencv.hpp>

// STD
#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

//Self
#include "extractor_parameters.h"
#include "model.h"
#include "roi_extractor.h"

namespace screw_detection {
namespace {
    const std::string MISC_FILE_NAME = "/misc_data.yaml";

    const std::string SCREWS_FOLDER = "/screws";
    const std::string HOLES_FOLDER = "/holes";
    const std::string UNSORTED_FOLDER = "/unsorted";

    const std::string CIRCLE_FOLDER_PREFIX = "/circle_";

    const std::string OLD_SUFFIX = "_old";
    const std::string FILE_TYPE_SUFFIX = ".png";

    //Control buttons as ASCII code
    const int BUTTON_SCREW = (int)'w'; //65362;
    const int BUTTON_HOLE = (int)'s'; //65364;
    const int BUTTON_SKIP = (int)'d'; //65363;
    const int BUTTON_RETURN = (int)'a'; //65361;
    const int BUTTON_QUIT = (int)'q';
}

/**
 * @brief ScrewData
 * contains data sets for training the model. Each a single screw image, a roi reference and if the image
 * contains a screw
 */
struct ScrewData {
    cv::Mat image;
    cv::Vec2f roi_reference;
    bool screws;
};

/**
 * @brief TrainerParameters
 * contains parameters for the ScrewTrainer
 */
struct TrainerParameters {
    std::string image_prefix;
    std::string image_suffix;
    int image_start_number;

    std::string image_path;
};

/**
 * @brief ScrewTrainer
 * Creates a ROIExtractor and uses its detection features to train a ScrewDetectionModel
 */
class ScrewTrainer {

public:
    /**
 * @param  extractor_params Parameters to pass to the created ROIExtractor. 
 * Does not expect the parameters to be initiated
 * @param  trainer_params General parameters for training
 * @param  model_params Parameters for training the model
 */
    ScrewTrainer(ExtractorParameters& extractor_params, TrainerParameters& trainer_params,
        ModelParameters& model_params);

    /**
    * @brief Runs both spliceImages() and trainModel(). Use this if you do not want to check the resulting
    * spliced images before training
    */
    void run(void);

    /**
    * @brief Loads images from the image path following the conventions described 
    * in the TrainerParameters and uses the specified ExtractorParameters to splice all images into single 
    * cropped images containing one circle each and writing those to disk into the same folder
   */
    void spliceImages(void);

    /**
    * @brief Loads the previously spliced images and trains a model on these parameters
   */
    void trainModel(void);

private:
    TrainerParameters trainer_params_;
    ExtractorParameters extractor_parameters_;

    ScrewDetectorModel ml_model_;
    std::shared_ptr<ROIExtractor> extractor_;

    std::vector<ScrewData> data_;

    const int EXTEND_SCREW_IMAGES_ = 10;
    const int CV_IMAGE_INTERPOLATION_TYPE_ = cv::INTER_LINEAR;
    const int TRAINER_IMAGE_SIZE_ = 480;

    // Functions
    /**
 * @brief Load full images for detecting circles and following splicing. Needs to be manually sorted
 */
    void loadFullImages(void);

    /**
 * @brief Loads sorted images from subfolders /screws and /holes.
 */
    void loadSplicedImages(void);

    /**
 * @brief Saves each circle image in corresponding folder depending on user input.
 */
    void saveCutData(void);

    /**
 * @brief Train Model using ROIExtractor.
 * @param  trainer_data row wise matrix with all feature values of one object in a single row
 * @param  result_data Single column matrix with the result number corresponding to the same row of trainr_data
 */
    void train(cv::Mat& trainer_data, cv::Mat& result_data);

    /**
 * @brief Tests Model validity.
 * @param  trainer_data row wise matrix with all feature values of one object in a single row
 * @param  result_data Single column matrix with the result number corresponding to the same row of trainr_data
 */
    void test(const cv::Mat& trainer_data, const cv::Mat& result_data);
    /**
 * @brief Converts vector[image][screw][feature] into cv::Mat to use for Model, one row per screw. 
 * one column per feature
 * @param  trainer_data Empty matrix. Returns feature values of each object in all_features row-wise
 * @param  result_data Empty matrix. Returns result of each object row-wise
 * @param  all_features Input all featurs calculated. Order ist all_features[image][object][feature]
 */
    void convertToTrainerData(cv::Mat& trainer_data, cv::Mat& result_data,
        const std::vector<std::vector<std::vector<float>>>& all_features);
};

namespace {
    /**
    * @brief Splits strings at each seperator
   */
    std::vector<std::string> split(const std::string& s, const char seperator);

}

}

#endif
