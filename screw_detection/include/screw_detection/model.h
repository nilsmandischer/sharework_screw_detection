/**
 * @file
 * @brief This file contains the declaration of the ScrewDetectorModel class.
 *
 * @author Sebastian Döbler
 * @version 1.0
 */

#ifndef SCREW_DETECTION_MODEL_H
#define SCREW_DETECTION_MODEL_H

//OpenCV
#include <opencv2/objdetect.hpp>
#include <opencv2/opencv.hpp>

//STD
#include <string>
#include <vector>

namespace screw_detection {

/**
 * @brief Defines the cv::DTree to use. It is not recommended to use something else than Forst, 
 * unless other features are used.
*/
enum ModelType { Forest,
    AdaBoost,
    SVM };

const ModelType USED_MODEL_TYPE = Forest;

const std::string MODEL_FILE_NAME = "/model_data.yaml";

/**
 * @brief ModelParameters
 * contains all parameters used for the creation of the used cv::DTree with default values.
 * For parameter meanings refer to cv::DTree
 */
struct ModelParameters {
    //Universal (at least used by two Models)
    int max_depth;
    int min_sample_count;
    double weight_screws;
    float regression_accuracy = 0.0f;
    bool use_surrogates = false;
    int cv_folds = 1;
    bool use_1se_rule = false;
    bool truncate_pruned_tree = true;
    const int CATEGORY_AMOUNT = 2;

    //Termination
    cv::TermCriteria::Type term_criteria = cv::TermCriteria::MAX_ITER;
    int max_iterations;

    //Only Forest
    int active_var_count = 0;

    //Only AdaBoost
    int weak_count = 1;

    //Only SVM
    cv::ml::SVM::Types svm_type = cv::ml::SVM::C_SVC;
    cv::ml::SVM::KernelTypes svm_kernel = cv::ml::SVM::RBF;
};

/**
     * @brief ScrewDetectorModel
     * is an interface to variing cv::DTrees and calculates the features used for screw detection
     */
class ScrewDetectorModel {
public:
    ScrewDetectorModel(const ModelType type);

    /**
     * @brief initializes the DTree type specified in the constructor with the passed parameters.
    */
    void setParameters(const ModelParameters& parameters);

    /**
     * @brief Calls cv::DTree write
    */
    void write(const std::string& path);

    /**
     * @brief Calls cv::DTree read
    */
    void read(const std::string& path);

    /**
     * @brief Calls cv::DTree train
    */
    void train(const cv::Mat& trainer_data, const cv::Mat& result_data);

    /**
     * @brief Writes some model information to the console to estimate the performance. 
     * Only Forest Version is actually reliable
    */
    void getModelInfo(void);

    /**
         * @brief Uses trained model to estimate if passed rois contain a screw
     * @param  are_screws Retruns for each element of rois vector wether element is a screw
     * @param  rois Cropped images containing only the single screws ROI
     * @param circles Circles inside each single roi in the global frame
     * @param roi_reference Reference used for feature calculation
         */
    void predict(std::vector<bool>& are_screws, const std::vector<cv::Mat>& rois,
        const std::vector<cv::Vec3f>& circles, const cv::Vec2f& roi_reference);
    /**
     * Calls cv::DTree predict
    */
    float predictCV(const cv::Mat& sample);

    /**
      * @brief Calculates all feature values and returns as 2 layer vector
     * @param  screw_values vector[image][feature] = feature value for each image in rois 
     * and corresponding circle in circles
     * @param  rois Images of each single screw cropped
     * @param  circles Circles for each ROI, needs to have same size as rois
     * @param  roi_reference Reference which is used for feature calculation
     */
    void calculateFeatures(std::vector<std::vector<float>>& screw_values, const std::vector<cv::Mat>& rois,
        const std::vector<cv::Vec3f>& circles, const cv::Vec2f& roi_reference);
    /**
     * @brief Experimental addition of using HOG Features. Did not increase performance on our data.
    */
    void calculateHOGFeatures(std::vector<float>& feature_values, const cv::Mat& image, const bool use_flip = false);

private:
    //Prediction model
    cv::Ptr<cv::ml::RTrees> forest_;
    cv::Ptr<cv::ml::Boost> adaboost_;
    cv::Ptr<cv::ml::SVM> svm_;
    ModelType model_type_;

    //Used to calculate a feature
    static constexpr double PLATEUS_OUTLIER_TOLERANCE_ = 10;
    static constexpr int NUM_OUTLIERS_NEW_PLATEU_ = 4;
    static constexpr float CENTER_SPOT_RADIUS_PERCENT_ = 0.25f;
    static constexpr float BRIGHT_RING_RADIUS_PERCENT_ = 0.5f;
};
}

#endif
