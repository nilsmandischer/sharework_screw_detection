#include <screw_detection/detection.h>

namespace screw_detection {

ScrewDetector::ScrewDetector(const ExtractorParameters& params, const std::string& data_path)
    : roi_extractor_(params)
    , ml_model_(USED_MODEL_TYPE)
{
    //Generate tree path and load model
    std::string model_data_path = data_path + MODEL_FILE_NAME;
    ml_model_.read(model_data_path);
}

bool ScrewDetector::processImage(std::vector<cv::Vec3f>& circles, cv::Vec2f& roi_reference,
    std::vector<bool>& are_screws, const cv::Mat& image)
{
    cv::Mat roi;
    return processImage(circles, roi_reference, roi, are_screws, image);
}
bool ScrewDetector::processImage(std::vector<cv::Vec3f>& circles, cv::Vec2f& roi_reference, cv::Mat& roi,
    std::vector<bool>& are_screws, const cv::Mat& image)
{
    //Get single screw rois
    std::vector<cv::Mat> rois;
    try {
        rois = roi_extractor_.getScrewImages(circles, roi_reference, roi, image);
    } catch (const std::runtime_error& ex) {
        std::cout << ex.what() << std::endl;
        return false;
    }

    //Predict for each gotten single screw roi
    ml_model_.predict(are_screws, rois, circles, roi_reference);

    return true;
}

void ScrewDetector::setParameters(const ExtractorParameters& parameters)
{
    (&roi_extractor_)->~ROIExtractor();
    new (&roi_extractor_) ROIExtractor(parameters);
}

}
