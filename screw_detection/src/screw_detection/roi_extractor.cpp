#include <screw_detection/roi_extractor.h>

namespace screw_detection {

ROIExtractor::ROIExtractor(const ExtractorParameters& params)
    : parameters_(params)
{
    if (!parameters_.isInitiated()) {
        throw std::logic_error("ExtractorParameters have to be initialized before passing them into the ROIExtractor!");
    }
}

std::vector<cv::Mat> ROIExtractor::getScrewImages(std::vector<cv::Vec3f>& circles, cv::Vec2f& roi_reference,
    const cv::Mat& image, const int extend_screw_image)
{
    cv::Mat roi;
    return getScrewImages(circles, roi_reference, roi, image, extend_screw_image);
}

std::vector<cv::Mat> ROIExtractor::getScrewImages(std::vector<cv::Vec3f>& circles, cv::Vec2f& roi_reference,
    cv::Mat& roi, const cv::Mat& image, const int extend_screw_image)
{
    //Get all circles inside the image
    circles.clear();
    getScrewsInROI(circles, roi_reference, roi, image);

    //Cut the single screw image out of the original image
    std::vector<cv::Mat> rois;
    for (int c_id = 0; c_id != circles.size(); c_id++) {
        cv::Vec3f circle = circles[c_id];
        try {
            cv::Mat current_roi = image(cv::Range(circle[1] - circle[2] - extend_screw_image,
                                            circle[1] + circle[2] + 1 + extend_screw_image),
                cv::Range(circle[0] - circle[2] - extend_screw_image,
                    circle[0] + circle[2] + 1 + extend_screw_image));
            rois.push_back(current_roi);
        } catch (cv::Exception cv_ex) {
            //If the circle is partly outside the ROI this will be triggered
            //the ROI is expanded to compensate so it shouldnt happen
            circles.erase(circles.begin() + c_id);
            c_id--;
            continue;
        }
    }
    if (rois.size() != circles.size()) {
        throw std::logic_error("Not equal amount of circles and images?!");
    }
    return rois;
}

void ROIExtractor::getScrewsInROI(std::vector<cv::Vec3f>& circles, cv::Vec2f& roi_reference, cv::Mat& roi,
    const cv::Mat& image)
{
    roi = parameters_.roi_parameters()->getROI(roi_reference, image,
        parameters_.screw_parameters().max_circle_size());

    getScrewCircles(circles, roi);
    if (circles.size() == 0) {
        throw std::runtime_error("\e[1;31m No circles found!\e[0m");
    }
    return;
}

void ROIExtractor::getScrewCircles(std::vector<cv::Vec3f>& circles, const cv::Mat& image)
{
    roi_utils::getCirclesHough(circles, image, parameters_.screw_parameters());
}

namespace roi_utils {
    void getCirclesHough(std::vector<cv::Vec3f>& circles, const cv::Mat& image, const HoughParameters& parameters)
    {
        getCirclesHough(circles, image, parameters.contour_threshold(), parameters.accumulator_threshold(),
            parameters.min_circle_size(), parameters.max_circle_size(), parameters.min_circle_distance());
        return;
    }

    void getCirclesHough(std::vector<cv::Vec3f>& circles, const cv::Mat& image, const int contour_threshold,
        const int accumulator_threshold, const int min_circle_size, const int max_circle_size,
        const int min_circle_distance)
    {
        cv::Mat image_gray;
        if (image.channels() != 1) {
            cv::cvtColor(image, image_gray, cv::COLOR_BGR2GRAY);
        } else {
            image_gray = image.clone();
        }

        //Blur image
        cv::blur(image_gray, image_gray, cv::Size(BLUR_SIDE_LENGTH, BLUR_SIDE_LENGTH));

        cv::HoughCircles(image_gray, circles, cv::HOUGH_GRADIENT, 1, min_circle_distance,
            contour_threshold, accumulator_threshold, min_circle_size, max_circle_size);
        return;
    }
}
}
