/**
 * @file
 * @brief This file contains the declaration of the ROIExtractor CustomROI abstract class and ExtractorParameters,
 * HoughParameters and ImageParameters.
 *
 * @author Sebastian Döbler
 * @version 1.0
 */

#ifndef SCREW_DETECTION_ROI_EXTRATOR_PARAMETERS_H
#define SCREW_DETECTION_ROI_EXTRATOR_PARAMETERS_H

//STD
#include <algorithm>
#include <stdexcept>

//CV
#include <opencv2/opencv.hpp>

namespace screw_detection {
typedef int Pixel;
typedef float CM;

/** HoughStyle
 * @brief implies conversion of tolerances for Hough Circle Detection
*/
enum HoughStyle { Single,
    Multiple };

/** CustomROI
 * @brief is a pure abstract class. Used to enable the extraction of a custom ROI inside the ROIExtractor class
*/
class CustomROI {
public:
    /**
     * @brief Returns the masked out ROI of the passed image.
     * @param reference Return a reference point of the roi. Preferably the center of all screws.
     * Used to calculate feature values inside the ScrewDetectionModel class.
     * @param max_circle_size Use this to extend the returned ROI by the estimated maximum circle size from the 
     * calculated ExtractorParameters.screw_parameters.max_circle_size(). 
     * This ensures all circles are fully inside the returned ROI.
    */
    virtual cv::Mat getROI(cv::Vec2f& reference, const cv::Mat& image, const Pixel& max_circle_size) = 0;

    /**
     * @brief This function is called together with ExtractorParameters.initiateParameters(). 
     * Set initiated_ = true after your initialization has been completed to allow for the CustomROI to be used.
     * @param ppc Use the pixel_per_centimeter(ppc) to calculate parameters necessary for your CustomROI
    */
    virtual void initiateParameters(const float& ppc) = 0;

    /**
     * @brief Used to make sure the CustomROI has been initiated before being used.
    */
    const bool& isInitiated() const { return this->initiated_; }

private:
    //Status
    bool initiated_ = false;
};

/** ImageParameters
 * @brief are used to contain necessary parameters related to the expected image
*/
class ImageParameters {
public:
    /**
    * @brief Requires height and angle to calculate the ppc.
    * @param camera_height the height perpendicular to the plane in which the screws are expected
    * @param camera_angle horizontal opening angle of the used camera
    */
    ImageParameters(const CM& camera_height, const float& camera_angle)
    {
        this->camera_height_ = camera_height;
        this->camera_angle_ = camera_angle;
    }
    /**
     * @brief Same as basic constructor, but also calls setImageSize()
    */
    ImageParameters(const CM& camera_height, const float& camera_angle, const Pixel height, const Pixel width)
    {
        this->camera_height_ = camera_height;
        this->camera_angle_ = camera_angle;
        setImageSize(height, width);
    }

    /**
     * @brief initiates the ImageParameters by calculating the ppc from the image metrics
    */
    void setImageSize(const Pixel height, const Pixel width)
    {
        this->image_width_ = width;
        this->image_height_ = height;
        calculatePPC();
        initiated_ = true;
    }

    //Bunch of getters which may throw std::logic_error if the ImageParameters have not been initiated
    const CM& camera_height() const { return this->camera_height_; }
    const float& camera_angle() const { return this->camera_angle_; }
    const Pixel& image_width() const
    {
        if (!initiated_) {
            throw std::logic_error("The ImageParameters were not initiated, thus this variable is not availabl!");
        }
        return this->image_width_;
    }
    const Pixel& image_height() const
    {
        if (!initiated_) {
            throw std::logic_error("The ImageParameters were not initiated, thus this variable is not availabl!");
        }
        return this->image_height_;
    }
    const float ppc() const
    {
        if (!initiated_) {
            throw std::logic_error("The ImageParameters were not initiated, thus this variable is not availabl!");
        }
        return this->ppc_;
    }

    //Operator

    ImageParameters& operator=(ImageParameters parameters)
    {
        std::swap(*this, parameters);
        return *this;
    }

private:
    //Status
    bool initiated_ = false;
    //Calculated Image Parameters
    float ppc_; // Pixel per centimeter

    //Raw Image Parameters
    CM camera_height_;
    float camera_angle_;
    Pixel image_width_;
    Pixel image_height_;

    void calculatePPC()
    {
        ppc_ = image_width_ / (2 * tan(camera_angle_ * 2 * M_PI / 360 / 2) * camera_height_);
    }
};

/** HoughParameters
 * @brief contains parameters for cv::HoughCircles, which are derived from real world measurements and the ppc
*/
class HoughParameters {
public:
    /**
     * @param circle_size the diameter of the expected circles in [cm]
     * @param accumulator_threshold Threshold for center detection
     * @param contour_threshold Threshold for canny edge detection
    */
    HoughParameters(const CM& circle_size, const int accumulator_threshold, const int contour_threshold)
    {
        this->circle_size_ = circle_size;
        this->accumulator_threshold_ = accumulator_threshold;
        this->contour_threshold_ = contour_threshold;
    }

    /**
     * @brief used to change the size of the expected circles and automaticly reinitialize the parameters 
     * if they were already initialized
    */
    void setCircleSize(const CM& size)
    {
        this->circle_size_ = size;
        if (initiated_) {
            initiateParameters(ppc_, style_);
        }
        return;
    }

    /**
     * @brief Calculates the tolerances depending on the HoughStyle
    */
    void initiateParameters(const float& ppc, const HoughStyle style)
    {
        switch (style) {
        case Single: {
            min_circle_distance_ = INT16_MAX;

            circle_size_px_ = circle_size_ * ppc;
            min_circle_size_ = circle_size_px_ * SIZE_TO_MIN_SIZE_SINGLE;
            max_circle_size_ = circle_size_px_ * SIZE_TO_MAX_SIZE_SINGLE;
            break;
        }
        case Multiple: {
            min_circle_distance_ = circle_size_ * ppc * SIZE_TO_DISTANCE;

            circle_size_px_ = circle_size_ * ppc;
            min_circle_size_ = circle_size_px_ * SIZE_TO_MIN_SIZE_MULTIPLE;
            max_circle_size_ = circle_size_px_ * SIZE_TO_MAX_SIZE_MULTIPLE;
            break;
        }
        default:
            throw std::invalid_argument("Passed HoughStyle does not have a way to calculate Parameters!");
        }
        initiated_ = true;
        this->ppc_ = ppc;
        this->style_ = style;
        return;
    }

    //Misc

    const Pixel& min_circle_size() const
    {
        if (!initiated_) {
            throw std::logic_error("The HoughParameters were not initiated, thus this variable is not availabl!");
        }
        return this->min_circle_size_;
    }
    const Pixel& max_circle_size() const
    {
        if (!initiated_) {
            throw std::logic_error("The HoughParameters were not initiated, thus this variable is not availabl!");
        }
        return this->max_circle_size_;
    }
    const Pixel& min_circle_distance() const
    {
        if (!initiated_) {
            throw std::logic_error("The HoughParameters were not initiated, thus this variable is not availabl!");
        }
        return this->min_circle_distance_;
    }
    const CM& circle_size() const { return this->circle_size_; }
    const Pixel& circle_size_px() const { return this->circle_size_px_; }

    const int& accumulator_threshold() const { return this->accumulator_threshold_; }
    const int& contour_threshold() const { return this->contour_threshold_; }

    HoughParameters& operator=(HoughParameters parameters)
    {
        std::swap(*this, parameters);
        return *this;
    }

private:
    //Storage
    bool initiated_ = false;
    float ppc_;
    HoughStyle style_;

    //Constants
    //Multiple
    const float SIZE_TO_DISTANCE = 5;
    const float SIZE_TO_MIN_SIZE_MULTIPLE = 0.8;
    const float SIZE_TO_MAX_SIZE_MULTIPLE = 1.2;
    //Single
    const float SIZE_TO_MIN_SIZE_SINGLE = 0.98;
    const float SIZE_TO_MAX_SIZE_SINGLE = 1.02;

    //Processed Parameters
    Pixel circle_size_px_;
    Pixel min_circle_size_;
    Pixel max_circle_size_;
    Pixel min_circle_distance_;

    //Raw Parameters
    CM circle_size_;
    int accumulator_threshold_;
    int contour_threshold_;
};

/**
 * @brief ExtractorParameters
 * holds all external parameters used by ROIExtractor
 */
class ExtractorParameters {
public:
    /**
     * @param screw_parameters HoughParameters used to detect screws inside the ROI
     * @param image_parameters ImageParameters that hold basic image information
     * @param custom_roi A ROI inside each image which holds the screws
    */
    ExtractorParameters(const HoughParameters& screw_parameters, const ImageParameters& image_parameters,
        std::shared_ptr<CustomROI> custom_roi)
        : screw_parameters_(screw_parameters)
        , image_parameters_(image_parameters)
        , custom_roi_(custom_roi)
    {
    }

    void initiateParameters(const Pixel image_height, const Pixel image_width)
    {
        //Image Parameters
        image_parameters_.setImageSize(image_height, image_width);
        float ppc = image_parameters_.ppc();

        //Hough Parameters
        screw_parameters_.initiateParameters(ppc, Multiple);

        //Custom ROI
        custom_roi_->initiateParameters(ppc);

        initiated_ = true;
    }

    //Misc

    const HoughParameters& screw_parameters() const { return this->screw_parameters_; }
    const ImageParameters& image_parameters() const { return this->image_parameters_; }
    const std::shared_ptr<CustomROI> roi_parameters() const { return this->custom_roi_; }

    const bool& isInitiated() const { return this->initiated_; }

    ExtractorParameters& operator=(ExtractorParameters parameters)
    {
        std::swap(*this, parameters);
        return *this;
    }

private:
    //Status
    bool initiated_ = false;

    //Circle detection Hough
    HoughParameters screw_parameters_;

    //Image
    ImageParameters image_parameters_;

    //Custom ROI
    std::shared_ptr<CustomROI> custom_roi_;
};
}
#endif
