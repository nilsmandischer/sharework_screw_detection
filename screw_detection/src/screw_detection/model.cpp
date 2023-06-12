#include <screw_detection/model.h>

namespace screw_detection {

ScrewDetectorModel::ScrewDetectorModel(const ModelType type)
    : model_type_(type)
{
    switch (model_type_) {
    case Forest:
        forest_ = cv::ml::RTrees::create();
        return;
    case AdaBoost:
        adaboost_ = cv::ml::Boost::create();
        return;
    case SVM:
        svm_ = cv::ml::SVM::create();
        return;
    }
    throw std::logic_error("No function to create Model for specified ModelType!");
}

void ScrewDetectorModel::setParameters(const ModelParameters& parameters)
{
    //For parameter references check out the corresponding OpenCV documentation
    switch (model_type_) {
    case Forest: {
        forest_->setMaxDepth(parameters.max_depth);
        forest_->setMinSampleCount(parameters.min_sample_count);
        forest_->setRegressionAccuracy(parameters.regression_accuracy);
        forest_->setUseSurrogates(parameters.use_surrogates);
        forest_->setMaxCategories(parameters.CATEGORY_AMOUNT);
        forest_->setCVFolds(parameters.cv_folds);
        forest_->setUse1SERule(parameters.use_1se_rule);
        forest_->setTruncatePrunedTree(parameters.truncate_pruned_tree);
        cv::Mat priors = cv::Mat::ones(2, 1, CV_64F);
        priors.at<double>(1) = parameters.weight_screws;
        forest_->setPriors(priors);

        forest_->setCalculateVarImportance(true);
        forest_->setActiveVarCount(parameters.active_var_count);
        forest_->setTermCriteria(cv::TermCriteria(parameters.term_criteria, parameters.max_iterations, 0));
        return;
    }
    case AdaBoost: {
        adaboost_->setMaxDepth(parameters.max_depth);
        adaboost_->setMinSampleCount(parameters.min_sample_count);
        adaboost_->setRegressionAccuracy(parameters.regression_accuracy);
        adaboost_->setUseSurrogates(parameters.use_surrogates);
        adaboost_->setMaxCategories(parameters.CATEGORY_AMOUNT);
        adaboost_->setCVFolds(parameters.cv_folds);
        adaboost_->setUse1SERule(parameters.use_1se_rule);
        adaboost_->setTruncatePrunedTree(parameters.truncate_pruned_tree);
        cv::Mat priors = cv::Mat::ones(2, 1, CV_64F);
        priors.at<double>(1) = parameters.weight_screws;
        adaboost_->setPriors(priors);
        adaboost_->setWeakCount(parameters.weak_count);
        return;
    }
    case SVM: {
        svm_->setType(parameters.svm_type);
        //svm_->setNu(0); //Smoothness of decision boundary
        svm_->setKernel(parameters.svm_kernel);
        svm_->setTermCriteria(cv::TermCriteria(parameters.term_criteria, parameters.max_iterations, 0));
        return;
    }
    }
    throw std::logic_error("No function to set parameters for specified ModelType!");
}

void ScrewDetectorModel::calculateFeatures(std::vector<std::vector<float>>& screw_values, const std::vector<cv::Mat>& rois,
    const std::vector<cv::Vec3f>& circles, const cv::Vec2f& roi_reference)
{
    cv::Scalar foreground(255, 255, 255);
    cv::Scalar background(0, 0, 0);
    for (uint im_id = 0; im_id != rois.size(); im_id++) {
        cv::Vec3f circle = circles[im_id];

        std::vector<float> current_circle_values;

        //Getting current_roi
        cv::Mat current_roi;
        if (rois[im_id].channels() != 1) {
            cv::cvtColor(rois[im_id], current_roi, cv::COLOR_BGR2GRAY);
        } else {
            current_roi = rois[im_id];
        }

        //Average color-----------------------------------
        cv::Mat roi_mask = cv::Mat::zeros(current_roi.rows, current_roi.cols, CV_8U);
        cv::circle(roi_mask, cv::Point(roi_mask.cols / 2, roi_mask.rows / 2), circle[2], foreground, -1, 8, 0);

        cv::Scalar total_mean = cv::mean(current_roi, roi_mask);
        float screw_color = total_mean.val[0];
        current_circle_values.push_back(screw_color);

        //Bright center spot----------------------------------
        cv::Mat mask_spot = cv::Mat::zeros(current_roi.rows, current_roi.cols, CV_8U);
        cv::circle(mask_spot, cv::Point(mask_spot.cols / 2, mask_spot.rows / 2),
            circle[2] * CENTER_SPOT_RADIUS_PERCENT_, foreground, -1, 8, 0);

        cv::Scalar mean_spot = cv::mean(current_roi, mask_spot);
        float screw_color_spot = mean_spot.val[0];
        current_circle_values.push_back(screw_color_spot);

        //Bright ring without screw------------------------------
        cv::Mat ring_mask = roi_mask.clone();
        cv::circle(ring_mask, cv::Point(ring_mask.cols / 2, ring_mask.rows / 2),
            circle[2] * BRIGHT_RING_RADIUS_PERCENT_,
            background, -1, 8, 0);
        cv::Scalar outer_ring_mean = cv::mean(current_roi, ring_mask);
        float outer_ring_color = outer_ring_mean.val[0];
        current_circle_values.push_back(outer_ring_color);

        //Average outer color to adjust for brightness-----------------
        cv::Mat outer_mask = 255 - roi_mask;
        cv::Scalar total_outer_mean = cv::mean(current_roi, outer_mask);
        float metal_color = total_outer_mean.val[0];

        current_circle_values.push_back(screw_color - metal_color);
        current_circle_values.push_back(screw_color_spot - metal_color);
        current_circle_values.push_back(outer_ring_color - metal_color);

        //Length of bright outer ring ---------------------------
        //Direction through ROI
        cv::Vec2f circle_diff_vec(circle[0] - roi_reference[0], circle[1] - roi_reference[1]);
        cv::Point2f circle_diff = cv::Point(circle_diff_vec[0], circle_diff_vec[1]);

        double length = std::sqrt(std::pow(circle_diff.x, 2) + std::pow(circle_diff.y, 2));
        circle_diff.y = circle_diff.y / length; //Normalize direction
        circle_diff.x = circle_diff.x / length;

        //Construct start and end starting at center
        cv::Point2f start = cv::Point2f(current_roi.cols / 2., current_roi.rows / 2.) - circle_diff * circle[2];
        cv::Point end = cv::Point2f(current_roi.cols / 2., current_roi.rows / 2.) + circle_diff * circle[2];

        int max_steps;
        //To ensure always at least one index is moved by 1 when raytracing
        if (std::abs(circle_diff.x) > std::abs(circle_diff.y)) {
            float abs_x = std::abs(circle_diff.x);
            circle_diff.y = circle_diff.y / abs_x;
            circle_diff.x = circle_diff.x / abs_x;

            //To compensate for increased/decreased step size through longer/shorter circle_diff
            max_steps = circle[2] * 2 * abs_x;
        } else {
            float abs_y = std::abs(circle_diff.y);
            circle_diff.x = circle_diff.x / abs_y;
            circle_diff.y = circle_diff.y / abs_y;

            max_steps = circle[2] * 2 * abs_y;
        }

        float plateau_length = 0;
        float num_plateaus = 1;

        //Set initial prev point to start a plateau right away with setting it to same val as start
        //-> start is not an outcast
        cv::Scalar prev_point_vec = current_roi.at<uchar>(start);
        float prev_point = prev_point_vec.val[0];

        int consecutive_points = 0; //Used to track outcasts and start new plateau counter
        for (int steps = 1; steps != max_steps; steps++) {
            cv::Point new_point = start + steps * circle_diff; //advance point

            //Get grayscale of new point
            cv::Scalar curr_point_vec = current_roi.at<uchar>(new_point);
            float curr_point = curr_point_vec.val[0];

            //If brighter than average screw brightness, add one to plateuau length
            if (curr_point >= screw_color) {
                plateau_length++;
            }

            //Get diff between brightness of current point and previous point which was not an outcast
            float delta = curr_point - prev_point;

            if (std::abs(delta) > PLATEUS_OUTLIER_TOLERANCE_) {
                //Add one to counter of outcasts, dont change prev_point
                consecutive_points++;

                if (consecutive_points == NUM_OUTLIERS_NEW_PLATEU_) { //If 4 Outcasts found a new plateau is started
                    num_plateaus++;
                    consecutive_points = 0;
                    prev_point = curr_point;
                }
            } else {
                prev_point = curr_point;
            }
        }

        current_circle_values.push_back(plateau_length);
        current_circle_values.push_back(num_plateaus);

        //std::vector<float> hog_values;
        //calculateHOGFeatures(hog_values, current_roi);
        //current_circle_values.insert(current_circle_values.end(), hog_values.begin(), hog_values.end());

        screw_values.push_back(current_circle_values);
    }

    return;
}

void ScrewDetectorModel::predict(std::vector<bool>& are_screws, const std::vector<cv::Mat>& rois,
    const std::vector<cv::Vec3f>& circles, const cv::Vec2f& roi_reference)
{
    //Calculate all feature values
    std::vector<std::vector<float>> screw_values;
    calculateFeatures(screw_values, rois, circles, roi_reference);

    //To not create a new object each iteration
    cv::Mat current_screw(1, screw_values[0].size(), CV_32FC1);

    //For each calculated feature row
    for (uint s_id = 0; s_id != screw_values.size(); s_id++) {
        //Get values
        for (uint f_id = 0; f_id != screw_values[s_id].size(); f_id++) {
            current_screw.at<float>(0, f_id) = screw_values[s_id][f_id];
        }

        //Predict
        float result = predictCV(current_screw);

        //Check result
        if (result == 0) {
            are_screws.push_back(false);
        } else if (result == 1) {
            are_screws.push_back(true);
        } else {
            std::cout << "\e[1;35m Unknown case detected!\e[0m" << std::endl;
        }
    }
}

void ScrewDetectorModel::calculateHOGFeatures(std::vector<float>& feature_values, const cv::Mat& image, const bool use_flip)
{
    //Experimental
    cv::HOGDescriptor hog(
        cv::Size(20, 20), //winSize
        cv::Size(10, 10), //blocksize
        cv::Size(5, 5), //blockStride,
        cv::Size(10, 10), //cellSize,
        9, //nbins,
        1, //derivAper,
        -1, //winSigma,
        cv::HOGDescriptor::L2Hys, //histogramNormType,
        0.2, //L2HysThresh,
        1, //gammal correction,
        64, //nlevels=64
        1); //Use signed gradients

    cv::Size win_size = cv::Size(image.cols, image.rows);
    hog.blockStride = win_size / 8;
    hog.winSize = hog.blockStride * 8;
    hog.blockSize = hog.blockStride * 2;
    hog.cellSize = hog.blockStride * 2;

    cv::Mat gray;
    //Make sure the image is bigger than the HOG window size
    if (image.cols >= win_size.width && image.rows >= win_size.height) {
        //Mask the image to the HOG window size
        cv::Rect r = cv::Rect((image.cols - win_size.width) / 2,
            (image.rows - win_size.height) / 2,
            win_size.width,
            win_size.height);

        //Convert image type if necessary
        if (image.channels() != 1) {
            cv::cvtColor(image(r), gray, cv::COLOR_BGR2GRAY);
        } else {
            gray = image.clone();
        }

        hog.compute(gray, feature_values);

        //Flipping the image can improve HOG performance but obvsly doubles the computation time
        if (use_flip) {
            std::vector<float> flipped_values;
            cv::flip(gray, gray, 1);
            hog.compute(gray, flipped_values, cv::Size(8, 8), cv::Size(0, 0));
            feature_values.insert(feature_values.end(), flipped_values.begin(), flipped_values.end());
        }
    }
    return;
}

void ScrewDetectorModel::write(const std::string& path)
{
    std::cout << "Writing model data to " + path << std::endl;
    cv::FileStorage storage(path, 1);
    switch (model_type_) {
    case Forest:
        forest_->write(storage);
        return;
    case AdaBoost:
        adaboost_->write(storage);
        return;
    case SVM:
        svm_->write(storage);
        return;
    }
    throw std::logic_error("No function to write model to disk for specified ModelType!");
}

void ScrewDetectorModel::read(const std::string& path)
{
    std::cout << "Reading model data from " + path << std::endl;
    cv::FileStorage storage(path, 0);

    switch (model_type_) {
    case Forest:
        forest_->read(storage.root());
        return;
    case AdaBoost:
        adaboost_->read(storage.root());
        return;
    case SVM:
        svm_->read(storage.root());
        return;
    }
    throw std::logic_error("No function to read model from disk for specified ModelType!");
}

void ScrewDetectorModel::train(const cv::Mat& trainer_data, const cv::Mat& result_data)
{
    switch (model_type_) {
    case Forest:
        forest_->train(cv::ml::TrainData::create(trainer_data, cv::ml::ROW_SAMPLE, result_data));
        return;
    case AdaBoost:
        adaboost_->train(cv::ml::TrainData::create(trainer_data, cv::ml::ROW_SAMPLE, result_data));
        return;
    case SVM:
        svm_->train(cv::ml::TrainData::create(trainer_data, cv::ml::ROW_SAMPLE, result_data));
        return;
    }
    throw std::logic_error("No function to train model for specified ModelType!");
}

float ScrewDetectorModel::predictCV(const cv::Mat& sample)
{
    switch (model_type_) {
    case Forest:
        return forest_->predict(sample);
    case AdaBoost:
        return adaboost_->predict(sample);
    case SVM:
        return svm_->predict(sample);
    }
    throw std::logic_error("No function to predict for specified ModelType!");
}

void ScrewDetectorModel::getModelInfo(void)
{
    switch (model_type_) {
    case Forest: {
        //Decision power of each feature
        cv::Mat variable_importance = forest_->getVarImportance();

        std::cout << "Estimated variable importance" << std::endl;
        for (int i = 0; i < variable_importance.rows; i++) {
            std::cout << "Variable " << i << ": " << variable_importance.at<float>(i, 0) << std::endl;
        }
        return;
    }
    case AdaBoost: {
        //Check all splits of trees that were selected to have meaning
        std::vector<cv::ml::DTrees::Split> splits = adaboost_->getSplits();

        std::vector<int> used_indices; //store all feature indices that produced a tree split
        for (int i = 0; i != splits.size(); i++) {
            int index = splits[i].varIdx;
            // Only add features that are not already added
            if (std::find(used_indices.begin(), used_indices.end(), index) == used_indices.end()) {
                used_indices.push_back(index);
            }
        }

        int weak_count = adaboost_->getWeakCount();
        std::cout << "A total of " + std::to_string(weak_count) + " features were trimmed!" << std::endl;
        std::cout << "The following features were used:" << std::endl;
        std::cout << "Feature ";
        for (int i = 0; i < used_indices.size(); i++) {
            std::cout << used_indices[i] << ", ";
        }
        std::cout << std::endl;
        return;
    }
    case SVM:
        //TBC
        svm_->getSupportVectors();
        throw std::logic_error("No function to get Model Info for SVM!");
    }
    throw std::logic_error("No function to get Model Info for specified ModelType!");
}
}
