#include <screw_detection/training.h>

namespace screw_detection {

ScrewTrainer::ScrewTrainer(ExtractorParameters& extractor_params, TrainerParameters& trainer_params,
    ModelParameters& model_parameters)
    : ml_model_(USED_MODEL_TYPE)
    , extractor_parameters_(extractor_params)
{
    ml_model_.setParameters(model_parameters);
    trainer_params_ = trainer_params;
}

void ScrewTrainer::spliceImages(void)
{
    loadFullImages();

    //Parameters are initiated inside loadFullImages() thus the ROIExtractor needs to be created here
    extractor_.reset(new ROIExtractor(extractor_parameters_));

    saveCutData();

    std::cout << "Finished splicing images. If desired check the created image folders." << std::endl;
    std::cout << "Proceed with running the Model trainer with save_cut_training_data = false" << std::endl;
}

void ScrewTrainer::trainModel(void)
{
    loadSplicedImages();

    //Parameters are initiated inside loadSplicedImages() thus the ROIExtractor needs to be created here
    extractor_.reset(new ROIExtractor(extractor_parameters_));

    cv::Mat trainer_data;
    cv::Mat result_data;

    train(trainer_data, result_data);

    test(trainer_data, result_data);

    std::string tree_path = trainer_params_.image_path + MODEL_FILE_NAME;
    ml_model_.write(tree_path);
}

void ScrewTrainer::run()
{
    spliceImages();
    trainModel();
}

void ScrewTrainer::saveCutData()
{
    //Delets previously backed up data, backs up most recent data
    std::string screws_images_path = trainer_params_.image_path + SCREWS_FOLDER;
    std::string holes_images_path = trainer_params_.image_path + HOLES_FOLDER;
    std::string unsorted_images_path = trainer_params_.image_path + UNSORTED_FOLDER;

    std::cout << "Removing previously backed up data" << std::endl;

    if (std::filesystem::exists(screws_images_path + OLD_SUFFIX)) {
        std::filesystem::remove_all(screws_images_path + OLD_SUFFIX);
    }

    if (std::filesystem::exists(holes_images_path + OLD_SUFFIX)) {
        std::filesystem::remove_all(holes_images_path + OLD_SUFFIX);
    }

    if (std::filesystem::exists(unsorted_images_path + OLD_SUFFIX)) {
        std::filesystem::remove_all(unsorted_images_path + OLD_SUFFIX);
    }
    std::cout << "Backing up previous data" << std::endl;
    try {
        if (std::filesystem::exists(screws_images_path)) {
            std::filesystem::rename(screws_images_path, screws_images_path + OLD_SUFFIX);
        }
    } catch (const std::exception& exc) {
        std::cout << "Backup failed for screw samples (ERROR: " << exc.what() << ")" << std::endl;
    }

    try {
        if (std::filesystem::exists(holes_images_path)) {
            std::filesystem::rename(holes_images_path, holes_images_path + OLD_SUFFIX);
        }
    } catch (const std::exception& exc) {
        std::cout << "Backup failed for hole samples (ERROR: " << exc.what() << ")" << std::endl;
    }

    try {
        if (std::filesystem::exists(unsorted_images_path)) {
            std::filesystem::rename(unsorted_images_path, unsorted_images_path + OLD_SUFFIX);
        }
    } catch (const std::exception& exc) {
        std::cout << "Backup failed for unsorted samples (ERROR: " << exc.what() << ")" << std::endl;
    }

    //To save the current data
    std::filesystem::create_directory(screws_images_path);
    std::filesystem::create_directory(holes_images_path);
    std::filesystem::create_directory(unsorted_images_path);

    std::cout << "Getting all circles" << std::endl;
    std::vector<cv::Mat> all_circles;
    std::vector<cv::Vec2f> all_roi_references;
    //For each loaded full image
    for (uint d_id = 0; d_id != data_.size(); d_id++) {
        //Get each single screw image
        std::vector<cv::Vec3f> current_circles;
        cv::Vec2f current_roi_reference;
        std::vector<cv::Mat> screw_images;
        try {
            screw_images = extractor_->getScrewImages(current_circles, current_roi_reference, data_[d_id].image,
                EXTEND_SCREW_IMAGES_);
        } catch (const std::runtime_error& er) {
            std::cout << "Skipping input " + std::to_string(d_id) + " because no ROI was found!" << std::endl;
            continue;
        }
        //Save the circle and the reference coordinates
        for (uint s_id = 0; s_id != screw_images.size(); s_id++) {
            cv::Vec3f circle = current_circles[s_id];

            all_circles.push_back(screw_images[s_id]);
            cv::Vec2f diff = cv::Vec2f(current_circles[s_id][0] - current_roi_reference[0],
                current_circles[s_id][1] - current_roi_reference[1]);
            all_roi_references.push_back(diff);
        }
    }
    std::cout << "Found " + std::to_string(all_circles.size()) + " circles " << std::endl;

    //Shows each single screw image and waits for user input to answer if
    //the image shows a screw or empty hole
    std::string window_name = "Screw or Hole?";
    cv::namedWindow(window_name, cv::WINDOW_FULLSCREEN);

    std::cout << "Controls" << std::endl;
    std::cout << "   If image shows " << std::endl;
    std::cout << "      a screw press: " << static_cast<char>(BUTTON_SCREW) << std::endl;
    std::cout << "      a hole press: " << static_cast<char>(BUTTON_HOLE) << std::endl;
    std::cout << "   Press " << static_cast<char>(BUTTON_RETURN) << " to return to the last image" << std::endl;
    std::cout << "   Press " << static_cast<char>(BUTTON_SKIP) << " to skip the current image" << std::endl;
    std::cout << "   Press " << static_cast<char>(BUTTON_QUIT) << " to quit the process" << std::endl;

    //Save circles
    for (uint im_id = 0; im_id != all_circles.size();) {
        //Make image uniform size
        cv::Mat image;
        cv::resize(all_circles[im_id], image, cv::Size(TRAINER_IMAGE_SIZE_, TRAINER_IMAGE_SIZE_), 0, 0,
            CV_IMAGE_INTERPOLATION_TYPE_);
        //Show and wait for inout
        cv::imshow(window_name, image);

        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        std::string circle_name = CIRCLE_FOLDER_PREFIX + std::to_string(im_id) + FILE_TYPE_SUFFIX;

        int key = cv::waitKey(0);

        //Act depending on pressed button
        if (key == BUTTON_SCREW) {
            std::cout << "Image " + std::to_string(im_id) + " is screw" << std::endl;
            cv::imwrite(screws_images_path + circle_name, all_circles[im_id]);
            im_id++;
        } else if (key == BUTTON_HOLE) {
            std::cout << "Image " + std::to_string(im_id) + " is hole" << std::endl;
            cv::imwrite(holes_images_path + circle_name, all_circles[im_id]);
            im_id++;
        } else if (key == BUTTON_SKIP) {
            std::cout << "Skipping image " + std::to_string(im_id) << std::endl;
            cv::imwrite(unsorted_images_path + circle_name, all_circles[im_id]);
            im_id++;
        } else if (key == BUTTON_RETURN) {
            if (im_id == 0) {
                std::cout << "Can not go back another ID, already at image 0!" << std::endl;
                continue;
            }
            im_id--;

            std::cout << "Going back one image, currently at image " + std::to_string(im_id) << std::endl;

            circle_name = "/circle_" + std::to_string(im_id) + FILE_TYPE_SUFFIX;
            std::filesystem::remove(screws_images_path + circle_name);
            std::filesystem::remove(holes_images_path + circle_name);
            std::filesystem::remove(unsorted_images_path + circle_name);
        } else if (key == BUTTON_QUIT) {
            std::cout << "Press " << static_cast<char>(BUTTON_QUIT) << " again to quit" << std::endl;
            std::cout << "Press any other button to return" << std::endl;
            int key_q = cv::waitKey(0);
            if (key_q == BUTTON_QUIT) {
                std::cout << "Quitting" << std::endl;
                break;
            } else {
                std::cout << "Continuing" << std::endl;
            }
        } else {
            if (key != -1) {
                std::cout << "Button unknown (ID " << std::to_string(key).c_str() << ")!" << std::endl;
            }
        }
    }
    cv::destroyWindow(window_name);

    //Save references
    std::string misc_path = trainer_params_.image_path + MISC_FILE_NAME;
    std::cout << "Writing misc data to " + misc_path << std::endl;
    std::ofstream out_file(misc_path);
    out_file << std::to_string(data_[0].image.cols) << "\n";
    out_file << std::to_string(data_[0].image.rows) << "\n";
    // the important part
    for (const cv::Vec2f& e : all_roi_references) {
        std::string out_string = std::to_string(e[0]) + "," + std::to_string(e[1]);
        out_file << out_string << "\n";
    }
    return;
}

void ScrewTrainer::test(const cv::Mat& trainer_data, const cv::Mat& result_data)
{
    //Checking for all training cases if the
    std::cout << "Testing trained model with training data" << std::endl;
    uint error_count = 0;
    for (uint t_id = 0; t_id != trainer_data.rows; t_id++) {
        float guess = ml_model_.predictCV(trainer_data.row(t_id));
        if (guess != result_data.at<int>(t_id)) {
            error_count++;
        }
    }
    std::cout << "Failed to predict " + std::to_string(error_count) + " out of "
            + std::to_string(trainer_data.rows) + " cases."
              << std::endl;
    if (error_count != 0) {
        std::cout << "The error count should always be zero, seems like somethign went wrong?"
                  << std::endl;
    }

    //Catch unimplemented function version cause not critical
    try {
        ml_model_.getModelInfo();
    } catch (const std::exception& ex) {
        std::cout << ex.what();
        return;
    }
    return;
}

void ScrewTrainer::train(cv::Mat& trainer_data, cv::Mat& result_data)
{
    std::vector<std::vector<std::vector<float>>> all_features;

    std::cout << "Getting all features" << std::endl;
    for (uint d_id = 0; d_id != data_.size(); d_id++) {
        //Get Circles
        std::vector<cv::Vec3f> circles;
        extractor_->getScrewCircles(circles, data_[d_id].image);
        if (circles.size() == 0) {
            continue;
        }

        //Calculate feature values
        std::vector<std::vector<float>> screw_values;
        /*roi_reference is saved as circle - roi_reference. calculateFeatures() calculates circle - roi_reference again, 
        but this time with a local screw coordinate because cropped images are used.
        To compensate for 
        (circle_wrong - (circle - roi_reference)) != circle - roi_reference 
        multiply data_.roi_reference := circle-roi_reference with -1 
        and add circle_wrong*/
        cv::Vec2f reference_adjusted = (-1) * cv::Vec2f(data_[d_id].roi_reference[0] + circles[0][0], data_[d_id].roi_reference[1] + circles[0][1]);
        std::vector<cv::Mat> single_roi;
        single_roi.push_back(data_[d_id].image);
        ml_model_.calculateFeatures(screw_values, single_roi, circles, reference_adjusted);
        all_features.push_back(screw_values);
    }

    if (all_features.empty()) {
        throw std::runtime_error("Feature values are empty?!");
    }

    convertToTrainerData(trainer_data, result_data, all_features);

    std::cout << "Training Model" << std::endl;
    ml_model_.train(trainer_data, result_data);
    std::cout << "Finished Training" << std::endl;

    return;
}

void ScrewTrainer::convertToTrainerData(cv::Mat& trainer_data, cv::Mat& result_data,
    const std::vector<std::vector<std::vector<float>>>& all_features)
{
    //Just copies all_features into the format expected by OpenCV
    int number_of_screws = 0;
    for (uint im_id = 0; im_id != all_features.size(); im_id++) {
        number_of_screws += all_features[im_id].size();
    }

    int number_of_features = all_features[0][0].size();

    result_data = cv::Mat(number_of_screws, 1, CV_32SC1);
    trainer_data = cv::Mat(number_of_screws, number_of_features, CV_32FC1);
    int current_row = 0;
    for (uint im_id = 0; im_id != all_features.size(); im_id++) {

        bool are_screws = data_[im_id].screws;
        for (uint sc_id = 0; sc_id != all_features[im_id].size(); sc_id++) {

            for (uint f_id = 0; f_id != number_of_features; f_id++) {
                trainer_data.at<float>(current_row, f_id) = all_features[im_id][sc_id][f_id];
            }
            result_data.at<int>(current_row) = are_screws;

            current_row++;
        }
    }
}

void ScrewTrainer::loadFullImages()
{
    std::cout << "Starting to import full Images" << std::endl;

    std::string image_path = trainer_params_.image_path;

    std::cout << "Loading images from " + image_path << std::endl;
    if (!std::filesystem::exists(image_path)) {
        throw std::invalid_argument("Image path does not exist!");
    }

    // Loads all consecutively numbered images in folder.
    //If there was no image for 5 numbers, the loading process will finish
    std::string image_file_name;
    int data_count = 0;
    int error_count = 0;
    for (uint t_id = trainer_params_.image_start_number;; t_id++) {
        ScrewData new_set;
        image_file_name = "/" + trainer_params_.image_prefix + std::to_string(t_id) + trainer_params_.image_suffix;
        new_set.image = cv::imread(image_path + image_file_name);

        if (!new_set.image.empty()) {
            new_set.screws = false;
            data_.push_back(new_set);
            error_count = 0;
            data_count++;
        } else {
            error_count++;
        }

        if (error_count == 5) {
            break;
        }
    }

    if (data_count != 0) {
        std::cout << "Loaded " + std::to_string(data_count) + " data sets." << std::endl;
        extractor_parameters_.initiateParameters(data_[0].image.rows, data_[0].image.cols);
    } else {
        throw std::runtime_error("No data found!");
    }
    return;
}

void ScrewTrainer::loadSplicedImages()
{
    std::cout << "Starting to import spliced Input" << std::endl;

    std::string screw_image_path = trainer_params_.image_path + SCREWS_FOLDER;
    std::string hole_image_path = trainer_params_.image_path + HOLES_FOLDER;

    std::cout << "Loading images from " + screw_image_path + " and " + hole_image_path << std::endl;

    if ((!std::filesystem::exists(screw_image_path))
        || (!std::filesystem::exists(hole_image_path))) {
        throw std::logic_error("Spliced image path or spliced hole path do no exist!");
    }

    //Read references
    std::string misc_path = trainer_params_.image_path + MISC_FILE_NAME;
    std::ifstream in_file(misc_path);

    if (!in_file.good()) {
        throw std::runtime_error("Could not open file.");
    }

    std::string line;
    std::vector<cv::Vec2f> all_roi_references;

    //Initial Settings are width and height
    std::getline(in_file, line);
    Pixel width = std::stof(line);
    std::getline(in_file, line);
    Pixel height = std::stof(line);
    extractor_parameters_.initiateParameters(height, width);

    //Get direction from reference to hole
    while (std::getline(in_file, line)) { //GetLine
        std::vector<std::string> vec = split(line, ',');
        cv::Vec2f reference;
        reference[0] = std::stof(vec[0]);
        reference[1] = std::stof(vec[1]);
        all_roi_references.push_back(reference); //Save
    }

    if (in_file.fail()) {
        std::cerr << "Input file stream error bit is set, possible read error on file." << std::endl;
    }

    // Loads spliced image which were saved before
    //If for five consecutive numbers no image was found in all three folders
    //The loading has finished
    std::string image_file_name;
    int data_count = 0;

    int error_count = 0;

    for (uint im_id = 0;; im_id++) {
        ScrewData new_set;
        image_file_name = CIRCLE_FOLDER_PREFIX + std::to_string(im_id) + FILE_TYPE_SUFFIX;
        new_set.image = cv::imread(screw_image_path + image_file_name);
        new_set.roi_reference = all_roi_references[im_id];

        if (!new_set.image.empty()) {
            new_set.screws = true;
            data_.push_back(new_set);
            data_count++;
            error_count = 0;
        } else {
            new_set.image = cv::imread(hole_image_path + image_file_name);

            if (!new_set.image.empty()) {
                new_set.screws = false;
                data_.push_back(new_set);
                data_count++;
                error_count = 0;
            } else {
                error_count++;
            }
        }

        if (error_count == 5) {
            break;
        }
    }

    if (data_count != 0) {
        std::cout << "Loaded " + std::to_string(data_count) + " data sets" << std::endl;
    } else {
        throw std::runtime_error("No data found!");
    }
    return;
}

namespace {
    std::vector<std::string> split(const std::string& s, const char seperator)
    {
        std::vector<std::string> output;

        std::string::size_type prev_pos = 0, pos = 0;

        while ((pos = s.find(seperator, pos)) != std::string::npos) {
            std::string substring(s.substr(prev_pos, pos - prev_pos));

            output.push_back(substring);

            prev_pos = ++pos;
        }

        output.push_back(s.substr(prev_pos, pos - prev_pos)); // Last word

        return output;
    }
}
}
