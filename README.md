# Screw Detection

This package locates and classifies screws inside indivually defineable regions of interest (ROI). Refer to the screw_detection_ros package (https://github.com/nilsmandischer/sharework_screw_detection_ros) for an example how to use this library.

## Dependencies

The package has following dependencies:
- OpenCV 

## Installation
Prerequisits: Installed all dependencies

Build the screw_detection_library
```console
mkdir build && cd build && cmake .. && make
```
Optionally install the library using __(not recommended)__
```console
make install
```

There are two main ways of including the library in your project:
1. __(not recommended)__ install the library and use "find_package(ScrewDetection REQUIRED)"
2. _(recommended)_ Set the "ScrewDetection_DIR" variable in your project to the build directory of this project (e.g. "SET(ScrewDetection_DIR "${BASE_DIR}/modules/sharework_screw_detection/screw_detection/build") and use "find_package(ScrewDetection CONFIG REQUIRED)"

## Structure

The package contains one library:
- screw_detection

This library contains three public headers:
- detection.h: Main header to use a trained Model to detect screws in images.
- training.h: Main header to train the model and write the model to disk.
- extractor_parameters.h: Contains all parameter sets used by the detector and trainer and the definition of pure abstract class CustomROI.

And two private headers:
- model.h: Contains the ScrewDetectorModel class, which acts as an interface to cv::DTrees and handles feature calculation inside both the Detector and the Trainer.
- roi_extractor.h: Contains the ROIExtractor class to extract parsed images of single screws from an image or the underlying circle contour and position of these screws. Uses the CustomROI to detect the ROI in which the screws are located inside the Detector and the Trainer.

## Setup and Usage

__Parameter Identification:__
Use a pre trained model (found under .../screw_detection/examples/), the ScrewDetector Class and an interface with your code to adjust the parameters on the fly to work best for the scenario. It is recommended to do this using the Docker provided in the screw_detection_ros package, for more information refer to screw_detection_ros's documentation.
It is recommended to find a set of working parameters before taking a full set of training images, to ensure the taken images can actually be used and the setup does not have to be adjusted and the images retaken.

__Training:__
1) Take training images and store them with increasing numbering. The identifier and start number can be chosen freely, e.g. "image_1.jpg". These images contain the whole ROI and an arbitrary number of screws inside those images. It is recommended to either have screws in all possible holes or in none per image. This is to ensure an easier classification during training.
2) Create the necessary objects to hold the parameters for the ScrewTrainer. See extractor_parameters.h.
3) _(optional)_ Confirm that your parameters work by using a pretrained model (see .../examples/) and the ScrewDetector. Refer to the screw_detection_ros package implementation examples since this depends on the used CustomROI.
4) Create a ScrewTrainer object with the same parameters. Make sure to initiate the ExtractorParameters beforehand. You do not need to initae each parameter set individually.
5) Use ScrewTrainer.spliceImages() to load the training images and splice them into single images of screws/holes. This will prompt the user to classify each single image using a command line application for later training and write the training samples to disk.
6) _(optional)_ Review the written training samples in their corresponding folders. Feel free to delete single samples.
7) Use ScrewTrainer.trainModel() to train the model and write the model to disk for later usage.

__Detection:__
1) Create the necessary objects to hold the parameters for the ScrewDetector. See extractor_parameters.h.
2) Create a ScrewDetector object with the same parameters as used when training the model. Make sure to initiate the parameters beforehand. You do not need to initae each parameter set individually.
3) Use ScrewDetector.processImage() to get a detection result for a single image.
