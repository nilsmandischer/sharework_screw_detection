cmake_minimum_required(VERSION 2.8.3)
project(screw_detection)
SET(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-std=c++17")

find_package(OpenCV REQUIRED)

include_directories(
  include
  ${OpenCV_INCLUDE_DIRS}
)

#Add Screw Detection Lib
add_library(screw_detection STATIC
    src/screw_detection/detection.cpp
    src/screw_detection/model.cpp
    src/screw_detection/training.cpp
    src/screw_detection/roi_extractor.cpp
)
target_include_directories(screw_detection
    PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>  
      $<INSTALL_INTERFACE:include>
)
target_link_libraries(screw_detection
    ${OpenCV_LIBS}
)

####INSTALL#####


install(TARGETS screw_detection
    EXPORT ScrewDetectionTargets
    DESTINATION lib
)
install(DIRECTORY include/screw_detection/
  DESTINATION include/screw_detection
  FILES_MATCHING PATTERN "*.h"
)

install(EXPORT ScrewDetectionTargets
  FILE ScrewDetectionTargets.cmake
  DESTINATION "lib/ScrewDetection"
)

include(CMakePackageConfigHelpers)
# generate the config file that includes the exports
configure_package_config_file(${CMAKE_CURRENT_SOURCE_DIR}/Config.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/ScrewDetectionConfig.cmake"
  INSTALL_DESTINATION "lib/ScrewDetection"
  NO_SET_AND_CHECK_MACRO
  NO_CHECK_REQUIRED_COMPONENTS_MACRO
)

install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/ScrewDetectionConfig.cmake
  DESTINATION lib/ScrewDetection
)

export(EXPORT ScrewDetectionTargets
  FILE "${CMAKE_CURRENT_BINARY_DIR}/ScrewDetectionTargets.cmake"
)
