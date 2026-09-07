#########################################################
#           RhythmFormer (rPPG) API library            #
#########################################################
cmake_minimum_required(VERSION 3.21.0)

set(RHYTHMFORMER_API_TARGET rhythmformer_api)
project(${RHYTHMFORMER_API_TARGET}
        DESCRIPTION     "RhythmFormer remote photoplethysmography use case API library"
        LANGUAGES       C CXX)

add_library(${RHYTHMFORMER_API_TARGET} STATIC
        RhythmFormerProcessing.cc
        MotionDetector.cc
        FomoFaceDetector.cc)

target_link_libraries(${RHYTHMFORMER_API_TARGET} PUBLIC common_api)

add_library(mlek::${RHYTHMFORMER_API_TARGET} ALIAS ${RHYTHMFORMER_API_TARGET})

message(STATUS "*******************************************************")
message(STATUS "Library                                : " ${RHYTHMFORMER_API_TARGET})
message(STATUS "CMAKE_SYSTEM_PROCESSOR                 : " ${CMAKE_SYSTEM_PROCESSOR})
message(STATUS "*******************************************************")