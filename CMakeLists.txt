cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(Document_Image_Quality_Assessment)

set(CMAKE_CXX_STANDARD 17)
#set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG_INIT "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_RELEASE_INIT "-Wall -Wextra")


set(REQUIRED_LIBS ${TFLITE_LIBS})
# list(APPEND REQUIRED_LIBS ${TFLITE_LIBS})

find_package(OpenCV 4 REQUIRED)
find_package(Threads REQUIRED)
find_library(TFLITE_LIBRARY tensorflowlite "lib")
#find_library(FLATBUFFER_LIBRARY flatbuffers "lib")
include_directories("include")

set(REQUIRED_LIBS "stdc++fs")
list(APPEND REQUIRED_LIBS ${TFLITE_LIBRARY})
#list(APPEND REQUIRED_LIBS ${FLATBUFFER_LIBRARY})
list(APPEND REQUIRED_LIBS ${OpenCV_LIBS})
list(APPEND REQUIRED_LIBS ${Threads_LIBS})

set(CMAKE_C_FLAGS "-pthread")
set(SOURCE_FILES diqa.cpp)

add_executable("diqa" ${SOURCE_FILES})
SET_TARGET_PROPERTIES(diqa PROPERTIES LINK_FLAGS "-no-pie")
target_link_libraries("diqa" ${REQUIRED_LIBS} -lpthread)

#cmake -DCMAKE_BUILD_TYPE=Debug ..
#cmake -DCMAKE_BUILD_TYPE=Release ..
