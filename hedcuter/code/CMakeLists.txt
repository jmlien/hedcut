cmake_minimum_required(VERSION 2.6)

project(hedcuter)

set(PACKAGE_VERSION "0.1.0")

########## set compiler flags ##########
if(NOT WIN32)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations")
endif()

if(WIN32)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  /D _CRT_SECURE_NO_WARNINGS")
endif()



#####################################################
#                COMMON Libraries                   #
#####################################################
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
find_package(OpenCV 3.0 REQUIRED)

include_directories(
        ${OPENGL_INCLUDE_DIRS}
        ${GLUT_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS})


#################################################################
# Unfolder GUI
#################################################################

set(APP_NAME hedcuter)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR})

set(HEDCUT_SRC hedcut.cpp   main.cpp    wcvt.cpp    wcvt_gpu.cpp
)

add_executable(${APP_NAME} ${HEDCUT_SRC})
#add_dependencies(${APP_NAME})

target_link_libraries(${APP_NAME} ${OpenCV_LIBS}  ${OPENGL_LIBRARIES} ${GLUT_LIBRARY})
