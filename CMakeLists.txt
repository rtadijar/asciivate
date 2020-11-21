cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

set( CMAKE_EXPORT_COMPILE_COMMANDS ON )

project( Asciivate )
enable_language(CUDA)

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable( Asciivate src/asciivate.cpp src/asciivate.cu)

target_link_libraries( Asciivate ${OpenCV_LIBS} )

set_target_properties( Asciivate PROPERTIES WITH_CUDA ON )
target_compile_options(Asciivate PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-arch=sm_60 >)