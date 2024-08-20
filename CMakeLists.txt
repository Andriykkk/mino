cmake_minimum_required(VERSION 3.11)

project(MLLibrary C)

# Enable compiler warnings
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall -Wextra")

# Specify the source files
set(SOURCES
    src/main.c
    src/mat/mat.c   # Add mat.c to the source list
)

# Add the executable
add_executable(MyCApp ${SOURCES})
 
# Include the directory where mat.h is located
target_include_directories(MyCApp PRIVATE src/mat)
target_include_directories(MyCApp PRIVATE src/macros)
 