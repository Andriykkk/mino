cmake_minimum_required(VERSION 3.11)

project(MLLibrary C)

# Enable compiler warnings
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall -Wextra")

# Add the executable
add_executable(MyCApp src/main.c)
 