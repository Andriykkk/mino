# C-GLFW-OpenGL Template

This repository is a template for setting up a simple C project using GLFW and OpenGL. It includes configurations for Visual Studio Code to ensure that the IDE recognizes GLFW headers without errors. This setup is perfect for beginners who try learning C and wanting to try OpenGL programming.

## Prerequisites

- [CMake](https://cmake.org/download/) installed on your system.
- A C compiler (e.g., MinGW, GCC, Clang).
- [Visual Studio Code](https://code.visualstudio.com/) with the CMake Tools extension installed(if you want, you can use any other IDE).

## Getting Started

### Clone the Repository

```sh
git clone https://github.com/your-username/c-glfw-opengl-template.git
cd c-glfw-opengl-template
```

### VSCode Configuration

This project is pre-configured for Visual Studio Code to avoid include path errors for GLFW. The necessary settings are included in the `.vscode` folder.

## Building the Project

### Using CMake Tools Extension

1. Open the project in Visual Studio Code.
2. Ensure you have the CMake Tools extension installed.
3. Press `F1`, type `CMake: Configure`, and select it.
4. Press `F1`, type `CMake: Build`, and select it.
5. Press `F1`, type `CMake: Run Without Debugging` to run the project.

### Using Command Line

If you don't have the CMake Tools extension, you can build the project using the command line:

1. Create a build directory and navigate into it:

    ```sh
    mkdir build
    cd build
    ```

2. Run CMake to configure the project:

    ```sh
    cmake ..
    ```

3. Build the project:

    ```sh
    cmake --build .
    ```

4. Run the executable (located in the `build` directory):

    ```sh
    ./GLFW-CMake-starter
    ```
