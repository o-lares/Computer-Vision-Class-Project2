ELEE6280 - Assignment 2 - Feature Tracking and 3D Reconstruction
This project is an implementation of feature tracking and 3D reconstruction using SIFT features, Brute-Force matching, and RANSAC for outlier elimination. It is part of the ELEE6280 Intro Robotics course at the University of Georgia, submitted by Oscar Lares on April 28, 2023.

The code processes a sequence of images and tracks the features between them, computing the 3D positions of the tracked points using triangulation. The resulting 3D point cloud is visualized in 2D, and the trajectory of the camera is also plotted. The results are saved to video files.

Dependencies
The following libraries are required to run the code:

OpenCV (tested with version 4.5.2)
NumPy (tested with version 1.21.0)

Input Data
The input images are expected to be located in a folder specified by the folder_path variable in the code. The images should be in the PNG format with a '.png' extension.

Output
The code produces two output video files:
trajectory_video_lares.avi: A video showing the trajectory of the camera as it moves through the scene.
point_cloud_video_lares.avi: A video showing the 3D point cloud, visualized in 2D as a top-down view.

Running the Code
To run the code, simply execute the script with a Python interpreter:
python Assignment2_Lares.py