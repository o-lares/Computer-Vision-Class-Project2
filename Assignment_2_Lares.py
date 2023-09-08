
# Assignment 2 for ELEE6280 - by Oscar Lares

# This is my previous code from Assignment 1 with updates made so RANSAC is used to eliminate outliers
# Some parts of code adapted from these 2 sources: 
# https://medium.com/data-breach/introduction-to-orb-oriented-fast-and-rotated-brief-4220e8ec40cf
# https://github.com/tshanmukh/Facial-KeyPoint-Detection/blob/master/ORB.ipynb

# Submitted 4/28/2023


#import modules needed
import os
import cv2
import numpy as np
import math


#specifying folder path and extension for where images are stored
folder_path = 'D:/OneDrive - University of Georgia/School/Classes/ELEE6280 - Intro Robotics/Assignment 2/kitti_Seq'
extension = '.png'

#load images into an array, use os.listdir to access folder path and read images in loop
images = []
for filename in sorted(os.listdir(folder_path)):
    if filename.endswith(extension):
        image = cv2.imread(os.path.join(folder_path, filename), cv2.IMREAD_COLOR)
        images.append(image)

#create SIFT feature detector and find SIFT keypoints and descriptors for first image
sift = cv2.SIFT_create()
kp, desc = sift.detectAndCompute(images[0], None)

#create a Brute Force Based matcher object
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

#initialize total keypoint and matched keypoints to help keep track of accuracy
total_kp = 0
matched_kp = 0

#create videowriter object to save tracking results to video format
fourcc = cv2.VideoWriter_fourcc(*'XVID')
trajectory_writer = cv2.VideoWriter('trajectory_video_lares.avi', fourcc, 5, images[0].shape[::-1][1:3], True)
point_cloud_writer = cv2.VideoWriter('point_cloud_video_lares.avi', fourcc, 5, images[0].shape[::-1][1:3], True)


#create loop for remaining images after getting kp and desc for first one in order to track the features

#initialize trajectory and pointcloud map variables to draw on
TrajectoryMap = np.zeros((1000, 1000, 3), dtype=np.uint8)
PointCloudMap = np.zeros((1000, 1000, 3), dtype=np.uint8)
scaling_factor_traj = 300
scaling_factor_cloud = 4

frameNumber=1

for i in range(1, len(images)):
    
    #find SIFT keypoints and descriptors for image 'i'
    kp2, desc2 = sift.detectAndCompute(images[i], None)
    
    #perform the matching between the SIFT descriptors of the current image and the previous image
    matches = bf.match(desc, desc2)
      
    # apply RANSAC to remove outliers
    pts1 = np.float32([kp[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)
    
    #Camera intrinsic matrix, given in the homework document
    K = np.array([[707.0493, 0.0, 604.0814], [0.0, 707.0493, 180.5066], [0.0, 0.0, 1.0]])
    
    #Find Fundamental Matrix
    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 0.1, 0.99)
    if frameNumber == 1:
        print("First F matrix:", F)
    if F is None:
        continue
    
    #Find Essential matrix from F and from given K
    try:
        E = K.T.dot(F.dot(K))
    except:
        continue
    if frameNumber == 1:
        print('First E matrix:', E)
    if E is None:
        continue
    
    #Recover the post (R and t) from Essential matrix
    points, R, t, mask = cv2.recoverPose(E, pts1, pts2, K)
    
    # If translation is small or backwards, ingore, b/c car is stopped so don't transform tracking or rotation gets out-of-whack
    if t[2][0] > 0 or abs(t[2][0]) < 0.2: 
        continue
    if t[2][0] <= 1:
        currentR = R
        currentT = t.T
        currentRt = np.append(currentR, currentT.T, axis=1).T
    else:              
        currentRt = np.append(R, t.T, axis=1).T
        t_homog = np.append(t, np.array([[1]]), axis=0)            
        currentT = currentRt.T.dot(t_homog).T                        
        currentR = R.dot(currentR)
    
    M1 = np.hstack((R, t))
    M2 = np.hstack((np.eye(3, 3), np.zeros((3, 1))))

    P1 = np.dot(K,  M2)
    P2 = np.dot(K,  M1)
    point_4d_hom = cv2.triangulatePoints(P1, P2, np.float32(pts1), np.float32(pts2))       
    Points3D = []
    Points3D_Colors = []
    for index, point in enumerate(point_4d_hom.T):
        point = point[:4]/point[-1]
        point[0] = -point[0]            # Reverse X direction
        # Filter points            
        distance = math.sqrt(math.pow(point[0], 2) + math.pow(point[1], 2) + math.pow(point[2],2))
        if distance > 75 or distance < 5 or point[2] > 0:
            Point3D = np.array([0, 0, 0])
        else:               
            # NEXT TRY adding currentT and doing the currentR to each point  
            Point3D = currentRt.T.dot(point)                   
            if pts1[index][0][1] < images[i].shape[0] and pts1[index][0][0] < images[i].shape[1]:
                Points3D.append(Point3D)               
                color = images[i][int(pts1[index][0][1]), int(pts1[index][0][0]), :]
                Points3D_Colors.append(color)

    #Draw the map, starting at center (500,500)
    xCoord_traj = int(currentT[0][0] * scaling_factor_traj + 500) #currentT
    yCoord_traj = int(currentT[0][2] * scaling_factor_traj + 500) #currentT
    point = (xCoord_traj, yCoord_traj)
    cv2.circle(TrajectoryMap, point, 3, (255,0,0), -1)

    # Draw 3D point cloud on 2D map top-down
    for index, point in enumerate(Points3D):
        xCoord = int(point[0] * scaling_factor_cloud + 500)
        yCoord = int(point[2] * scaling_factor_cloud + 500)
        cloudpoint = (xCoord, yCoord)
        color = tuple ([int(x) for x in Points3D_Colors[index]])
        cv2.circle(PointCloudMap, cloudpoint, 3, color, 1)
    
    cv2.imshow("Point Cloud Map", PointCloudMap)    
    cv2.imshow("Trajectory Map", TrajectoryMap)
    trajectory_writer.write(TrajectoryMap)
    point_cloud_writer.write(PointCloudMap)
    
    # Add a small delay and a keyboard interrupt check
    key = cv2.waitKey(2) & 0xFF
    if key == ord("q"):
        break
        
    #update kp and desc for the next iteration in the loop
    kp, desc = kp2, desc2

    frameNumber=0

#end the video writer object and destroy all windows
trajectory_writer.write(TrajectoryMap)
point_cloud_writer.write(PointCloudMap)
trajectory_writer.release()
point_cloud_writer.release()

cv2.destroyAllWindows()

