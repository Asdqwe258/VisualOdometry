import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

def ReadCameraModel(models_dir):

# ReadCameraModel - load camera intrisics and undistortion LUT from disk
#
#
# INPUTS:
#   image_dir: directory containing images for which camera model is required
#   models_dir: directory containing camera models
#
# OUTPUTS:
#   fx: horizontal focal length in pixels
#   fy: vertical focal length in pixels
#   cx: horizontal principal point in pixels
#   cy: vertical principal point in pixels
#   G_camera_image: transform that maps from image coordinates to the base
#     frame of the camera. For monocular cameras, this is simply a rotation.
#     For stereo camera, this is a rotation and a translation to the left-most
#     lense.
#   LUT: undistortion lookup table. For an image of size w x h, LUT will be an
#     array of size [w x h, 2], with a (u,v) pair for each pixel. Maps pixels
#     in the undistorted image to pixels in the distorted image
################################################################################
#
# Copyright (c) 2019 University of Maryland
# Authors: 
#  Kanishka Ganguly (kganguly@cs.umd.edu)
#
# This work is licensed under the Creative Commons 
# Attribution-NonCommercial-ShareAlike 4.0 International License. 
# To view a copy of this license, visit 
# http://creativecommons.org/licenses/by-nc-sa/4.0/ or send a letter to 
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
#
################################################################################
    intrinsics_path = models_dir + "/stereo_narrow_left.txt"
    lut_path = models_dir + "/stereo_narrow_left_distortion_lut.bin"

    intrinsics = np.loadtxt(intrinsics_path)
    # Intrinsics
    fx = intrinsics[0,0]
    fy = intrinsics[0,1]
    cx = intrinsics[0,2]
    cy = intrinsics[0,3]

    # 4x4 matrix that transforms x-forward coordinate frame at camera origin and image frame for specific lens
    G_camera_image = intrinsics[1:5,0:4]
    # LUT for undistortion
    # LUT consists of (u,v) pair for each pixel)
    lut = np.fromfile(lut_path, np.double)
    lut = lut.reshape([2, lut.size//2])
    LUT = lut.transpose()


    return fx, fy, cx, cy, G_camera_image, LUT


fx, fy, cx, cy, _, LUT = ReadCameraModel('./Oxford_dataset_reduced/model')
K = np.array([[fx, 0, cx],[0, fy, cy],[0, 0, 1]])
print('k')
print(K)
sift = cv2.SIFT_create()
features = [0] * 377
descriptors = [0] * 377
count = 0
for filename in os.listdir('./Oxford_dataset_reduced/images'):
    print('processed '+ str(count) + 'images')
    path = (os.path.join('./Oxford_dataset_reduced/images', filename))
    img = cv2.imread(path,flags=-1)
    color_image = cv2.cvtColor(img, cv2.COLOR_BayerGR2BGR)
    #plt.imshow(color_image)
    #plt.show()
    features[count], descriptors[count] = sift.detectAndCompute(color_image, None)
    count += 1
print(np.shape(descriptors))
bf = cv2.BFMatcher()
count = 0
rotation = [0] * 376
translation = [0] * 376
#method taken from https://www.docs.opencv.org/master/dc/dc3/tutorial_py_matcher.html
#but slightly modified to fit the project
for x in range(1,len(features)):
    print('working on ' + str(count))
    matches = bf.knnMatch(descriptors[x - 1], descriptors[x], k = 2)
    #print(matches)
    pts1 = []
    pts2 = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            pts1.append(features[x-1][m.queryIdx].pt)
            pts2.append(features[x][m.trainIdx].pt)
    pts1 = np.float32(pts1)
    pts2 = np.float32(pts2)
    #print(pts2)
    F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.RANSAC)
    #print('f')
    #print(F)
    E = K.T @ F @ K
    #print('e')
    #print(E)
    points, R, t, mask = cv2.recoverPose(E,pts1,pts2,K)
    rotation[count] = R.astype(np.float32)
    translation[count] = t.astype(np.float32)
    count += 1
#print(rotation)
#print(translation)


full_mat = np.eye(4,dtype=np.float32)
out = [0] * 376
for x in range(len(rotation)):
    print('calculating ' + str(x))
    full_mat = np.array([[rotation[x][0][0],rotation[x][0][1],rotation[x][0][2],translation[x][0]],
                        [rotation[x][1][0],rotation[x][1][1],rotation[x][1][2], translation[x][1]],
                        [rotation[x][2][0],rotation[x][2][1],rotation[x][2][2],translation[x][2]],
                         [0.,0.,0.,1.]],dtype=np.float32) @ full_mat
    out[x] = np.linalg.inv(full_mat) @ np.array([0.,0.,0.,1.],dtype=np.float32)
#print(out)
xs = []
ys = []
zs = []
for o in out:
    xs.append(o[0])
    ys.append(o[1])
    zs.append(o[2])
#    plt.scatter(o[0],o[1])
plt.plot(xs,zs)
plt.show()
ax = plt.axes(projection="3d")
ax.plot3D(xs,zs,ys)
plt.show()
