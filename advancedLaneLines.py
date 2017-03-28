import cv2
import glob
import numpy as np
from scipy import ndimage
import random
import PIL
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


####  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Part A: Help Functions ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  ####

### plot_dist_vs_undist: plot an image before and after we calibrated the camera
###   Input: 
###		img_dist: original image in RGB.
###		img_undist: image after calibration in RGB.
###		corners: Indication if we are dealing with the Chessboard. If so this variable will hold the corners of the Chessboard. 
###     title: string for the title
###
###   Output: 
###      save the figure.
def plot_dist_vs_undist(img_dist, img_undist, corners=None, title=''):
	
	if corners is not None: # Chessboard case
		fig_name = 'chessboard'
		# crop the image show to the relevant section - only of the chessboard
		X_extra_from_corner = int(50+(max(corners[:,0,0]) - min(corners[:,0,0]))/8)
		Y_extra_from_corner = int(50+(max(corners[:,0,1]) - min(corners[:,0,1]))/5)
		X_min = int(max(0,-X_extra_from_corner+min(corners[:,0,0])))
		X_max = int(min(img_dist.shape[1],X_extra_from_corner+max(corners[:,0,0])))
		Y_min = int(max(0,-Y_extra_from_corner+min(corners[:,0,1])))
		Y_max = int(min(img_dist.shape[0],Y_extra_from_corner+max(corners[:,0,1])))
	else: # road case
		fig_name = 'road'
	
	# plot the two images - the Original and the Undistorted
	plt.figure(figsize=(16,8))
	plt.subplot(1,2,1)
	plt.imshow(img_dist)
	plt.title('Original: {}'.format(title))
	if corners is not None:
		plt.xlim([X_min,X_max])
		plt.ylim([Y_max,Y_min])
	plt.subplot(1,2,2)
	plt.imshow(img_undist)
	plt.title('Undistorted Image')
	if corners is not None:
		plt.xlim([X_min,X_max])
		plt.ylim([Y_max,Y_min])
	
	plt.savefig('output_images/original_vs_calibratted_{}.png'.format(fig_name))

	
### get_object_grid: return a grid given the max value in x,y axis. z axis is set to zero
###   Input: 
###		nx: number of grid points in the x-axis.
###		ny: number of grid points in the y-axis.
###
###   Output: 
###      objp: 2D array like: (0,0,0), (1,0,0), (2,0,0) ....,(9,6,0).
def get_object_grid(nx, ny):
	objp = np.zeros((nx*ny,3), np.float32)
	objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)      # x,y coordinates. z is set to zero.	
	return objp


### find_calibration_params: Find the calibration parameters, using the given calibration images
###   Note: not all the images that were given to us were 9x6 images (3/20). We made some changes to use also the other images
###   Input: 
###		nx: nominal number of corners in the x-axis. we will also check up to -dn around it
###		ny: nominal number of corners in the y-axis. we will also check up to -dn around it
###		dn: the maximal delta we will ad to each axis 
###     plot_example: after we finish with the calibration we choose a random calibration image and we plot the original image and the image after we calibrated it.
###
###   Output: 
###      image: the final image. In a RGB format.
def find_calibration_params(nx, ny, dn, plot_example=False):
	# Read all the calibration images
	imgs_fnames = glob.glob('camera_cal/calibration*.jpg')
	
	# Arrays to store object points and image points from all the images.
	obj_points = [] # 3d points in real world space
	img_points = [] # 2d points in image plane.
	
	dn_vec = np.mgrid[-dn:1, -dn:1].T.reshape(-1,2)          # we will also check around nx,ny.
	dn_vec = np.flipud(dn_vec)                               # we will start with the original nx,ny
	for fname in imgs_fnames:
		img = cv2.imread(fname)                              # read each image (the image will be in BGR)
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)         # convert image to gray
		for d in dn_vec:
			ret, corners = cv2.findChessboardCorners(gray, (nx+d[0],ny+d[1]), None)  # find the corners in the given image
			if ret==True:                                  # found corners, so lets append them to the obj_points
				objp = get_object_grid(nx+d[0],ny+d[1])    # objp is NOT the same for all pictures
				obj_points.append(objp)                    # append objp to the total list
				img_points.append(corners)                 # corners we found in this image
				break

	print('Calibrate on {} images out of {} images'.format(len(img_points), len(imgs_fnames)))
	ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)
	
	# Choose a random calibration image and we plot the original image and the image after we calibrated it 
	if plot_example:
		idx = np.random.randint(low=0, high=len(imgs_fnames))      # choose a random calibration image 
		img = cv2.imread(imgs_fnames[idx])                         # read the image
		img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)                  # Switch to RGB format
		dst = cv2.undistort(img, mtx, dist, None, mtx)             # undistorted image

		plot_dist_vs_undist(img, dst, corners=img_points[idx], title=imgs_fnames[idx])
	
	print('Finished Calibration')
	return ret, mtx, dist, rvecs, tvecs


### calibrate_road_image: calibrate and save a random example for a road image using the calibration parameters from the chessboard calibration
###   Input: 
###		mtx: the camera matrix
###		dist: distortion coefficients
###		idx: the index of the image we want from the test images (0-5). None - for randon index.
###
###   Output: 
###      dst: the undistorted image.	
###      save the image before and after calibration.	
def calibrate_road_image(mtx, dist, idx=None):
	imgs_fnames = glob.glob('test_images/test*.jpg')
	if idx is None:
		idx = np.random.randint(low=0, high=len(imgs_fnames))  # choose a random road image
	idx = max(0,min(idx,len(imgs_fnames)))                     # making sure we are in the range
	img = cv2.imread(imgs_fnames[idx])                         # read the image
	img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)                  # Switch to RGB format
	dst = cv2.undistort(img, mtx, dist, None, mtx)             # undistorted image
	plot_dist_vs_undist(img, dst, corners=None, title=imgs_fnames[idx])
	return dst
	

	
	
####  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Part B: Main Pipeline ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  ####
###
### We will find the calibration parameters, and test it one the test image
ret, mtx, dist, rvecs, tvecs = find_calibration_params(nx=9, ny=6, dn=3, plot_example=True)	# find Calibration parameters
rgb_undist_img = calibrate_road_image(mtx, dist, idx=0) # apply the Calibration parameters on one of the test images

###
### Next we will try to identify the lane lines on the undistorted image using:
###       - sobel (gradient vector)
###       - thresholding in different color spaces 
### We will binary slice the different outputs and finally combine the two (and overlapping regions)
### We will plot an example on what we got on a random road test image (the same image we got before)
b_sobel_undist_img = sobel_binary_th(rgb_undist_img) # b_ = binary  


