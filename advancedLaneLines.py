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
###     plot_en: after we finish with the calibration we choose a random calibration image and we plot the original image and the image after we calibrated it.
###
###   Output: 
###      image: the final image. In a RGB format.
def find_calibration_params(nx, ny, dn, plot_en=False):
	print('---> Start Calibration')
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

	print('\t-->Calibrate on {} images out of {} images'.format(len(img_points), len(imgs_fnames)))
	ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)
	
	# Choose a random calibration image and we plot the original image and the image after we calibrated it 
	if plot_en:
		idx = np.random.randint(low=0, high=len(imgs_fnames))      # choose a random calibration image 
		img = cv2.imread(imgs_fnames[idx])                         # read the image
		img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)                  # Switch to RGB format
		dst = cv2.undistort(img, mtx, dist, None, mtx)             # undistorted image

		plot_dist_vs_undist(img, dst, corners=img_points[idx], title=imgs_fnames[idx])
	
	return ret, mtx, dist, rvecs, tvecs


### calibrate_road_image: calibrate and save a random example for a road image using the calibration parameters from the chessboard calibration
###   Input: 
###		mtx: the camera matrix
###		dist: distortion coefficients
###		idx: the index of the image we want from the test images (0-5) or straight_lines images (0-1). None - for randon index.
###		fname: the string at the start of the files name
###		plot_en: if we want to plot the original image and the calibrated image
###
###   Output: 
###      dst: the undistorted image.	
###      save the image before and after calibration.	
def calibrate_road_image(mtx, dist, idx=None, fname='test', plot_en=False):
	imgs_fnames = glob.glob('test_images/'+fname+'*.jpg')
	if idx is None:
		idx = np.random.randint(low=0, high=len(imgs_fnames))  # choose a random road image
	idx = max(0,min(idx,len(imgs_fnames)))                     # making sure we are in the range
	img = cv2.imread(imgs_fnames[idx])                         # read the image
	img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)                  # Switch to RGB format
	dst = cv2.undistort(img, mtx, dist, None, mtx)             # undistorted image
	if plot_en:
		plot_dist_vs_undist(img, dst, corners=None, title=imgs_fnames[idx])
	return dst
	

### sobel_binary_th: apply sobel on the gray image on x-axis and y-axis. 
###                     Then we use 4 different thresholds on |sobel_x|,|sobel_y|,|sobel|,abg(sobel) to get a binary picture with the lane lines
###                     out = (|sobel_x|&|sobel_y|) | (|sobel|&abg(sobel))
###   Input: 
###		rgb_img: rgb image
###		kernel_size: kernel size of the sobel. default is 3
###		plot_en: plot the grey sacle image, the final result in binary image and a colored image of each of the components (sobel_x, |sobel|, ang_sobel) 
###
###   Output: 
###      b_sobel_total: binary image after all the thresholds were applied.	
def sobel_binary_th(rgb_img, kernel_size=3, plot_en=False):
	print('\t--> Start Sobel Binary Threshold')
	# 1) convert to gray scale
	gray_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)               # convert to gray scale
	
	# 2) Now we calculate all the sobels and sobel functions we need
	sobel_x = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=kernel_size) # sobel on x-axis
	sobel_y = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=kernel_size) # sobel on y-axis
	abs_sobel_x = np.absolute(sobel_x)                             # |sobel_x|
	abs_sobel_x = np.uint8(255*abs_sobel_x/np.max(abs_sobel_x))    # scale |sobel_x|
	abs_sobel_y = np.absolute(sobel_y)                             # |sobel_y|
	abs_sobel_y = np.uint8(255*abs_sobel_y/np.max(abs_sobel_y))    # scale |sobel_y|
	mag_sobel = np.sqrt(sobel_x**2 + sobel_y**2)                   # |sobel| = sqrt(sobel_x^2 + sobel_y^2) 
	mag_sobel = np.uint8(255*mag_sobel/np.max(mag_sobel))          # scale |sobel|
	# for the angle we use a larger kernel_size to reduce noise
	sobel_x_lpf = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=(5*kernel_size))      # sobel on x-axis with bigger kernel
	sobel_y_lpf = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=(5*kernel_size))      # sobel on y-axis with bigger kernel
	ang_sobel = np.arctan2(np.absolute(sobel_y_lpf), np.absolute(sobel_x_lpf))      # abg(sobel) [-pi,pi)
	
	# we put in a list all the sobels we will use
	all_sobel_name=['abs_sobel_x','abs_sobel_y','mag_sobel','ang_sobel']
	all_sobel=[abs_sobel_x,abs_sobel_y,mag_sobel,ang_sobel] 
	
	# 3) creating binary images - each will threshold a different sobel information 
	#    We start by setting the thresholds
	all_th_en=['th', 'all_ones', 'all_zeros', 'all_zeros'] # threshold enablers: "all zeros", "all ones" or to use the threshols  
	all_th = np.array([[20,100], [80,100], [70,100], [0.7, 1.3]]) # the threshold for each sobel function
	
	b_all_sobel = [] # a list of the binary sobels after we use the threshold
	for i in range(len(all_sobel)):
		if ('all_ones' == all_th_en[i]):                            # all ones
			b_all_sobel.append(np.ones_like(all_sobel[i])) 
			print('\t\t{}: all Ones'.format(all_sobel_name[i]))
		elif ('all_zeros' == all_th_en[i]):                         # all zeros
			b_all_sobel.append(np.zeros_like(all_sobel[i])) 
			print('\t\t{}: all Zeros'.format(all_sobel_name[i]))
		else:                                                       # normal threshold
			b_sobel = np.zeros_like(all_sobel[i]) 
			b_sobel[(all_sobel[i] >= all_th[i,0]) & (all_sobel[i] <= all_th[i,1])] = 1    # apply threshold
			b_all_sobel.append(b_sobel) 
	
	
	# 4) combine all the binary images
	b_sobel_ax = np.zeros_like(b_all_sobel[0])
	b_sobel_ax[(b_all_sobel[0] == 1) & (b_all_sobel[1] == 1)] = 1
	b_sobel_mag_ang = np.zeros_like(b_all_sobel[2])
	b_sobel_mag_ang[(b_all_sobel[2] == 1) & (b_all_sobel[3] == 1)] = 1
	b_sobel_total = np.zeros_like(b_sobel_ax)
	b_sobel_total[(b_sobel_ax == 1) | (b_sobel_mag_ang == 1)] = 1
	
	if plot_en: # plot the binary image and the colored binary components
		color_binary = np.dstack((b_sobel_ax, np.zeros_like(b_sobel_ax), b_sobel_mag_ang)) 
		plt.figure(figsize=(16,8))
		plt.subplot(1,3,1)
		plt.imshow(gray_img, cmap='gray')
		plt.title('Gray image')
		plt.subplot(1,3,2)
		plt.imshow(255*color_binary)
		plt.title('Sobel breakdown (r=sobel_ax, b=mag_ang(sobel))')
		plt.subplot(1,3,3)
		plt.imshow(b_sobel_total, cmap='gray')
		plt.title('Binary after sobel th')


		plt.savefig('output_images/applay_sobel_th.png')
	
	return b_sobel_total

	
### color_binary_th: apply color threshold on the different channels in the RGB, HLS color space 
###                    out = (r_channel | s_channel)
###   Input: 
###		rgb_img: rgb image
###		plot_en: plot the undistorted image, the final result in binary image and the colored binary of those two threshold. 
###
###   Output: 
###      b_color_total: binary image after all the thresholds were applied.	
def color_binary_th(rgb_img, plot_en=False):
	print('\t--> Start Color Binary Threshold')
	# 1) take the relevant channels: red, Saturation
	r_img = rgb_img[:,:,0]                                       # Red channel 
	s_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HLS)[:,:,2]      # Saturation channel 
	# we put in a list all the channels we will use
	all_clr_name=['red','sat']
	all_clr=[r_img,s_img] 
	
	# 2) creating binary images - each will threshold a different channel 
	#    We start by setting the thresholds
	all_th_en=['all_zeros', 'th'] # threshold enablers: "all zeros", "all ones" or to use the threshols  
	all_th = np.array([[170,255], [170,255]]) # the threshold for each channel
	
	b_all_clr = [] # a list of the binary channels after we use the threshold
	for i in range(len(all_clr)):
		if ('all_ones' == all_th_en[i]):                            # all ones
			b_all_clr.append(np.ones_like(all_clr[i])) 
			print('\t\t{}: all Ones'.format(all_clr_name[i]))
		elif ('all_zeros' == all_th_en[i]):                         # all zeros
			b_all_clr.append(np.zeros_like(all_clr[i])) 
			print('\t\t{}: all Zeros'.format(all_clr_name[i]))
		else:                                                       # normal threshold
			b_clr = np.zeros_like(all_clr[i]) 
			b_clr[(all_clr[i] >= all_th[i,0]) & (all_clr[i] <= all_th[i,1])] = 1    # apply threshold
			b_all_clr.append(b_clr) 
	

	# 4) combine all the binary images
	b_clr_total = np.zeros_like(b_all_clr[0])
	b_clr_total[(b_all_clr[0] == 1) | (b_all_clr[1] == 1)] = 1
	
	if plot_en: # plot the binary image and the colored binary components
		color_binary = np.dstack((b_all_clr[0], np.zeros_like(b_all_clr[0]), b_all_clr[1])) 
		plt.figure(figsize=(16,8))
		plt.subplot(1,3,1)
		plt.imshow(rgb_img)
		plt.title('Original image')
		plt.subplot(1,3,2)
		plt.imshow(255*color_binary)
		plt.title('Channel breakdown (r=red, b=sat)')
		plt.subplot(1,3,3)
		plt.imshow(b_clr_total, cmap='gray')
		plt.title('Binary after color th')

		plt.savefig('output_images/applay_color_th.png') 
	
	return b_clr_total


### apply_binary_th: apply the following thresholds:
###                    - sobel (all kind of functions)
###                    - color channel threshold in different color space 
###                        out = (sobel | color)
###   Input: 
###		rgb_img: rgb image
###		plot_en: plot the undistorted image, the final result in binary image and the colored binary of those two threshold. 
###
###   Output: 
###      b_total: binary image after all the thresholds were applied	
def apply_binary_th(rgb_img, plot_en=False):
	print('---> Start Binary Threshold')
	b_sobel_undist_img = sobel_binary_th(rgb_img, kernel_size=3, plot_en=True) # b_ stands for binary  
	b_color_undist_img = color_binary_th(rgb_img, plot_en=True) # b_ stands for binary
	
	# combine all the binary images
	b_total = np.zeros_like(b_sobel_undist_img)
	b_total[(b_sobel_undist_img == 1) | (b_color_undist_img == 1)] = 1
	
	if plot_en: # plot the binary image and the colored binary components
		color_binary = np.dstack((b_sobel_undist_img, np.zeros_like(b_total),b_color_undist_img)) 
		plt.figure(figsize=(16,8))
		plt.subplot(1,3,1)
		plt.imshow(rgb_img)
		plt.title('Original image')
		plt.subplot(1,3,2)
		plt.imshow(255*color_binary)
		plt.title('Channel breakdown (r=sobel, b=color)')
		plt.subplot(1,3,3)
		plt.imshow(b_total, cmap='gray')
		plt.title('Binary after sobel&color th')

		plt.savefig('output_images/binary_final.png') 
	
	return b_total

	
### find_birdeye_matrix: Calc the bird-eye Tarnsform Matrix (and the inverse) 
###                      We will look on both images we have "straight_lines" and average the output
###   Input: 
###		M_op: wich matrix we want to return: 0-average on both straight_lines images. 1:straight_lines1, 2: straight_lines2
###		plot_en: plot the original image and the bird-eye image (for both pictures we have)
###			mtx: calibration matrix. if we want to plot the straight_lines images we will need the calibration matrix
###			dist: distortion coefficients. if we want to plot the straight_lines images we will need the distortion coefficients
###
###   Output: 
###      M: The tarnsform Matrix (normal to bird-eye)	
###      M_inv: The inverse tarnsform Matrix (bird-eye to normal)	
def find_birdeye_matrix(M_op=0, plot_en=False, mtx=None, dist=None):
	print('---> Start Calc the bird-eye Tarnsform Matrix')
	
	# 1) How many images do we have and the size of them
	prefix='straight_lines'                                # the prefix for the 'straight_lines' images
	imgs_fnames = glob.glob('test_images/'+prefix+'*.jpg') # all the images with straight_lines
	n_img = len(imgs_fnames)                               # number of images
	rgb_undist_img = calibrate_road_image(mtx, dist, idx=0, fname=prefix, plot_en=False) # load a image 
	img_size = (rgb_undist_img.shape[1], rgb_undist_img.shape[0])  # image size
	if False: # just to find the points on the original image
		plt.figure(figsize=(16,8))
		plt.imshow(rgb_undist_img)
		plt.show()
	
	# 2) we set the points for the trapeze of source images and for the bird-eye image
	#    Each trapeze is defined by 6 values: y_down, y_up, x_right_down, x_left_down, x_right_down, x_left_up
	y_down = [689,689]           # for the two images
	y_up = [450,450]             # for the two images
	x_right_down = [1055,1062]   # for the two images
	x_left_down = [250,260]      # for the two images
	x_right_up = [683,688]       # for the two images
	x_left_up = [596,595]        # for the two images
		
	# 3) Using getPerspectiveTransform we will calculate the transform matrix (and the inverse)  per image
	src=[]    # will hold all the sources points. trapeze (4 points) for each image
	dst=[]    # will hold all the destination points. rectangle (4 points) for each image
	M=[]      # will hold the transform matrix per image
	M_inv=[]  # will hold the inverse transform matrix per image
	for i in range(n_img):
		src.append(np.float32([[x_left_down[i],y_down[i]], [x_left_up[i]  ,y_up[i]],     [x_right_up[i]  ,y_up[i]],     [x_right_down[i],y_down[i]]]))
		dst.append(np.float32([[x_left_down[i],y_down[i]], [x_left_down[i],0], [x_right_down[i],0], [x_right_down[i],y_down[i]]]))
		M.append(cv2.getPerspectiveTransform(src[-1], dst[-1]))
		M_inv.append(cv2.getPerspectiveTransform(dst[-1], src[-1]))
	
	# 4) Average the transform matrix (and the inverse)
	M_avg = np.average(np.array(M),0)
	M_inv_avg = np.average(np.array(M_inv),0)
	
	if plot_en: # plot the straight_lines (after removing distortion) and their birdeye view
		plt.figure(figsize=(16,8))
		for i in range(n_img):
			rgb_undist_img = calibrate_road_image(mtx, dist, idx=i, fname=prefix, plot_en=False)              # get the calibrated image
			cv2.line(rgb_undist_img, (src[i][0,0],src[i][0,1]), (src[i][1,0],src[i][1,1]), [255,0,0], 2)      # draw the lane lines in the car view image - left lane
			cv2.line(rgb_undist_img, (src[i][2,0],src[i][2,1]), (src[i][3,0],src[i][3,1]), [255,0,0], 2)      # draw the lane lines in the car view image - right lane
			plt.subplot(2,2,i+1)
			plt.imshow(rgb_undist_img)
			plt.title('Car view:{}'.format(imgs_fnames[i]))
			
			warped = cv2.warpPerspective(rgb_undist_img, M[i], img_size)                                      # apply the transformation
			plt.subplot(2,2,i+3)
			plt.imshow(warped)
			plt.title('Birdeye view:{}'.format(imgs_fnames[i]))

		plt.savefig('output_images/birdeye_on_straight_lines.png')
	
	# 5) which matrix (and inverse) we want to return
	if M_op==0: # average matrix
		M_ret, m_inv_ret = M_avg, M_inv_avg  
	else: # matrix of straight_lines1 or straight_lines2
		M_ret, m_inv_ret = M[M_op-1], M_inv[M_op-1]
	
	return M_ret, m_inv_ret
	
####  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Part B: Main Pipeline ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  ####

###
### We will find the calibration parameters, and test it one the test image
ret, mtx, dist, rvecs, tvecs = find_calibration_params(nx=9, ny=6, dn=3, plot_en=True)	# find Calibration parameters
rgb_undist_img = calibrate_road_image(mtx, dist, idx=0, fname='straight_lines', plot_en=True) # apply the Calibration parameters on one of the test images

###
### Next we will try to identify the lane lines on the undistorted image using:
###       - sobel (gradient vector)
###       - thresholding in different color spaces 
### We will binary slice the different outputs and finally combine the two (OR operator on the overlapping regions)
### We will plot an example on what we got on a random road test image (the same image we got before)
b_undist_img = apply_binary_th(rgb_undist_img, plot_en=True) # b_ stands for binary  
M, M_inv = find_birdeye_matrix(M_op=1, plot_en=True, mtx=mtx, dist=dist)
