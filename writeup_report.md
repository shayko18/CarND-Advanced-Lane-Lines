##Writeup Report


---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/original_vs_calibratted_chessboard_0.png "calibratted_chess"
[image2]: ./output_images/original_vs_calibratted_chessboard_rnd.png "calibratted_chess"
[image3]: ./output_images/original_vs_calibratted_road.png "calibratted_road"
[image4]: ./output_images/applay_sobel_th.png "Binary_Sobel"
[image5]: ./output_images/applay_color_th.png "Binary_Color"
[image6]: ./output_images/binary_final.png "Binary_Final"
[image7]: ./output_images/birdeye_on_straight_lines.png "birdeye_sl"
[image8]: ./output_images/fit_polynomial_startover.png "polyfit"
[image9]: ./output_images/fit_polynomial_use_prev.png "polyfit_prev"
[image10]: ./output_images/final_est_lane.png "final_est"
[video1]: ./project_video_with_lane_est.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. [Here](https://github.com/shayko18/CarND-Advanced-Lane-Lines) is my project repository.  

You're reading it!
###Camera Calibration

####1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the function "find_calibration_params" (Lines 78-112).  

I staredt by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `obj_points` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `img_points` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection. 
Most of the chessboard images were 9x6, but some were is different size so I scaned a little bit around 9x6 in order to find and use all the 20 chessboard images that were given. 

I then used the output `obj_points` and `img_points` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result:
 
On the first chessboard image:
![alt text][image1]

On a randon chessboard image:
![alt text][image2]

###Pipeline (single images)

There is a main process_image() function (lines 676-751) that we used both for the single images and for the video.
Before we run this function we calibrate the camera and we find the birdeye transform matrix.


####1. Provide an example of a distortion-corrected image.
To correct the distortion we use the function "calibrate_road_image" that simply use cv2.undistort() with the parameters we found before.
To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image3]
You can see the effect of the distortion on the white car in the right side of the image.

####2. Describe how you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.
I used a combination of color and gradient thresholds to generate a binary image. 

**gradient thresholds**

The function "sobel_binary_th" (Lines 154-220) will use the gradients in order to create a thresholded binary image. There are 4 quantities we are using:

- abs_sobel_x: absolute on the gradient in x-axis
- abs_sobel_y: absolute on the gradient in y-axis
- mag_sobel: gradient magnitude = sqrt(abs_sobel_x^2+abs_sobel_y^2)
- angle_soble: the angle of the gradient = atan(abs_sobel_y/abs_sobel_x)

Each one has its own threshold and we combine them in the following way:

final_output_soble = (B[abs_sobel_x] & B[abs_sobel_y]) | (B[mag_sobel] & B[angle_soble])

B[*] is the binary threshold.
After some tries I set the thresholds so I'll only use abs_sobel_x with thresholds of [30,200]. (B[abs_sobel_y] was set to all ones, and B[mag_sobel], B[angle_soble] to all zeros)
The Kernel size was set to 3.
Here is an example:
![alt text][image4]

**Color thresholds**

The function "color_binary_th" (Lines 231-278) will use a color channels to create a thresholded binary image. 
It will use two channels: 

- The S channel in HLS color space 
- The R channel in RGB color space 

Each one has its own threshold and we combine them in the following way:

final_output_color = (B[s_channel] | B[r_channel])

B[*] is the binary threshold
After some tries I set the thresholds so I'll only use s_channel with thresholds of [175,255]. (B[r_channel] was set to all zeros)
Here is an example:
![alt text][image5]

**apply binary thresholds**

The function "apply_binary_th" (Lines 291-317) apply the two binary images (from the sobel and from the color space) in the following way:

final_output = final_output_sobel | final_output_color

Here's an example of my output for this step. 

![alt text][image6]

####3. Describe how you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `find_birdeye_matrix()` (Lines 331-392).  The `find_birdeye_matrix()` reads the two "straight_lines" images that were given to us and for each image we hardcoded the source points:

    y_down = [689,689]           # for the two images
    y_up = [450,450]             # for the two images
    x_right_down = [1055,1062]   # for the two images
    x_left_down = [250,260]      # for the two images
    x_right_up = [683,688]       # for the two images
    x_left_up = [596,595]        # for the two images
so the four points we use are:

	A_src: x_left_down[i],y_down[i]
	B_src: x_left_up[i]  ,y_up[i]
	C_src: x_right_up[i]  ,y_up[i]
	D_src: x_right_down[i],y_down[i]

the destination points were selected to be a symetrical trapeze:

	A_dst: x_left_down[i],y_down[i]
	B_dst: x_left_down[i],0
	C_dst: x_right_down[i],0
	D_dst: x_right_down[i],y_down[i]

This function will return the transform matrix.
I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image. To apply this matrix we simply use "cv2.warpPerspective()" with the matrix we found.

Here is the "straight_lines" images after we calculated the birdeye matrix:
![alt text][image7]

For the project we used only the first matrix.

####4. Describe how you identified lane-line pixels and fit their positions with a polynomial?

The function "fit_lane_line()" (Lines 482-602) will take the "birdeye" binary threshold image and will fit a 2ed order polynomial. There are two options here:

- startover=True: Using the histogram algorithm that was shown in class we first find the first rectangle to scan (both for the left and right lanes) and we continue from there until we have all the left lane and right lane points. 
- startover=False: We will use a given polynomial (usually from the previous frame) in order to sacn arounf it for relevant points (Lines 544-545).

Once we have the relevant points we use numpy.polyfit() to fit the 2ed order polynomial to each lane.

Example when "startover"=True:
![alt text][image8]

Exanple "startover"=False, we use a given polynomial (one for the left lane and one for the right) in order to search aroud them:
![alt text][image9]

####5. Describe how you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The function "calc_curve_offset()" will calculate the curve and the offset. 

**curvature**

It takes the left and right lanes 2ed order fits (in the pixels domain), we have a pixel to meters transformation:

    ym_per_pix = 30/img_shape[1]     # meters per pixel in y dimension
    xm_per_pix = 3.7/lane_width_pix  # meters per pixel in x dimension
we need to find the fit in the real world (meters). We can simply scale the pixel world fit in the following way:

    scale_vec = np.float32([xm_per_pix/(ym_per_pix**2), xm_per_pix/ym_per_pix, xm_per_pix])
    left_fit_cr = left_fit*scale_vec
    right_fit_cr = right_fit*scale_vec
Now we have the fit in the "meters world" and we use the curvature formula to find the curvature for each lane. We will latter can average those two curvatures to get one finle curvature estimation.

**Offset**

we simply find the left and right lane origin in pixels and then, by assuming the camera is in the middle of the car and by using xm_per_pix we can easily find the offset:

    offset = ((left_lane_org_x+right_lane_org_x)/2.0 - img_shape[0]/2.0) # assume the camera in in the middle of the car
    offset *= (100.0*xm_per_pix) # switching from pixels to cm
 

####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in the function "draw_lane_lines()" (Lines 615-651). We simple use the inverse matrix of the birdeye transformation on the polynomial we found.
Here is an example of my result on a test image:

![alt text][image10]

---

###Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

We pipeline each image. The stratover parameter for the poly fit was set to True for the first frame and from then it was False unless we were not able to detect lanes in 3 consecutive frames. In order to decide if a lane is detected or not we used the expected curvature and the lanes origin location.
We also avraged the lane fit in the currect frame with the estimation we had so far (IIR filter). the IIR coeff was set differenty in the case that we were able to detect a lane and for the case that we didn't detect a lane. (Lines 718-733) 
Here's a [link to my video result](./project_video_with_lane_est.mp4)

In the "detected_case_vec" we collect data on the number of frames that were detected successfully (first bin) and for the rest of the frames we see the reason that they were declared to be not good (in fucntion "is_good_lanes"):

- We were not able to fit any line (no points were found)
- wrong position of left line near the car
- wrong position of right line near the car
- wrong position of wrong left curve
- wrong position of wrong right curve

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?


- One problem was that we had to manually configure the threshold for all the binary thresholding we did. There are too many threshold (and combinations of binary images to take) and this is not such a good way to go beacuse we can find ourselves in the different lighting/shadow condition that our thresholds will not be good enough. Using the CNN from the previous project was much more robust...


- We tried to average (using an IIR filter: y[n]=(1-a)y[n-1]+ax[n]) the results we got from each frame in order to filter out or to smooth the fit we got from each frame (of course it was also helpful that the left lane was continues).


- We took advantage of the fact that we know where (on which area) the car was driving, so we know what is the expected curvature. In the real world it would be more difficult although we still have the standard (in each country) to relay on. 
  
