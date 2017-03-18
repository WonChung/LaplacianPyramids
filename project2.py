from __future__ import print_function

import cv2
import numpy as np
import sys
import struct
import pdb

#TODO: add hw_starter3 command line argument reader
#TODO: Do w/ a color/greyscale image?

#load image
# source = cv2.imread('lenna.png', cv2.IMREAD_GRAYSCALE)
source = cv2.imread('bernie.jpg').astype(np.float32)
source2 = cv2.imread('justin.jpg').astype(np.float32)
trump = cv2.imread('trump_before.jpg').astype(np.float32)
###
# Alignment (This is not our TO GO EVEN FURTHER BEYOND - It's in the report - Research and Study)

#########
# Read Images
bernie_before = cv2.imread('bernie_before.jpg')
justin_before = cv2.imread('justin_before.jpg')

# Resize Images
bernie_before = cv2.resize(bernie_before, (400, 400))
justin_before = cv2.resize(justin_before, (400, 400))

# Width and Height of Image
width, height,channels = bernie_before.shape

# Show Before Images
cv2.imshow('Original: bernie Before', bernie_before)
cv2.waitKey()
cv2.imshow('Original: justin Before', justin_before)
cv2.waitKey()

# Transform the First Image
M = np.float32([[1,0,-20],[0,1,-15]])
bernie_after = cv2.warpAffine(bernie_before,M,(height,width))

# Show the Images After the Transformation
cv2.imshow('Aligned: bernie After',bernie_after)
cv2.waitKey()
cv2.imshow('Aligned: justin After',justin_before)
cv2.waitKey()

# Write After Images
cv2.imwrite('bernie_after.jpg', bernie_after)
cv2.imwrite('justin_after.jpg', justin_before)

# Read the After Images
bernie_after = cv2.imread('bernie_after.jpg').astype(np.float32)
justin_after = cv2.imread('justin_after.jpg').astype(np.float32)
#########

# Professor, below is the old code.
# We left it since it technically works but is not efficent
# Also it took us a long time, so we didn't want to erase it

#########

#OLD CODE - START

#########
# def pyr_build(image): # OLD pyr_build
# 	minSize = 15 # size of the smallest pyramid
# 	image.astype(np.float32)
# 	w, h, channels = image.shape # width, height, and channels of image
#
# 	pyramid_name_tuples = createPyramids(source, minSize) #creates the regular gaussian pyramid
#
# 	#MAKE SURE THERE ARE NO image parameter name conflicts
# 	#convert the pyramids into laplacian
# 	laplacian = []
# 	for i in range(0, len(pyramid_name_tuples)-1):
# 		Size = (pyramid_name_tuples[i][0].shape[1], pyramid_name_tuples[i][0].shape[0])
# 		currentLayer = pyramid_name_tuples[i][0].astype(np.float32)
# 		upscaled = cv2.pyrUp(pyramid_name_tuples[i+1][0].astype(np.float32), dstsize=Size)
# 		pair = (cv2.subtract(currentLayer, upscaled), i)
# 		laplacian.append(pair)
#
# 	#display pyramids
# 	#TODO: last layer is just a grey block
# 	#TODO: the last layer shouldnt be a grey block. it hsould be a smaller version of the original image.
# 	lastLayer = (pyramid_name_tuples[-1][0], len(pyramid_name_tuples)-1)
# 	laplacian.append(lastLayer) #this may break later
#
# 	#display original pyramids
# 	for image, text in pyramid_name_tuples:
# 		cv2.imshow(str(text), image)
#
# 	#display laplacian pyramids
#
# 	for image in laplacian:
# 		cv2.imshow(str(image[1]), 0.5 + 0.5*(image[0] / np.abs(image[0]).max()))
# 	#need this line when you convert to a function
# 	laplacianArray = [i[0] for i in laplacian]
# 	return laplacianArray
#########
# def createPyramids(image, minSize):
# 	pyramid_name_tuples=[]
# 	i=0
# 	while True:
# 		if(image.shape[0]>minSize and image.shape[1]>minSize): #run recursively until pyramid size is 15
# 			pair = (image, image.shape)
# 			pyramid_name_tuples.append(pair)
# 			print(i, image.shape)
# 			i=i+1
# 			image = cv2.pyrDown(image)
# 		else:
# 			print("too small, breaking out ",i, image.shape)
# 			return pyramid_name_tuples
# 			break
#########
# def pyr_reconstruct(lp): # OLD pyr_reconstruct
# 	reconstructed = lp[-1] #start reconstructing fromt he last element, Ln
# 	gaussianLayer = createPyramids(source, 15)
# 	for i in range(len(lp)-2, 0,-1): #this should traverse the list in reverse order. start from -1 to avoid last element
# 		#TODO: Call pyr_build(img) instead of directly accessing pyramid_name_tuples
# 		reconstructed = np.add(lp[i].astype(np.float32), cv2.pyrUp(gaussianLayer[i+1][0]).astype(np.float32)) #make sure lp[i] starts from the last indexp
#
# 	cv2.imshow("reconstructed image", reconstructed.astype(np.uint8))
# 	while True:
# 	    k = cv2.waitKey(15)
# 	    k = np.uint8(k).view(np.int8)
# 	    if k >= 0:
# 	        break
# 	return reconstructed
#########
# OLD CODE - END
#########
# Part 1 pyr_build()
def pyr_build(image):
	# image.astype(np.float32)
	w, h, channels = image.shape # width, height, and channels of image

	laplacianArray = []
	while (h > 7 and w > 7):
		# Down Sampling
		down = cv2.pyrDown(image, None)

		# Up Sampling
		up = cv2.pyrUp(down, None, (h,w))

		# Append the Laplacian Image
		laplacianArray.append(image-up)

		# Update variables
		image = down
		w, h, channels = image.shape

	# Add Last Image and Return the Laplacian Array
	laplacianArray.append(image)
	return laplacianArray

# Test pyr_build
trumpImg = pyr_build(trump)

for image in trumpImg:
    cv2.imshow("Pyramid Build: Trump", 0.5 + 0.5*(image / np.abs(image).max()))
    cv2.waitKey()

temp = cv2.GaussianBlur(trump, (25,25), 15)
trumpImgBlur = pyr_build(temp)
for image in trumpImgBlur:
 	cv2.imshow("Pyramid Build Blur: Trump", 0.5 + 0.5*(image / np.abs(image).max()))
	cv2.waitKey()

# Part 2 pyr_reconstruct()
def pyr_reconstruct(lp):
	reconstructed = lp[-1] #start reconstructing fromt he last element, Ln
	# Traverse the list in reverse order, starting from len(lp)-1 to avoid the last element
	for i in range(len(lp)-1, 0, -1):
		w, h, channels = lp[i-1].shape # Get new width, height each time
		up = cv2.pyrUp(reconstructed, None, (h,w)) # Up sampling on each new reconstructed
		reconstructed = lp[i-1] + up # Set reconstructed to new up sampled image
	return reconstructed

# Test if Reconstruction is Correct
reconstructedImage = pyr_reconstruct(pyr_build(source))
cv2.imshow("Reconstructed Image", np.clip(reconstructedImage, 0, 255).astype(np.uint8))
cv2.waitKey()

# Part 3 laplacian_blend()
# Laplacian Blending and Traditional Blending
def laplacian_blend(imgA, imgB):
	# Specify width, height, center points, angle, and sigma
	# Resize the Images
	w, h = 400, 400
	imgA = cv2.resize(imgA, (w, h))
	imgB = cv2.resize(imgB, (w, h))
	cx = w/2
	cy = h/2
	angle=0
	sigma=15

	# Create Mask and ellipse
	# Perform Gaussian Blur
	mask = np.zeros((w, h), dtype=np.float32)
	cv2.ellipse(mask, (cx, cy),(w/4, h/3), angle, 0, 360, (255, 255, 255), -1,cv2.LINE_AA)
	mask = mask/255
	mask_blurred = cv2.GaussianBlur(mask, (25,25), sigma)

	# Show Blurred mask
	cv2.imshow("Blurred Mask using GaussianBlur", mask_blurred)
	cv2.waitKey()

	# Traditional Blending
	alphaBlending = alpha_blend(imgA, imgB, mask_blurred)
	cv2.imshow("Traditional Blending", np.clip(alphaBlending, 0, 255).astype(np.uint8))
	cv2.waitKey()

	# Laplacian Blending
	#now we give each layer and the rescaled alpha mask to alpha blend
	#create a new array to contain each layer that has been blended.
	pyramidsA = pyr_build(imgA)
	pyramidsB = pyr_build(imgB)
	blendedLayers = []
	for i in range(0, len(pyramidsA)): #Note for William: the size of the pyramid layer decreases as the index in the list increases
		blendedLayers.append(alpha_blend(pyramidsA[i],pyramidsB[i], cv2.resize(mask_blurred, (pyramidsA[i].shape[1],pyramidsA[i].shape[0]))))
	# blendedLayers.reverse() #we are appending so we need to flip the list and the end - NVMD we don't

	# Show Laplacian Blended Image
	blendedLayers = pyr_reconstruct(blendedLayers)
	cv2.imshow("Laplacian Blending", np.clip(blendedLayers, 0, 255).astype(np.uint8))
	cv2.waitKey()
	# return blendedLayers
	return np.clip(blendedLayers, 0, 255).astype(np.uint8)

#alpha_blend method given by Matt
def alpha_blend(A, B, alpha):
	A = A.astype(alpha.dtype)
	B = B.astype(alpha.dtype)
	# if A and B are RGB images, we must pad
	# out alpha to be the right shape
	if len(A.shape) == 3:
	    alpha = np.expand_dims(alpha, 2)
	return A + alpha*(B-A)

# Test Laplacian Blend
# laplacian_blend(trump, hillary)
# laplacian_blend(source, source2)
laplacian_blend(bernie_after, justin_after)

# Hybrid Imaging Part
def hybridImaging(imgA, imgB):
	imgA = cv2.resize(imgA, (400, 400))
	imgB = cv2.resize(imgB, (400, 400))
	wA, hA, channelsA = imgA.shape
	wB, hB, channelsB = imgB.shape

	sigmaA = 15.0
	lowPassA = cv2.GaussianBlur(imgA, (25, 25), sigmaA)

	sigmaB = 3.0
	lowPassB = cv2.GaussianBlur(imgB, (7,7), sigmaB)
	highPassB = imgB - lowPassB

	I = 1.5*lowPassA + 2.0*highPassB
	cv2.imshow("Hybrid Image", np.clip(I, 0, 255).astype(np.uint8))
	cv2.waitKey()

	laplacian = pyr_build(I)
	for image in laplacian:
	    cv2.imshow("Hybrid Image Laplacian", 0.5 + 0.5*(image / np.abs(image).max()))
	    cv2.waitKey()

# Test Hybrid Image
# hybridImaging(source, source2)
# hybridImaging(trump, hillary)
hybridImaging(bernie_after, justin_after)


#########
# MORE OLD CODE
#########

# Extra: TO GO EVEN FURTHER BEYOND - SUPER SAIYAN 3
# def superSaiyanWilliam(imgC):
# 	input_filename = 'superSaiyanWilliam.mp4'
# 	capture = cv2.VideoCapture(input_filename)
# 	ok, frame = capture.read()
# 	# laplacian_blend(frame, imgC)
# 	frameNumber=0; #Used to keep track of the frame number of the video
# 	fps = 30
# 	fourcc, ext = (cv2.VideoWriter_fourcc('D', 'I', 'V', 'X'), 'avi')
# 	filename = 'captured.'+ext
# 	writer = cv2.VideoWriter(filename, fourcc, fps, (400, 400))
# 	while(frameNumber < 500):
# 		frameNumber = frameNumber + 1 #increment frameNumber
# 		# print('frame number ', frameNumber) #print current frame number if desired
# 		ok, frame = capture.read(frame)
# 		video = laplacian_blend(frame, imgC)
# 		# laplacian_blend(frame, imgC)
# 		# Bail if none.
# 		if frame is None:
# 			print('Video Finished!')
# 			break
# 		if not ok:
# 			print('Bad frame in video! Aborting!')
# 			break
# 		if writer:
# 			writer.write(video)
# 			# Throw it up on the screen.
# 			cv2.imshow('Video', video)

# superSaiyanWilliam(source3)

#########
# MORE OLD CODE END
#########
