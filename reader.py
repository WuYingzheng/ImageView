import cv2
import numpy as np
import io

# Building the input:
###############################################################################
img = cv2.imread('data/image.png')

#yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
#y, u, v = cv2.split(yuv)

# Convert BGR to YCrCb (YCrCb apply YCrCb JPEG (or YCC), "full range", 
# where Y range is [0, 255], and U, V range is [0, 255] (this is the default JPEG format color space format).
yvu = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
y, v, u = cv2.split(yvu)

print("type: ", type(u), u.shape)
# Downsample U and V (apply 420 format).
u = cv2.resize(u, (u.shape[1]//2, u.shape[0]//2))
v = cv2.resize(v, (v.shape[1]//2, v.shape[0]//2))

# Open In-memory bytes streams (instead of using fifo)
f = io.BytesIO()

# Write Y, U and V to the "streams".
f.write(y.tobytes())
f.write(u.tobytes())
f.write(v.tobytes())

f.seek(0)
###############################################################################

# Read YUV420 (I420 planar format) and convert to BGR
###############################################################################
data = f.read(y.size*3//2)  # Read one frame (number of bytes is width*height*1.5).


fd = open("spike_i420_640x360.yuv", "wb")
fd.write(data)
fd.close()

# Reshape data to numpy array with height*1.5 rows
yuv_data = np.frombuffer(data, np.uint8).reshape(y.shape[0]*3//2, y.shape[1])


# Convert YUV to BGR
bgr = cv2.cvtColor(yuv_data, cv2.COLOR_YUV2BGR_I420)

# Display result:
cv2.imshow("COLOR_YUV2BGR_I420", bgr)

cv2.waitKey(0)


# How to How should I be placing the u and v channel information in all_yuv_data?
# -------------------------------------------------------------------------------
# Example: place the channels one after the other (for a single frame)
f.seek(0)
y0 = f.read(y.size)
u0 = f.read(y.size//4)
v0 = f.read(y.size//4)
yuv_data = y0 + u0 + v0
yuv_data = np.frombuffer(yuv_data, np.uint8).reshape(y.shape[0]*3//2, y.shape[1])
bgr = cv2.cvtColor(yuv_data, cv2.COLOR_YUV2BGR_I420)
###############################################################################

# Display result:
cv2.imshow("bgr incorrect colors", bgr)
cv2.waitKey(0)

###############################################################################
f.seek(0)
y = np.frombuffer(f.read(y.size), dtype=np.uint8).reshape((y.shape[0], y.shape[1]))  # Read Y color channel and reshape to height x width numpy array
u = np.frombuffer(f.read(y.size//4), dtype=np.uint8).reshape((y.shape[0]//2, y.shape[1]//2))  # Read U color channel and reshape to height x width numpy array
v = np.frombuffer(f.read(y.size//4), dtype=np.uint8).reshape((y.shape[0]//2, y.shape[1]//2))  # Read V color channel and reshape to height x width numpy array

# Resize u and v color channels to be the same size as y
u = cv2.resize(u, (y.shape[1], y.shape[0]))
v = cv2.resize(v, (y.shape[1], y.shape[0]))

array = np.array([[1, 2, 3, 4], [4, 5, 6, 9], [20, 11, 30, 4]])
cv2.resize(array, (4,6))


yvu = cv2.merge((y, v, u)) # Stack planes to 3D matrix (use Y,V,U ordering)

bgr = cv2.cvtColor(yvu, cv2.COLOR_YCrCb2BGR)
###############################################################################


# Display result:
cv2.imshow("bgr", bgr)
cv2.waitKey(0)
cv2.destroyAllWindows()


"""
with open("image.y", 'rb') as w:
    byteArray = np.frombuffer(w.read(), dtype=np.uint8)

Y = np.reshape(byteArray, (367, 237))

cv2.imshow('yuv:y', Y)

img = cv2.imread('image.jpg')
print(img.shape)

# cv2.imshow('Grayscale image', gray_img)

cv2.imshow('Y channel', yuv_img[:, :, 0])
cv2.imshow('U channel', yuv_img[:, :, 1])
cv2.imshow('V channel', yuv_img[:, :, 2])

print("Y: {}, U: {}, V: {}".format(yuv_img[:, :, 0].shape, yuv_img[:, :, 1].shape, yuv_img[:, :, 2].shape))
# yuv_img[:, :, 0].astype('int8').tofile("image.y")

cv2.waitKey(0)

# https://stackoverflow.com/questions/54566713/how-to-convert-yuv-420-888-to-bgr-using-opencv-python

# Define width and height of image
w,h = 640,480

# Create black-white gradient from top to bottom in Y channel
f = lambda i, j: int((i*256)/h)
Y = np.fromfunction(np.vectorize(f), (h,w)).astype(np.uint8) 
# DEBUG
cv2.imwrite('Y.jpg',Y)

# Dimensions of subsampled U and V
UVwidth, UVheight = w//2, h//2

# U is a black-white gradient from left to right
f = lambda i, j: int((j*256)/UVwidth)
U = np.fromfunction(np.vectorize(f), (UVheight,UVwidth)).astype(np.uint8)  
# DEBUG 
cv2.imwrite('U.jpg',U)

So, in summary, I believe a 2x2 image in NV21 image is stored with interleaved VU, like this:

Y Y Y Y V U V U
and a 2x2 NV12 image is stored with interleaved UV, like this:

Y Y Y Y U V U V
and a YUV420 image (Raspberry Pi) is stored fully planar:

Y Y Y Y U U V V
"""
