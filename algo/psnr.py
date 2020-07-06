import numpy
import math

"""
Peak signal-to-noise ratio (PSNR) is the ratio between the maximum possible power of an
image and the power of corrupting noise that affects the quality of its representation.
To estimate the PSNR of an image, it is necessary to compare that image to an ideal
clean image with the maximum possible power.

https://www.geeksforgeeks.org/python-peak-signal-to-noise-ratio-psnr/

"""

def psnr(ori, los):
    mse = np.mean((ori - los) ** 2)   # 计算平均值
    if (mse == 0):         # it means no noise in signal
        return 100
    max_pixel = 255.0
    p = 20 * math.log10(max_pixel / math.sqrt(mse))
    return p