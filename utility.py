from PyQt5.QtGui        import *


from enum import Enum
import inspect, os, ntpath
from abc import ABC, abstractmethod

import cv2
import scipy.ndimage
import numpy as np

class ImageFmt(Enum):
    UNKNOW          = -1
    YUV420_IYUV     = 0
    YUV420_YV12     = 1
    YUV420_NV12     = 2
    YUV422_YU16     = 3
    YUV422_YV16     = 4
    YUV422_NV16     = 5
    YUV444_YU24     = 6
    YUV444_YV24     = 7
    YUV444_NV24     = 8

    RGB         = 9
    BGR         = 10

class PixelDepth(Enum):
    UNKNOW          = -1

    IS_8_BIT    = 0
    IS_10_BIT   = 1
    IS_12_BIT   = 2
    IS_16_BIT   = 3
    IS_32_BIT   = 4



def IsYUV420(fmt:ImageFmt):
    if fmt is ImageFmt.YUV420_IYUV or \
        fmt is ImageFmt.YUV420_YV12 or \
        fmt is ImageFmt.YUV420_NV12:
        return True
    else:
        return False

def IsYUV422(fmt:ImageFmt):
    if fmt is ImageFmt.YUV422_YU16 or \
        fmt is ImageFmt.YUV422_YV16 or \
        fmt is ImageFmt.YUV422_NV16:
        return True 
    else:
        return False

def IsYUV444(fmt:ImageFmt):
    if fmt is ImageFmt.YUV444_YU16 or \
        fmt is ImageFmt.YUV444_YV16 or \
        fmt is ImageFmt.YUV444_NV16:
        return True 
    else:
        return False


class YUV_COMPONENT(Enum):
    EMPTY    = -1
    YCbCr    = 0
    Y_ONLY   = 1
    U_ONLY  = 2
    V_ONLY  = 3

class ColorSpace(object):
    def __init__(self):
        self.qcolor = QColor()
        self.y = 0
        self.u = 0
        self.v = 0
        self.r = 0
        self.g = 0
        self.b = 0
        self.hue = 0
        self.sat = 0
        self.val = 0

    def RGB(self):
        return [self.r, self.g, self.b]

    def YUV(self):
        return [self.y, self.u, self.v]

    def HSV(self):
        return [self.hue, self.sat, self.val]

    def update(self):
        pass

def clamp(num, min_value, max_value):
   return max(min(num, max_value), min_value)

# https://www.mikekohn.net/file_formats/yuv_rgb_converter.php
def rgb2yuv(R, G, B):
    y = R *  .299000 + G *  .587000 + B *  .114000
    u = R * -.168736 + G * -.331264 + B *  .500000 + 128
    v = R *  .500000 + G * -.418688 + B * -.081312 + 128
    return y, u, v

def yuv2rgb(Y:int, U:int, V:int):
    R = Y + 1.4075 * (V - 128)
    G = Y - 0.3455 * (U - 128) - (0.7169 * (V - 128))
    B = Y + 1.7790 * (U - 128)
    R = clamp(R, 0, 255)
    G = clamp(G, 0, 255)
    B = clamp(B, 0, 255)
    return R, G, B

"""
generally read a file from file system
"""
class ImageTerminator(ABC):
    def __init__(self, fmt=None, depth=PixelDepth.IS_8_BIT, byteArray=None):
        self.byteArray = byteArray
        self.fmt       = fmt
        self.depth     = depth

    @abstractmethod
    def QImageRGB(self):
        pass

    @abstractmethod
    def QImageY(self):
        pass

    @abstractmethod
    def QImageU(self):
        pass

    @abstractmethod
    def QImageV(self):
        pass

    @abstractmethod
    def CVImageRGB(self):
        pass

class ImplYUV(ImageTerminator):
    def __init__(self,
        fmt,
        pitch = 0,
        height = 0,
        depth=PixelDepth.IS_8_BIT,
        array=None):
        super.__init__(fmt, depth, array)
        self.ArrayY    = None      # One dimision array for Y with fmt 444
        self.ArrayU    = None      # One dimision array for Y with fmt 444
        self.ArrayV    = None      # One dimision array for Y with fmt 444
        self.fmt       = fmt
        self.pitch     = self.pitch
        self.height    = self.height
        self.depth     = 0

        if byteArray is not None:
            self.fromByteArray(byteArray)
        else:
            pass

    def fromFile(self, file):
        assert isinstance(file, str)
        try:
            fd = open(file, 'rb+')    # open file in read & write mode
            self.byteArray = np.frombuffer(fd.read(), np.uint8)
        except:
            print("can't open file: ", path)

        if IsYUV420(self.fmt) or IsYUV422(self.fmt) or IsYUV444(self.fmt):
            self.ArrayY = splictYfromYUV(self.byteArray)
            self.ArrayU = splictUfromYUV(self.byteArray)
            self.arrayV = splictVfromYUV(self.byteArray)
        else:
            pass

    def fromByteArray(self, byteArray):
        assert isinstance(byteArray, (numpy.ndarray))
        self.byteArray = byteArray

    def QImageRGB(self):
        image = None
        rgb = self.byteArrayRGB()
        if rgb is not None:
            image = QImage(rgb.data, self.pitch, self.height, QImage.Format_RGB888)
        return image

    def QImageY(self):
        image = None
        y = self.byteArrayY()
        if y is not None:
            image = QImage(y.data, self.pitch, self.height, QImage.Format_Indexed8)
        return image

    def QImageU(self):
        image = None
        u = self.byteArrayU()
        w = self.pitch
        h = self.height
        if IsYUV420(self.fmt):
            w = w >> 1
            h = h >> 1
        elif IsYUV422(self.fmt):
            w = w >> 1
        else:
            pass

        if u is not None:
            image = QImage(u.data, w, h, QImage.Format_Indexed8)
        return image

    def QImageV(self):
        image = None
        v = self.byteArrayV()
        w = self.pitch
        h = self.height
        if IsYUV420(self.fmt):
            w = w >> 1
            h = h >> 1
        elif IsYUV422(self.fmt):
            w = w >> 1
        else:
            pass

        if v is not None:
            image = QImage(v.data, w, h, QImage.Format_Indexed8)
        return image

    def byteArrayRGB(self):
        rgb = None
        self.fd.seek(0, 0)
        byteArray = np.frombuffer(self.fd.read(), np.uint8)
        total = (self.height*3//2) * self.pitch
        byteArray = byteArray[0 : total]
        yuv = np.reshape(byteArray, (self.height*3//2, self.pitch))
        rgb = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB_I420)
        return rgb

    # return raw byte array of Y data sequence
    def splictYfromYUV(self, array):
        assert isinstance(array, np.ndarray)

        arrayY = array[0 : self.pitch * self.height]
        try:
            arrayY = np.reshape(arrayY, (self.height, self.pitch))
        except:
            print("Y: can't not resharp data to [{}, {}]".format(self.height, self.pitch))
        return arrayY

    def splictUfromYUV(self, array, fmt=ImageFmt.YUV420_IYUV):
        assert isinstance(array, np.ndarray)
        base = self.pitch * self.height
        if (fmt is ImageFmt.YUV420_IYUV):
            offset = (self.pitch >> 1) * (self.height >> 1)
            arrayU = array[base : base + offset]
            arrayU = np.reshape(arrayU, ((self.height >> 1), (self.pitch >> 1)))
            # Upsample U and V (apply 420 format).
            arrayU = cv2.resize(arrayU, (self.height, self.pitch))
        elif (fmt is ImageFmt.YUV420_YV12):
            pass
        else:
            print("U: format not supported!")

        return arrayU

    def splictVfromYUV(self, array, fmt=ImageFmt.YUV420_IYUV):
        assert isinstance(array, np.ndarray)
        base = self.pitch * self.height

        if (fmt is ImageFmt.YUV420_IYUV):
            byteArray = byteArray[(self.pitch >> 1) * (self.height >> 1) : (self.pitch) * (self.height >> 1)]
        if (fmt is ImageFmt.YUV420_YV12):
            byteArray = byteArray[0 : (self.pitch >> 1) * (self.height >> 1)]
        if (fmt is ImageFmt.YUV420_NV12):
            byteArray = yuv_img[ :, 1]
        if (fmt is ImageFmt.YUV422_YU16):
            byteArray = byteArray[self.pitch * (self.height >> 1) : self.pitch * self.height]
        if (fmt is ImageFmt.YUV422_YV16):
            byteArray = byteArray[0 : self.pitch * (self.height >> 1)]
        if (fmt is ImageFmt.YUV422_NV16):
            byteArray = byteArray[ :, 1]

        v = np.reshape(byteArray, ((self.pitch >> 1), (self.height >> 1)))

        return v





    def byteArrayUV(self, fmt = None):
        if fmt == None:
            fmt = self.fmt
        U = None
        self.fd.seek(self.pitch * self.height, 0)
        byteArray = np.frombuffer(self.fd.read(), dtype=np.uint8)
        try:
            if (self.fmt == ImageFmt.YUV420):
                u = byteArray[0 : (self.pitch >> 1) * (self.height >> 1)]
                U = np.reshape(u, (self.pitch >> 1), (self.height >> 1))
        except:
            print("U: can't not resharp data to [{} , {}]".format(self.height >> 1, self.pitch >> 1))
        return U

    def convertUV(self, fmt):
        if fmt is ImageFmt.YUV420_NV12:
            pass

    def updateInfo(self, fmt, pitch, height):
        self.pitch  = pitch
        self.height = height
        self.fmt    = fmt

        self.ArrayY = self.byteArrayY()
        self.ArrayU = self.byteArrayU()
        self.arrayV = self.byteArrayV()


    def atPointYUV(self, x, y):
        luma = 0
        ch_u = 0
        ch_v = 0
        self.fd.seek(self.pitch * y + x, 0)
        luma = np.frombuffer(self.fd.read(1), dtype=np.uint8)[0]
        base = self.pitch * self.height
        if (self.fmt is ImageFmt.YUV420_IYUV):
            offset = (self.pitch >> 1) * (y >> 1) + (x >> 1)
            self.fd.seek(base + offset, 0)
            ch_u = np.frombuffer(self.fd.read(1), dtype=np.uint8)[0]
            offset += (self.pitch >> 1) * (self.height >> 1)
            self.fd.seek(base + offset, 0)
            ch_v = np.frombuffer(self.fd.read(1), dtype=np.uint8)[0]
        elif (self.fmt is ImageFmt.YUV420_YV12):
            pass
        else:
            print("unsupported yet")
        return luma, ch_u, ch_v

def ndarray_zoom_scaling(label, new_h, new_w):
    """
    Implement scaling for ndarray with scipy.ndimage.zoom
    :param label: [H, W] or [H, W, C]
    :return: label_new: [new_h, new_w] or [new_h, new_w, C]
    Examples
    --------
    ori_arr = np.array([[1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9]], dtype=np.int32)
    new_arr = ndarray_zoom_scaling(ori_arr, new_h=6, new_w=6)
    print(new_arr)
    [[1 1 2 2 3 3]
    [1 1 2 2 3 3]
    [4 4 5 5 6 6]
    [4 4 5 5 6 6]
    [7 7 8 8 9 9]
    [7 7 8 8 9 9]]
    """
    scale_h = new_h / label.shape[0]
    scale_w = new_w / label.shape[1]
    if len(label.shape) == 2:
        label_new = scipy.ndimage.zoom(label, zoom=[scale_h, scale_w], order=0)
    else:
        label_new = scipy.ndimage.zoom(label, zoom=[scale_h, scale_w, 1], order=0)
    return label_new

ori_arr = np.array([[1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9]], dtype=np.int32)
new_arr = ndarray_zoom_scaling(ori_arr, new_h=6, new_w=6)
print(new_arr)