from PyQt5.QtWidgets    import QDialog, QWidget, QMainWindow, QFileDialog,  QListWidgetItem
from PyQt5.QtGui        import *
from PyQt5.QtCore       import Qt, QPoint, pyqtSlot, QRect, QSize, QCoreApplication
from UI_YUView          import Ui_YUView
from ImageSave          import ImageSave

import numpy as np
from enum import Enum
import inspect, os, ntpath
import cv2

from utility import *

def createHSV(w, h):
    color = QColor(255, 255, 255, 255)
    array = np.zeros(w * h, dtype="uint32")
    for y in range(0, h):
        for x in range(0, w):
            point = QPoint(x, y)
            color.setHsv(self.huePt(point), self.satPt(point), self.val)
            array[y*wi + x] = color.rgb()
    image = QImage(self.array.data, wi, hi, QImage.Format_RGB32)

class YUVItem(object):
    def __init__(self, path):
        self.path = path

class RAWItem(object):
    def __init__(self, path):
        self.path = path

class YUView(QMainWindow, Ui_YUView):
    def __init__(self, parent=0):
        super().__init__()
        self.setupUi(self)
        self.ImageItem = []
        self.current = None
        self.component = YUV_COMPONENT.EMPTY
        self.list_raw = []
        self.list_yuv = []
        self.try_update()
        # init an idle image item
        # self.idleImage = ImplYUV()
        # self.idleImage.fromByteArray()

    def update_listWidget_CFA(self):
        for raw in self.list_raw:
            item = QListWidgetItem()
            item.setText(_translate("listView", ntpath.basename(raw.path)))
            self.listWidget_CFA.addItem(item)

    def update_listWidget_YUV(self):
        _translate = QCoreApplication.translate
        self.listWidget_YUV.clear()
        for image in self.ImageItem:
            item = QListWidgetItem()
            item.setText(_translate("listView", ntpath.basename(image.path)))
            self.listWidget_YUV.addItem(item)

    def try_update(self):
        self.update_listWidget_CFA()
        self.update_listWidget_YUV()
        if not self.current:
            return
        index = self.comboBox_FMT.currentIndex()
        pitch = self.spinBox_BufferP.value()
        height = self.spinBox_BufferH.value()
        self.current.updateInfo(YUV_FMT(index), pitch, height)
        index = self.comboBox_Compo.currentIndex()

        image = None
        if YUV_COMPONENT(index) is YUV_COMPONENT.YCbCr:
            image = self.current.QImageRGB()
        elif YUV_COMPONENT(index) is YUV_COMPONENT.Y_ONLY:
            image = self.current.QImageY()
        elif YUV_COMPONENT(index) is YUV_COMPONENT.U_ONLY:
            image = self.current.QImageU()
        elif YUV_COMPONENT(index) is YUV_COMPONENT.V_ONLY:
            image = self.current.QImageV()
        else:
            print("Error: components is fault.")

        if image is not None:
            self.widget_Image.updateImage(image)

    @pyqtSlot(bool)
    def on_action_OpenFile_triggered(self):
        path = ""
        dialog = QFileDialog(self)
        if (dialog.exec()):
            path = dialog.selectedFiles()[0]
            self.current = ImplYUV(path)
            self.ImageItem.append(self.current)
            self.try_update()

    @pyqtSlot(bool)
    def on_action_Save_triggered(self):
        diag = ImageSave()
        if (diag.exec()):
            filename = diag.filename
            fmt = diag.format
            if (diag.enableYUV):
                fd = open(filename + ".yuv", "wb")
                fd.write(self.current.YUV().tobytes())
                fd.close()
            if (diag.enableY):
                fd = open(filename + ".y", "wb")
                fd.write(self.current.Y().tobytes())
                fd.close()
            if (diag.enableUV):
                fd = open(filename + ".uv", "wb")
                fd.write(self.current.UV().tobytes())
                fd.close()

    @pyqtSlot(int)
    def on_spinBox_BufferP_valueChanged(self, value):
        print("pitch is ", value)
        self.try_update()

    @pyqtSlot(int)
    def on_spinBox_BufferH_valueChanged(self, value):
        print("height is ", value)
        self.try_update()

    @pyqtSlot(int)
    def on_comboBox_Compo_currentIndexChanged(self, index):
        self.try_update()

    @pyqtSlot(int, int)
    def on_widget_Image_cursorMoved(self, x, y):
        cx = x / (self.widget_Image.width() - 1) * (self.spinBox_BufferP.value() - 1)
        cy = y / (self.widget_Image.height() - 1) * (self.spinBox_BufferH.value() - 1)
        self.spinBox_PixelX.setValue(int(cx))
        self.spinBox_PixelY.setValue(int(cy))
        if self.current:
            lu, u, v = self.current.atPointYUV(int(cx), int(cy))
            self.spinBox_Y.setValue(lu)
            self.spinBox_U.setValue(u)
            self.spinBox_V.setValue(v)
            r, g, b = yuv2rgb(lu, u, v)
            self.widget_Color.setColor(QColor(r, g, b))

#------------------------------------------------------------------------------

"""
with open("3840x2160.yuv", 'rb') as w:
    pitch = 3840
    height = 2160
    byteArray = np.frombuffer(w.read(), dtype=np.uint8)
    y = byteArray[0 : pitch*height]
    Y = np.reshape(y, (height, pitch))

cv2.imshow('yuv:y', Y)
"""
