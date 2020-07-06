from PyQt5.QtWidgets    import QDialog, QWidget
from PyQt5.QtGui        import *
from PyQt5.QtCore       import Qt, QPoint, pyqtSlot, QRect, QSize
from UI_ColorPickDialog import Ui_ColorPickDialog
import numpy as np

class ColorPicker(QDialog, Ui_ColorPickDialog):
    colorpic = None
    color = QColor()
    hue = 0
    sat = 0
    val = 200
    def __init__(self, parent=0):
        super().__init__()
        self.setupUi(self)
        self.hue = self.widget_HsvPanel.hue
        self.sat = self.widget_HsvPanel.sat
        self.val = self.widget_Luma.luma
        self.color.setHsv(self.hue, self.sat, self.val)
        self.widget_Color.setColor(self.color)

    def updateValue(self, hue, sat, val):
        self.color.setHsv(hue, sat, val)
        yuv = self.rgb2yuv(self.color.red(), self.color.green(), self.color.blue())
        self.spinBox_Hue.setValue(hue)
        self.spinBox_Sat.setValue(sat)
        self.spinBox_Val.setValue(val)
        self.spinBox_Red.setValue(self.color.red())
        self.spinBox_Green.setValue(self.color.green())
        self.spinBox_Blue.setValue(self.color.blue())
        self.spinBox_Y.setValue(yuv[0])
        self.spinBox_U.setValue(yuv[1])
        self.spinBox_V.setValue(yuv[2])
        self.widget_Color.setColor(self.color)

    @pyqtSlot(int, int)
    def on_widget_HsvPanel_colorChanged(self, hue, sat):
        self.hue = hue
        self.sat = sat
        self.widget_Luma.hue = hue
        self.widget_Luma.sat = sat
        self.widget_Luma.update()
        self.updateValue(hue, sat, self.val)

    @pyqtSlot(int)
    def on_widget_Luma_valueChanged(self, val):
        self.val = val
        self.updateValue(self.hue, self.sat, self.val)

    def rgb2yuv(self, r, g, b):
        y =  0.257 * r + 0.504 * g + 0.098 * b +  16
        u = -0.148 * r - 0.291 * g + 0.439 * b + 128
        v =  0.439 * r - 0.368 * g - 0.071 * b + 128
        return [y, u, v]

    def yuv2rgb(y, u, v):
        R = 1.164 * Y + 1.596 * V
        G = 1.164 * Y - 0.392 * U - 0.813 * V
        B = 1.164 * Y + 2.017 * U
