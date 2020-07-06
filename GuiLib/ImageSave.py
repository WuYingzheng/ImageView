from UI_ImageSave import Ui_Dialog_ImageSave
from PyQt5.QtWidgets    import QDialog
from PyQt5.QtCore       import pyqtSlot

from utility import *

class ImageSave(QDialog, Ui_Dialog_ImageSave):
    def __init__(self, parent=0):
        super().__init__()
        self.setupUi(self)
        self.format    = YUV_FMT.UNKNOW
        self.enableYUV = False
        self.enableY   = False
        self.enableUV  = False
        self.filename  = ""

    @pyqtSlot()
    def on_buttonBox_accepted(self):
        index = self.comboBox_YUVFormat.currentIndex()
        self.format = YUV_FMT(index)
        if self.checkBox_YUV.isChecked():
            self.enableYUV = True
        if self.checkBox_Y.isChecked():
            self.enableY = True
        if self.checkBox_UV.isChecked():
            self.enableUV = True
        self.filename = self.lineEdit_FileName.text()
