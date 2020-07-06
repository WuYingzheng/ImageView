from PyQt5.QtWidgets    import QWidget
from PyQt5.QtGui        import *
from PyQt5.QtCore       import Qt, QPoint, pyqtSignal, QRect, QSize
import numpy as np

class LumaSliderWidget(QWidget):
    hue = 0
    sat = 0
    luma = 255
    voff = 5
    hoff = 1
    painter = None
    sizeIndict = 5
    valueChanged = pyqtSignal(int)
    def __init__(self, parent=0):
        super().__init__()
        self.pen = QPen(QColor(0,0,0))                # set lineColor
        self.pen.setWidth(1)                          # set lineWidth
        self.brush = QBrush(QColor(0,0,0))            # set fillColor
        self.func = (None, None)
        self.update()

    def y2val(self, y):
        val = 255 * (self.height() - 1 - y - self.voff)/(self.barSize.height()-1)
        return val

    def val2y(self, val):
        y = self.height() -1 - (val/255 * (self.barSize.height() -1 )+ self.voff)
        return y

    def setVal(self, y):
        self.luma = self.y2val(y)
        if self.luma < 0:
            self.luma = 0
        elif self.luma > 255:
            self.luma = 255
        self.valueChanged.emit(self.luma)       
        self.update()

    def mouseMoveEvent(self, event: QMouseEvent):
        self.setVal(event.y())

    def mousePressEvent(self, event: QMouseEvent):
        self.setVal(event.y())

    def paintEvent(self, event):
        wi = self.width() - 2 * self.hoff - self.sizeIndict
        hi = self.height() - 2 * self.voff
        self.barSize = QSize(wi, hi)

        painter = QPainter(self)
        painter.eraseRect(1, 1, wi,hi)
        painter.setPen(self.pen)

        color = QColor(255, 255, 255, 255)
        array = np.zeros(wi*hi, dtype="uint32")

        for y in range(0, hi):
            val = ((hi-1) - y) / (hi-1) * 255
            color.setHsv(self.hue, self.sat, val)
            line = np.full(wi, color.rgb())
            start = y * wi
            end = start + wi
            array[start:end] = line

        image = QImage(array.data, wi, hi, QImage.Format_RGB32)
        pixmap = QPixmap.fromImage(image)
        painter.drawPixmap(self.hoff, self.voff, pixmap)

        point1 = QPoint(wi+self.hoff, self.val2y(self.luma))
        point2 = QPoint(point1.x() + self.sizeIndict , point1.y() - self.sizeIndict)
        point3 = QPoint(point1.x() + self.sizeIndict , point1.y() + self.sizeIndict)
        points = [point1, point2, point3]
        a = QPolygon(points)
        painter.setBrush(self.brush)   
        painter.drawPolygon(a)

class MonitorWidget(QWidget):
    cursorMoved = pyqtSignal(int, int)
    def __init__(self, parent=0, hoff=0, voff=0):
        super().__init__()
        self.hoff = hoff
        self.voff = voff
        self.pos = QPoint(hoff, voff)
        self.cursorSize = 8
        self.pixmap = None

    def updateImage(self, image:QImage):
        self.pixmap = QPixmap.fromImage(image)
        self.update()

    def paintEvent(self, event):
        w = self.width()
        h = self.height()
        painter = QPainter(self)
        if self.pixmap is not None:
            painter.drawPixmap(0, self.voff, self.pixmap.scaled(w, h))
        painter.setPen(QPen(QColor(0,0,0), 2))
        a = QPoint(self.pos.x() - self.cursorSize, self.pos.y())
        b = QPoint(self.pos.x() + self.cursorSize, self.pos.y())
        c = QPoint(self.pos.x(), self.pos.y() - self.cursorSize)
        d = QPoint(self.pos.x(), self.pos.y() + self.cursorSize)
        painter.drawLine(a, b)
        painter.drawLine(c, d)

    def resizeEvent(self, event):
        pass

    def mouseMoveEvent(self, event:QMouseEvent):
        self.updateInfo(event.x(), event.y())
        self.update()

    def mousePressEvent(self, event: QMouseEvent):
        self.updateInfo(event.x(), event.y())
        self.update()

    def updateInfo(self, x, y):
        xmin = self.hoff
        ymin = self.voff
        xmax = self.width() - self.hoff - 1
        ymax = self.height() - self.voff - 1
        if (x < xmin):
            x = xmin
        if (y < ymin):
            y = ymin
        if (x > xmax):
            x = xmax
        if (y > ymax):
            y = ymax
        self.pos = QPoint(x, y)
        self.cursorMoved.emit(x - self.hoff, y - self.voff)

class HsvPanelWidget(MonitorWidget):
    colorRect = None
    hue = 0
    sat = 0
    val = 200
    colorChanged = pyqtSignal(int, int)
    def __init__(self, parent=0, image = None, hoff = 0, voff = 0):
        super().__init__()
        self.hoff = hoff
        self.voff = voff
        self.pos = QPoint(self.hoff, self.voff)
        wi = self.width() - 2 * self.hoff
        hi = self.height() - 2 * self.voff
        start = QPoint(self.hoff, self.voff)
        self.colorRect = QRect(start, QSize(wi, hi))
        self.pixmap = None
        if image is not None:
            self.pixmap = QPixmap.fromImage(image)
        else:
            color = QColor(255, 255, 255, 255)
            self.array = np.zeros(wi * hi, dtype="uint32")
            for y in range(0, hi):
                for x in range(0, wi):
                    point = QPoint(x, y)
                    color.setHsv(self.huePt(point), self.satPt(point), self.val)
                    self.array[y*wi + x] = color.rgb()
            image = QImage(self.array.data, wi, hi, QImage.Format_RGB32)
            self.pixmap = QPixmap.fromImage(image)
        self.update()

    def setVal(self, x, y):
        xmin = self.hoff
        ymin = self.voff
        xmax = self.width() - self.hoff - 1
        ymax = self.height() - self.voff - 1
        if (x < xmin):
            x = xmin
        if (y < ymin):
            y = ymin
        if (x > xmax):
            x = xmax
        if (y > ymax):
            y = ymax
        self.pos = QPoint(x, y)
        self.hue = self.huePt(QPoint(x-self.hoff, y-self.voff))
        self.sat = self.satPt(QPoint(x-self.hoff, y-self.voff))
        self.colorChanged.emit(self.hue, self.sat)

    # relative (x, y) to colorRect
    def huePt(self, point):
        return 360 - point.x() *360 / (self.colorRect.width() - 1)

    # point: relative (x, y) to colorRect
    def satPt(self, point):
        return 255 - point.y() *255 / (self.colorRect.height() - 1)

class ColorWidget(QWidget):
    color = QColor(255, 255, 255)
    def __init__(self, color=QColor(255, 255, 255), parent=None):
        super().__init__()
        self.update()

    def setColor(self, color):
        self.color = color
        self.update()

    def paintEvent(self, event):
        w = self.width()
        h = self.height()
        array = np.full(w * h, self.color.rgb(), dtype="uint32")
        image = QImage(array.data, w, h, QImage.Format_RGB32)
        pixmap = QPixmap.fromImage(image)
        painter = QPainter(self)
        painter.drawPixmap(0, 0, pixmap)
