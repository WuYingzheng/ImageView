#!/usr/bin/env python3
import sys, os
sys.path.append(os.path.dirname(__file__) + "/GuiLib")

from YUView import YUView
from PyQt5.QtWidgets import QApplication


qapp = QApplication(sys.argv)
app = YUView()
app.show()
sys.exit(qapp.exec_())