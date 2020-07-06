.PHONY: all clean

all:
	pyuic5 -o GuiLib/UI_ColorPickDialog.py  GuiLib/ColorPickDialog.ui
	pyuic5 -o GuiLib/UI_YUView.py           GuiLib/YUView.ui
	pyuic5 -o GuiLib/UI_ImageSave.py        GuiLib/ImageSave.ui

clean:
	rm -f  GuiLib/UI_ColorPickDialog.py
