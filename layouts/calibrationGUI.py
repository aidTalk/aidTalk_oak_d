# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'calibrationGUI.ui'
#
# Created by: PyQt5 UI code generator 5.12.3
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(1280, 800)
        Dialog.setStyleSheet("background-color: rgb(245, 219, 189);")
        self.label = QtWidgets.QLabel(Dialog)
        self.label.setGeometry(QtCore.QRect(430, 240, 441, 71))
        font = QtGui.QFont()
        font.setPointSize(25)
        font.setBold(True)
        font.setWeight(75)
        font.setStrikeOut(False)
        font.setKerning(True)
        self.label.setFont(font)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.calibrationDotBase = QtWidgets.QFrame(Dialog)
        self.calibrationDotBase.setGeometry(QtCore.QRect(615, 345, 59, 59))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.calibrationDotBase.sizePolicy().hasHeightForWidth())
        self.calibrationDotBase.setSizePolicy(sizePolicy)
        self.calibrationDotBase.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.calibrationDotBase.setFrameShadow(QtWidgets.QFrame.Raised)
        self.calibrationDotBase.setObjectName("calibrationDotBase")
        self.calibrationDotCross_1 = QtWidgets.QFrame(self.calibrationDotBase)
        self.calibrationDotCross_1.setGeometry(QtCore.QRect(0, 0, 60, 60))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.calibrationDotCross_1.sizePolicy().hasHeightForWidth())
        self.calibrationDotCross_1.setSizePolicy(sizePolicy)
        self.calibrationDotCross_1.setStyleSheet("QFrame{\n"
"background-color: qlineargradient(spread:repeat, x1:0, y1:0, x2:1, y2:0, stop:0.48003 rgba(255, 255, 255, 0), stop:0.48154 rgba(0, 0, 0, 255), stop:0.52001 rgba(0, 0, 0, 255), stop:0.52112 rgba(255, 255, 255, 0))\n"
"}")
        self.calibrationDotCross_1.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.calibrationDotCross_1.setFrameShadow(QtWidgets.QFrame.Raised)
        self.calibrationDotCross_1.setObjectName("calibrationDotCross_1")
        self.calibrationDotBorder = QtWidgets.QFrame(self.calibrationDotBase)
        self.calibrationDotBorder.setGeometry(QtCore.QRect(0, 0, 60, 60))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.calibrationDotBorder.sizePolicy().hasHeightForWidth())
        self.calibrationDotBorder.setSizePolicy(sizePolicy)
        self.calibrationDotBorder.setStyleSheet("QFrame{\n"
"background-color: qradialgradient(spread:pad, cx:0.5, cy:0.5, radius:0.95, fx:0.5, fy:0.5, stop:0.392962 rgba(0, 0, 0, 0), stop:0.394428 rgba(0, 0, 0, 255), stop:0.470674 rgba(0, 0, 0, 255), stop:0.473607 rgba(0, 0, 0, 0));\n"
"}")
        self.calibrationDotBorder.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.calibrationDotBorder.setFrameShadow(QtWidgets.QFrame.Raised)
        self.calibrationDotBorder.setObjectName("calibrationDotBorder")
        self.calibrationDotCenter = QtWidgets.QFrame(self.calibrationDotBase)
        self.calibrationDotCenter.setGeometry(QtCore.QRect(0, 0, 60, 60))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.calibrationDotCenter.sizePolicy().hasHeightForWidth())
        self.calibrationDotCenter.setSizePolicy(sizePolicy)
        self.calibrationDotCenter.setStyleSheet("QFrame{\n"
"background-color: qradialgradient(spread:pad, cx:0.5, cy:0.5, radius:0.2, fx:0.5, fy:0.5, stop:0 rgba(0, 0, 0, 255), stop:0.519685 rgba(0, 0, 0, 255), stop:0.524752 rgba(255, 255, 255, 0), stop:0.99802 rgba(255, 255, 255, 0));\n"
"}")
        self.calibrationDotCenter.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.calibrationDotCenter.setFrameShadow(QtWidgets.QFrame.Raised)
        self.calibrationDotCenter.setObjectName("calibrationDotCenter")
        self.calibrationDotBg = QtWidgets.QFrame(self.calibrationDotBase)
        self.calibrationDotBg.setGeometry(QtCore.QRect(0, 0, 60, 60))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.calibrationDotBg.sizePolicy().hasHeightForWidth())
        self.calibrationDotBg.setSizePolicy(sizePolicy)
        self.calibrationDotBg.setStyleSheet("QFrame{\n"
"background-color: qradialgradient(spread:pad, cx:0.5, cy:0.5, radius:0.85, fx:0.5, fy:0.5, stop:0 rgba(255, 0, 0, 255), stop:0.519685 rgba(255, 0, 0, 255), stop:0.524752 rgba(255, 255, 255, 0), stop:0.99802 rgba(255, 255, 255, 0));\n"
"}")
        self.calibrationDotBg.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.calibrationDotBg.setFrameShadow(QtWidgets.QFrame.Raised)
        self.calibrationDotBg.setObjectName("calibrationDotBg")
        self.calibrationDotCross_2 = QtWidgets.QFrame(self.calibrationDotBase)
        self.calibrationDotCross_2.setGeometry(QtCore.QRect(0, 0, 60, 60))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.calibrationDotCross_2.sizePolicy().hasHeightForWidth())
        self.calibrationDotCross_2.setSizePolicy(sizePolicy)
        self.calibrationDotCross_2.setStyleSheet("QFrame{\n"
"background-color: qlineargradient(spread:repeat, x1:0, y1:0, x2:0, y2:1, stop:0.48003 rgba(255, 255, 255, 0), stop:0.48154 rgba(0, 0, 0, 255), stop:0.52001 rgba(0, 0, 0, 255), stop:0.52112 rgba(255, 255, 255, 0))\n"
"}")
        self.calibrationDotCross_2.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.calibrationDotCross_2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.calibrationDotCross_2.setObjectName("calibrationDotCross_2")
        self.calibrationDotBg.raise_()
        self.calibrationDotBorder.raise_()
        self.calibrationDotCenter.raise_()
        self.calibrationDotCross_1.raise_()
        self.calibrationDotCross_2.raise_()

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.label.setText(_translate("Dialog", "Mira fijamente al centro de los puntos"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    ui = Ui_Dialog()
    ui.setupUi(Dialog)
    Dialog.show()
    sys.exit(app.exec_())
