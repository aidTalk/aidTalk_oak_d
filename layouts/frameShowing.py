# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'frameShowing.ui'
#
# Created by: PyQt5 UI code generator 5.12.3
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_frameShowing(object):
    def setupUi(self, frameShowing):
        frameShowing.setObjectName("frameShowing")
        frameShowing.resize(800, 600)
        self.gridLayout = QtWidgets.QGridLayout(frameShowing)
        self.gridLayout.setObjectName("gridLayout")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.image_label = QtWidgets.QLabel(frameShowing)
        self.image_label.setObjectName("image_label")
        self.verticalLayout.addWidget(self.image_label)
        self.gridLayout.addLayout(self.verticalLayout, 0, 0, 1, 1)

        self.retranslateUi(frameShowing)
        QtCore.QMetaObject.connectSlotsByName(frameShowing)

    def retranslateUi(self, frameShowing):
        _translate = QtCore.QCoreApplication.translate
        frameShowing.setWindowTitle(_translate("frameShowing", "frameShowing"))
        self.image_label.setText(_translate("frameShowing", "TextLabel"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    frameShowing = QtWidgets.QDialog()
    ui = Ui_frameShowing()
    ui.setupUi(frameShowing)
    frameShowing.show()
    sys.exit(app.exec_())
