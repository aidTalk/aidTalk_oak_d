# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'menuWindow.ui'
#
# Created by: PyQt5 UI code generator 5.12.3
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(647, 402)
        Dialog.setStyleSheet("background-color: rgb(255, 224, 196);")
        self.gridLayout = QtWidgets.QGridLayout(Dialog)
        self.gridLayout.setObjectName("gridLayout")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setContentsMargins(0, -1, -1, -1)
        self.verticalLayout.setObjectName("verticalLayout")
        self.text_label = QtWidgets.QLabel(Dialog)
        font = QtGui.QFont()
        font.setPointSize(20)
        font.setBold(True)
        font.setWeight(75)
        self.text_label.setFont(font)
        self.text_label.setAlignment(QtCore.Qt.AlignCenter)
        self.text_label.setObjectName("text_label")
        self.verticalLayout.addWidget(self.text_label)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.PB_Gaze = QtWidgets.QPushButton(Dialog)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.PB_Gaze.sizePolicy().hasHeightForWidth())
        self.PB_Gaze.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(18)
        self.PB_Gaze.setFont(font)
        self.PB_Gaze.setStyleSheet("QPushButton{\n"
"background-color: rgb(214, 217, 217);\n"
"}\n"
"\n"
"QPushButton::hover{\n"
"    background-color : lightgreen\n"
"}")
        self.PB_Gaze.setObjectName("PB_Gaze")
        self.horizontalLayout.addWidget(self.PB_Gaze)
        self.PB_Head = QtWidgets.QPushButton(Dialog)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.PB_Head.sizePolicy().hasHeightForWidth())
        self.PB_Head.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(18)
        self.PB_Head.setFont(font)
        self.PB_Head.setStyleSheet("QPushButton{\n"
"background-color: rgb(214, 217, 217);\n"
"}\n"
"\n"
"QPushButton::hover{\n"
"    background-color : lightgreen\n"
"}")
        self.PB_Head.setObjectName("PB_Head")
        self.horizontalLayout.addWidget(self.PB_Head)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.gridLayout.addLayout(self.verticalLayout, 0, 0, 1, 1)

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.text_label.setText(_translate("Dialog", "¿Cómo desea controlar el teclado?"))
        self.PB_Gaze.setText(_translate("Dialog", "Controlar con la vista"))
        self.PB_Head.setText(_translate("Dialog", "Controlar con la cabeza"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    ui = Ui_Dialog()
    ui.setupUi(Dialog)
    Dialog.show()
    sys.exit(app.exec_())
