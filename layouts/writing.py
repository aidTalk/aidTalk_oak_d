# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'writing.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(792, 644)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(11)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(Dialog.sizePolicy().hasHeightForWidth())
        Dialog.setSizePolicy(sizePolicy)
        Dialog.setFocusPolicy(QtCore.Qt.NoFocus)
        Dialog.setStyleSheet("")
        self.gridLayout_2 = QtWidgets.QGridLayout(Dialog)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setContentsMargins(10, 5, 10, -1)
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setContentsMargins(0, 0, -1, 0)
        self.horizontalLayout_2.setSpacing(10)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.pushButton13 = QtWidgets.QPushButton(Dialog)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton13.sizePolicy().hasHeightForWidth())
        self.pushButton13.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(28)
        self.pushButton13.setFont(font)
        self.pushButton13.setStyleSheet("QPushButton:hover\n"
" {\n"
"    background-color:  rgb(164, 255, 225)\n"
"}\n"
"")
        self.pushButton13.setObjectName("pushButton13")
        self.horizontalLayout_2.addWidget(self.pushButton13)
        self.textEdit = QtWidgets.QTextEdit(Dialog)
        font = QtGui.QFont()
        font.setPointSize(28)
        self.textEdit.setFont(font)
        self.textEdit.setObjectName("textEdit")
        self.horizontalLayout_2.addWidget(self.textEdit)
        self.pushButton12 = QtWidgets.QPushButton(Dialog)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton12.sizePolicy().hasHeightForWidth())
        self.pushButton12.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(28)
        self.pushButton12.setFont(font)
        self.pushButton12.setStyleSheet("QPushButton:hover\n"
" {\n"
"    background-color:  rgb(164, 255, 225)\n"
"}\n"
"")
        self.pushButton12.setObjectName("pushButton12")
        self.horizontalLayout_2.addWidget(self.pushButton12)
        self.horizontalLayout_2.setStretch(0, 1)
        self.horizontalLayout_2.setStretch(1, 6)
        self.horizontalLayout_2.setStretch(2, 1)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_6.setContentsMargins(-1, -1, 0, -1)
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.gridLayout_3 = QtWidgets.QGridLayout()
        self.gridLayout_3.setContentsMargins(0, 5, 5, 5)
        self.gridLayout_3.setSpacing(15)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.pushButton5 = QtWidgets.QPushButton(Dialog)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton5.sizePolicy().hasHeightForWidth())
        self.pushButton5.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(48)
        self.pushButton5.setFont(font)
        self.pushButton5.setStyleSheet("QPushButton:hover\n"
" {\n"
"    background-color:  rgb(164, 255, 225)\n"
"}\n"
"")
        self.pushButton5.setObjectName("pushButton5")
        self.gridLayout_3.addWidget(self.pushButton5, 2, 1, 1, 1)
        self.pushButton4 = QtWidgets.QPushButton(Dialog)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton4.sizePolicy().hasHeightForWidth())
        self.pushButton4.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(48)
        self.pushButton4.setFont(font)
        self.pushButton4.setStyleSheet("QPushButton:hover\n"
" {\n"
"    background-color:  rgb(164, 255, 225)\n"
"}\n"
"")
        self.pushButton4.setObjectName("pushButton4")
        self.gridLayout_3.addWidget(self.pushButton4, 2, 0, 1, 1)
        self.pushButton3 = QtWidgets.QPushButton(Dialog)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton3.sizePolicy().hasHeightForWidth())
        self.pushButton3.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(48)
        self.pushButton3.setFont(font)
        self.pushButton3.setStyleSheet("QPushButton:hover\n"
" {\n"
"    background-color:  rgb(164, 255, 225)\n"
"}\n"
"")
        self.pushButton3.setObjectName("pushButton3")
        self.gridLayout_3.addWidget(self.pushButton3, 1, 2, 1, 1)
        self.pushButton6 = QtWidgets.QPushButton(Dialog)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton6.sizePolicy().hasHeightForWidth())
        self.pushButton6.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(48)
        self.pushButton6.setFont(font)
        self.pushButton6.setStyleSheet("QPushButton:hover\n"
" {\n"
"    background-color:  rgb(164, 255, 225)\n"
"}\n"
"")
        self.pushButton6.setObjectName("pushButton6")
        self.gridLayout_3.addWidget(self.pushButton6, 2, 2, 1, 1)
        self.pushButton2 = QtWidgets.QPushButton(Dialog)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton2.sizePolicy().hasHeightForWidth())
        self.pushButton2.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(48)
        self.pushButton2.setFont(font)
        self.pushButton2.setStyleSheet("QPushButton:hover\n"
" {\n"
"    background-color:  rgb(164, 255, 225)\n"
"}\n"
"")
        self.pushButton2.setObjectName("pushButton2")
        self.gridLayout_3.addWidget(self.pushButton2, 1, 1, 1, 1)
        self.pushButton9 = QtWidgets.QPushButton(Dialog)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton9.sizePolicy().hasHeightForWidth())
        self.pushButton9.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(48)
        self.pushButton9.setFont(font)
        self.pushButton9.setStyleSheet("QPushButton:hover\n"
" {\n"
"    background-color:  rgb(164, 255, 225)\n"
"}\n"
"")
        self.pushButton9.setObjectName("pushButton9")
        self.gridLayout_3.addWidget(self.pushButton9, 3, 2, 1, 1)
        self.pushButton7 = QtWidgets.QPushButton(Dialog)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton7.sizePolicy().hasHeightForWidth())
        self.pushButton7.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(48)
        self.pushButton7.setFont(font)
        self.pushButton7.setStyleSheet("QPushButton:hover\n"
" {\n"
"    background-color:  rgb(164, 255, 225)\n"
"}\n"
"")
        self.pushButton7.setObjectName("pushButton7")
        self.gridLayout_3.addWidget(self.pushButton7, 3, 0, 1, 1)
        self.pushButton8 = QtWidgets.QPushButton(Dialog)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton8.sizePolicy().hasHeightForWidth())
        self.pushButton8.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(48)
        self.pushButton8.setFont(font)
        self.pushButton8.setStyleSheet("QPushButton:hover\n"
" {\n"
"    background-color:  rgb(164, 255, 225)\n"
"}\n"
"")
        self.pushButton8.setObjectName("pushButton8")
        self.gridLayout_3.addWidget(self.pushButton8, 3, 1, 1, 1)
        self.pushButton1 = QtWidgets.QPushButton(Dialog)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton1.sizePolicy().hasHeightForWidth())
        self.pushButton1.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(48)
        self.pushButton1.setFont(font)
        self.pushButton1.setStyleSheet("\n"
"QPushButton:hover\n"
" {\n"
"    background-color:  rgb(164, 255, 225)\n"
"}\n"
"\n"
"")
        self.pushButton1.setObjectName("pushButton1")
        self.gridLayout_3.addWidget(self.pushButton1, 1, 0, 1, 1)
        self.horizontalLayout_6.addLayout(self.gridLayout_3)
        self.verticalLayout_5 = QtWidgets.QVBoxLayout()
        self.verticalLayout_5.setContentsMargins(10, 5, 0, 5)
        self.verticalLayout_5.setSpacing(15)
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.pushButton10 = QtWidgets.QPushButton(Dialog)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton10.sizePolicy().hasHeightForWidth())
        self.pushButton10.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(48)
        self.pushButton10.setFont(font)
        self.pushButton10.setStyleSheet("QPushButton:hover\n"
" {\n"
"    background-color:  rgb(164, 255, 225)\n"
"}\n"
"")
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("../../../../Downloads/Altavoz Alfa black.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.pushButton10.setIcon(icon)
        self.pushButton10.setIconSize(QtCore.QSize(30, 30))
        self.pushButton10.setObjectName("pushButton10")
        self.verticalLayout_5.addWidget(self.pushButton10)
        self.pushButton = QtWidgets.QPushButton(Dialog)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton.sizePolicy().hasHeightForWidth())
        self.pushButton.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(48)
        self.pushButton.setFont(font)
        self.pushButton.setStyleSheet("QPushButton:hover\n"
" {\n"
"    background-color:  rgb(195, 0, 0)\n"
"}")
        self.pushButton.setObjectName("pushButton")
        self.verticalLayout_5.addWidget(self.pushButton)
        self.pushButton11 = QtWidgets.QPushButton(Dialog)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton11.sizePolicy().hasHeightForWidth())
        self.pushButton11.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(48)
        self.pushButton11.setFont(font)
        self.pushButton11.setStyleSheet("QPushButton:hover\n"
" {\n"
"    background-color:  rgb(164, 255, 225)\n"
"}\n"
"")
        self.pushButton11.setObjectName("pushButton11")
        self.verticalLayout_5.addWidget(self.pushButton11)
        self.verticalLayout_5.setStretch(0, 2)
        self.verticalLayout_5.setStretch(1, 1)
        self.verticalLayout_5.setStretch(2, 2)
        self.horizontalLayout_6.addLayout(self.verticalLayout_5)
        self.horizontalLayout_6.setStretch(0, 3)
        self.horizontalLayout_6.setStretch(1, 1)
        self.verticalLayout.addLayout(self.horizontalLayout_6)
        self.verticalLayout.setStretch(0, 1)
        self.verticalLayout.setStretch(1, 20)
        self.gridLayout_2.addLayout(self.verticalLayout, 0, 0, 1, 1)

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.pushButton13.setText(_translate("Dialog", "MENU"))
        self.pushButton12.setText(_translate("Dialog", "EXIT"))
        self.pushButton5.setText(_translate("Dialog", "MNOÑ"))
        self.pushButton4.setText(_translate("Dialog", "JKL"))
        self.pushButton3.setText(_translate("Dialog", "GHI"))
        self.pushButton6.setText(_translate("Dialog", "PQRS"))
        self.pushButton2.setText(_translate("Dialog", "DEF"))
        self.pushButton9.setText(_translate("Dialog", "_?!,"))
        self.pushButton7.setText(_translate("Dialog", "TUV"))
        self.pushButton8.setText(_translate("Dialog", "WXYZ"))
        self.pushButton1.setText(_translate("Dialog", "ABC"))
        self.pushButton10.setText(_translate("Dialog", "PLAY"))
        self.pushButton.setText(_translate("Dialog", "HELP"))
        self.pushButton11.setText(_translate("Dialog", "DEL"))

