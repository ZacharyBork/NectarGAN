# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'base.ui'
##
## Created by: Qt User Interface Compiler version 6.9.0
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QAction, QBrush, QColor, QConicalGradient,
    QCursor, QFont, QFontDatabase, QGradient,
    QIcon, QImage, QKeySequence, QLinearGradient,
    QPainter, QPalette, QPixmap, QRadialGradient,
    QTransform)
from PySide6.QtWidgets import (QAbstractScrollArea, QApplication, QCheckBox, QComboBox,
    QDockWidget, QDoubleSpinBox, QFormLayout, QFrame,
    QGroupBox, QHBoxLayout, QLCDNumber, QLabel,
    QLineEdit, QMainWindow, QMenu, QMenuBar,
    QProgressBar, QPushButton, QScrollArea, QSizePolicy,
    QSlider, QSpacerItem, QSpinBox, QStatusBar,
    QTabWidget, QTextEdit, QVBoxLayout, QWidget)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(1639, 925)
        self.actionLoad_From_Config = QAction(MainWindow)
        self.actionLoad_From_Config.setObjectName(u"actionLoad_From_Config")
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.verticalLayout = QVBoxLayout(self.centralwidget)
        self.verticalLayout.setSpacing(3)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.frame = QFrame(self.centralwidget)
        self.frame.setObjectName(u"frame")
        self.frame.setFrameShape(QFrame.StyledPanel)
        self.frame.setFrameShadow(QFrame.Raised)
        self.horizontalLayout = QHBoxLayout(self.frame)
        self.horizontalLayout.setSpacing(0)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.close_experiment_settings = QPushButton(self.frame)
        self.close_experiment_settings.setObjectName(u"close_experiment_settings")
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.close_experiment_settings.sizePolicy().hasHeightForWidth())
        self.close_experiment_settings.setSizePolicy(sizePolicy)
        self.close_experiment_settings.setMaximumSize(QSize(11, 16777215))
        font = QFont()
        font.setPointSize(12)
        font.setBold(False)
        font.setItalic(False)
        font.setUnderline(False)
        self.close_experiment_settings.setFont(font)
        self.close_experiment_settings.setIconSize(QSize(8, 8))

        self.horizontalLayout.addWidget(self.close_experiment_settings)

        self.frame_7 = QFrame(self.frame)
        self.frame_7.setObjectName(u"frame_7")
        self.frame_7.setFrameShape(QFrame.Panel)
        self.frame_7.setFrameShadow(QFrame.Sunken)
        self.verticalLayout_8 = QVBoxLayout(self.frame_7)
        self.verticalLayout_8.setSpacing(3)
        self.verticalLayout_8.setObjectName(u"verticalLayout_8")
        self.verticalLayout_8.setContentsMargins(3, 3, 3, 3)
        self.frame_12 = QFrame(self.frame_7)
        self.frame_12.setObjectName(u"frame_12")
        sizePolicy1 = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.frame_12.sizePolicy().hasHeightForWidth())
        self.frame_12.setSizePolicy(sizePolicy1)
        self.frame_12.setFrameShape(QFrame.StyledPanel)
        self.frame_12.setFrameShadow(QFrame.Sunken)
        self.horizontalLayout_15 = QHBoxLayout(self.frame_12)
        self.horizontalLayout_15.setObjectName(u"horizontalLayout_15")
        self.horizontalLayout_15.setContentsMargins(2, 2, 2, 2)
        self.label_19 = QLabel(self.frame_12)
        self.label_19.setObjectName(u"label_19")
        font1 = QFont()
        font1.setPointSize(14)
        font1.setBold(True)
        self.label_19.setFont(font1)
        self.label_19.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_15.addWidget(self.label_19)

        self.line = QFrame(self.frame_12)
        self.line.setObjectName(u"line")
        self.line.setFrameShape(QFrame.Shape.VLine)
        self.line.setFrameShadow(QFrame.Shadow.Sunken)

        self.horizontalLayout_15.addWidget(self.line)

        self.label_20 = QLabel(self.frame_12)
        self.label_20.setObjectName(u"label_20")
        self.label_20.setFont(font1)
        self.label_20.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_15.addWidget(self.label_20)

        self.line_2 = QFrame(self.frame_12)
        self.line_2.setObjectName(u"line_2")
        self.line_2.setFrameShape(QFrame.Shape.VLine)
        self.line_2.setFrameShadow(QFrame.Shadow.Sunken)

        self.horizontalLayout_15.addWidget(self.line_2)

        self.label_21 = QLabel(self.frame_12)
        self.label_21.setObjectName(u"label_21")
        self.label_21.setFont(font1)
        self.label_21.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_15.addWidget(self.label_21)


        self.verticalLayout_8.addWidget(self.frame_12)

        self.frame_11 = QFrame(self.frame_7)
        self.frame_11.setObjectName(u"frame_11")
        self.frame_11.setFrameShape(QFrame.StyledPanel)
        self.frame_11.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_13 = QHBoxLayout(self.frame_11)
        self.horizontalLayout_13.setObjectName(u"horizontalLayout_13")
        self.x_label = QLabel(self.frame_11)
        self.x_label.setObjectName(u"x_label")
        sizePolicy2 = QSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        sizePolicy2.setHorizontalStretch(0)
        sizePolicy2.setVerticalStretch(0)
        sizePolicy2.setHeightForWidth(self.x_label.sizePolicy().hasHeightForWidth())
        self.x_label.setSizePolicy(sizePolicy2)
        self.x_label.setFrameShape(QFrame.Panel)
        self.x_label.setFrameShadow(QFrame.Sunken)
        self.x_label.setAlignment(Qt.AlignHCenter|Qt.AlignTop)

        self.horizontalLayout_13.addWidget(self.x_label)

        self.y_fake_label = QLabel(self.frame_11)
        self.y_fake_label.setObjectName(u"y_fake_label")
        sizePolicy2.setHeightForWidth(self.y_fake_label.sizePolicy().hasHeightForWidth())
        self.y_fake_label.setSizePolicy(sizePolicy2)
        self.y_fake_label.setFrameShape(QFrame.Panel)
        self.y_fake_label.setFrameShadow(QFrame.Sunken)
        self.y_fake_label.setAlignment(Qt.AlignHCenter|Qt.AlignTop)

        self.horizontalLayout_13.addWidget(self.y_fake_label)

        self.y_label = QLabel(self.frame_11)
        self.y_label.setObjectName(u"y_label")
        sizePolicy2.setHeightForWidth(self.y_label.sizePolicy().hasHeightForWidth())
        self.y_label.setSizePolicy(sizePolicy2)
        self.y_label.setFrameShape(QFrame.Panel)
        self.y_label.setFrameShadow(QFrame.Sunken)
        self.y_label.setAlignment(Qt.AlignHCenter|Qt.AlignTop)

        self.horizontalLayout_13.addWidget(self.y_label)


        self.verticalLayout_8.addWidget(self.frame_11)

        self.verticalSpacer = QSpacerItem(20, 0, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Preferred)

        self.verticalLayout_8.addItem(self.verticalSpacer)

        self.frame_20 = QFrame(self.frame_7)
        self.frame_20.setObjectName(u"frame_20")
        sizePolicy1.setHeightForWidth(self.frame_20.sizePolicy().hasHeightForWidth())
        self.frame_20.setSizePolicy(sizePolicy1)
        self.frame_20.setFrameShape(QFrame.Box)
        self.frame_20.setFrameShadow(QFrame.Raised)
        self.frame_20.setLineWidth(2)
        self.verticalLayout_25 = QVBoxLayout(self.frame_20)
        self.verticalLayout_25.setObjectName(u"verticalLayout_25")
        self.verticalLayout_25.setContentsMargins(3, 3, 3, 3)
        self.horizontalLayout_31 = QHBoxLayout()
        self.horizontalLayout_31.setObjectName(u"horizontalLayout_31")
        self.label_44 = QLabel(self.frame_20)
        self.label_44.setObjectName(u"label_44")

        self.horizontalLayout_31.addWidget(self.label_44)

        self.update_frequency_slider = QSlider(self.frame_20)
        self.update_frequency_slider.setObjectName(u"update_frequency_slider")
        self.update_frequency_slider.setMinimum(1)
        self.update_frequency_slider.setMaximum(1000)
        self.update_frequency_slider.setSingleStep(10)
        self.update_frequency_slider.setPageStep(50)
        self.update_frequency_slider.setValue(50)
        self.update_frequency_slider.setOrientation(Qt.Horizontal)
        self.update_frequency_slider.setInvertedAppearance(True)
        self.update_frequency_slider.setInvertedControls(False)

        self.horizontalLayout_31.addWidget(self.update_frequency_slider)

        self.update_frequency = QSpinBox(self.frame_20)
        self.update_frequency.setObjectName(u"update_frequency")
        self.update_frequency.setMinimum(1)
        self.update_frequency.setMaximum(99999)
        self.update_frequency.setSingleStep(10)
        self.update_frequency.setValue(50)

        self.horizontalLayout_31.addWidget(self.update_frequency)


        self.verticalLayout_25.addLayout(self.horizontalLayout_31)


        self.verticalLayout_8.addWidget(self.frame_20)


        self.horizontalLayout.addWidget(self.frame_7)


        self.verticalLayout.addWidget(self.frame)

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 1639, 22))
        self.menuFile = QMenu(self.menubar)
        self.menuFile.setObjectName(u"menuFile")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.experiment_settings = QDockWidget(MainWindow)
        self.experiment_settings.setObjectName(u"experiment_settings")
        sizePolicy3 = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        sizePolicy3.setHorizontalStretch(0)
        sizePolicy3.setVerticalStretch(0)
        sizePolicy3.setHeightForWidth(self.experiment_settings.sizePolicy().hasHeightForWidth())
        self.experiment_settings.setSizePolicy(sizePolicy3)
        self.experiment_settings.setFeatures(QDockWidget.NoDockWidgetFeatures)
        self.experiment_settings.setAllowedAreas(Qt.LeftDockWidgetArea)
        self.dockWidgetContents = QWidget()
        self.dockWidgetContents.setObjectName(u"dockWidgetContents")
        self.verticalLayout_6 = QVBoxLayout(self.dockWidgetContents)
        self.verticalLayout_6.setObjectName(u"verticalLayout_6")
        self.verticalLayout_6.setContentsMargins(0, 0, 0, 0)
        self.train_settings = QFrame(self.dockWidgetContents)
        self.train_settings.setObjectName(u"train_settings")
        self.train_settings.setFrameShape(QFrame.StyledPanel)
        self.train_settings.setFrameShadow(QFrame.Sunken)
        self.train_settings.setLineWidth(2)
        self.verticalLayout_13 = QVBoxLayout(self.train_settings)
        self.verticalLayout_13.setObjectName(u"verticalLayout_13")
        self.verticalLayout_13.setContentsMargins(3, 3, 3, 3)
        self.scrollArea = QScrollArea(self.train_settings)
        self.scrollArea.setObjectName(u"scrollArea")
        self.scrollArea.setFrameShadow(QFrame.Raised)
        self.scrollArea.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scrollArea.setWidgetResizable(True)
        self.scrollAreaWidgetContents = QWidget()
        self.scrollAreaWidgetContents.setObjectName(u"scrollAreaWidgetContents")
        self.scrollAreaWidgetContents.setGeometry(QRect(0, 0, 337, 670))
        self.verticalLayout_11 = QVBoxLayout(self.scrollAreaWidgetContents)
        self.verticalLayout_11.setSpacing(2)
        self.verticalLayout_11.setObjectName(u"verticalLayout_11")
        self.verticalLayout_11.setContentsMargins(3, 3, 3, 3)
        self.frame_16 = QFrame(self.scrollAreaWidgetContents)
        self.frame_16.setObjectName(u"frame_16")
        sizePolicy1.setHeightForWidth(self.frame_16.sizePolicy().hasHeightForWidth())
        self.frame_16.setSizePolicy(sizePolicy1)
        self.frame_16.setFrameShape(QFrame.StyledPanel)
        self.frame_16.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_20 = QHBoxLayout(self.frame_16)
        self.horizontalLayout_20.setObjectName(u"horizontalLayout_20")
        self.horizontalLayout_20.setContentsMargins(0, 0, 0, 0)
        self.label_22 = QLabel(self.frame_16)
        self.label_22.setObjectName(u"label_22")
        sizePolicy.setHeightForWidth(self.label_22.sizePolicy().hasHeightForWidth())
        self.label_22.setSizePolicy(sizePolicy)
        font2 = QFont()
        font2.setPointSize(12)
        font2.setBold(False)
        font2.setUnderline(False)
        self.label_22.setFont(font2)
        self.label_22.setFrameShape(QFrame.NoFrame)
        self.label_22.setFrameShadow(QFrame.Raised)

        self.horizontalLayout_20.addWidget(self.label_22)

        self.line_3 = QFrame(self.frame_16)
        self.line_3.setObjectName(u"line_3")
        sizePolicy4 = QSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        sizePolicy4.setHorizontalStretch(0)
        sizePolicy4.setVerticalStretch(0)
        sizePolicy4.setHeightForWidth(self.line_3.sizePolicy().hasHeightForWidth())
        self.line_3.setSizePolicy(sizePolicy4)
        self.line_3.setFrameShadow(QFrame.Raised)
        self.line_3.setLineWidth(1)
        self.line_3.setFrameShape(QFrame.Shape.HLine)

        self.horizontalLayout_20.addWidget(self.line_3)

        self.hide_experiment_settings = QCheckBox(self.frame_16)
        self.hide_experiment_settings.setObjectName(u"hide_experiment_settings")
        sizePolicy5 = QSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
        sizePolicy5.setHorizontalStretch(0)
        sizePolicy5.setVerticalStretch(0)
        sizePolicy5.setHeightForWidth(self.hide_experiment_settings.sizePolicy().hasHeightForWidth())
        self.hide_experiment_settings.setSizePolicy(sizePolicy5)

        self.horizontalLayout_20.addWidget(self.hide_experiment_settings)


        self.verticalLayout_11.addWidget(self.frame_16)

        self.frame_2 = QFrame(self.scrollAreaWidgetContents)
        self.frame_2.setObjectName(u"frame_2")
        sizePolicy1.setHeightForWidth(self.frame_2.sizePolicy().hasHeightForWidth())
        self.frame_2.setSizePolicy(sizePolicy1)
        self.frame_2.setFrameShape(QFrame.StyledPanel)
        self.frame_2.setFrameShadow(QFrame.Raised)
        self.formLayout = QFormLayout(self.frame_2)
        self.formLayout.setObjectName(u"formLayout")
        self.formLayout.setVerticalSpacing(6)
        self.formLayout.setContentsMargins(0, 0, 0, 0)
        self.label_2 = QLabel(self.frame_2)
        self.label_2.setObjectName(u"label_2")

        self.formLayout.setWidget(2, QFormLayout.ItemRole.LabelRole, self.label_2)

        self.horizontalLayout_3 = QHBoxLayout()
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.output_directory = QLineEdit(self.frame_2)
        self.output_directory.setObjectName(u"output_directory")

        self.horizontalLayout_3.addWidget(self.output_directory)

        self.browse_output = QPushButton(self.frame_2)
        self.browse_output.setObjectName(u"browse_output")

        self.horizontalLayout_3.addWidget(self.browse_output)


        self.formLayout.setLayout(2, QFormLayout.ItemRole.FieldRole, self.horizontalLayout_3)

        self.label_3 = QLabel(self.frame_2)
        self.label_3.setObjectName(u"label_3")

        self.formLayout.setWidget(3, QFormLayout.ItemRole.LabelRole, self.label_3)

        self.horizontalLayout_4 = QHBoxLayout()
        self.horizontalLayout_4.setObjectName(u"horizontalLayout_4")
        self.experiment_name = QLineEdit(self.frame_2)
        self.experiment_name.setObjectName(u"experiment_name")

        self.horizontalLayout_4.addWidget(self.experiment_name)

        self.label_4 = QLabel(self.frame_2)
        self.label_4.setObjectName(u"label_4")

        self.horizontalLayout_4.addWidget(self.label_4)

        self.experiment_version = QSpinBox(self.frame_2)
        self.experiment_version.setObjectName(u"experiment_version")
        self.experiment_version.setMinimum(1)

        self.horizontalLayout_4.addWidget(self.experiment_version)


        self.formLayout.setLayout(3, QFormLayout.ItemRole.FieldRole, self.horizontalLayout_4)


        self.verticalLayout_11.addWidget(self.frame_2)

        self.init_from_last_train = QPushButton(self.scrollAreaWidgetContents)
        self.init_from_last_train.setObjectName(u"init_from_last_train")

        self.verticalLayout_11.addWidget(self.init_from_last_train)

        self.frame_17 = QFrame(self.scrollAreaWidgetContents)
        self.frame_17.setObjectName(u"frame_17")
        sizePolicy1.setHeightForWidth(self.frame_17.sizePolicy().hasHeightForWidth())
        self.frame_17.setSizePolicy(sizePolicy1)
        self.frame_17.setFrameShape(QFrame.StyledPanel)
        self.frame_17.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_21 = QHBoxLayout(self.frame_17)
        self.horizontalLayout_21.setObjectName(u"horizontalLayout_21")
        self.horizontalLayout_21.setContentsMargins(0, 0, 0, 0)
        self.label_23 = QLabel(self.frame_17)
        self.label_23.setObjectName(u"label_23")
        sizePolicy.setHeightForWidth(self.label_23.sizePolicy().hasHeightForWidth())
        self.label_23.setSizePolicy(sizePolicy)
        font3 = QFont()
        font3.setPointSize(12)
        self.label_23.setFont(font3)
        self.label_23.setFrameShape(QFrame.NoFrame)
        self.label_23.setFrameShadow(QFrame.Raised)

        self.horizontalLayout_21.addWidget(self.label_23)

        self.line_4 = QFrame(self.frame_17)
        self.line_4.setObjectName(u"line_4")
        self.line_4.setFrameShadow(QFrame.Raised)
        self.line_4.setLineWidth(1)
        self.line_4.setFrameShape(QFrame.Shape.HLine)

        self.horizontalLayout_21.addWidget(self.line_4)

        self.hide_dataloader_settings = QCheckBox(self.frame_17)
        self.hide_dataloader_settings.setObjectName(u"hide_dataloader_settings")
        sizePolicy5.setHeightForWidth(self.hide_dataloader_settings.sizePolicy().hasHeightForWidth())
        self.hide_dataloader_settings.setSizePolicy(sizePolicy5)

        self.horizontalLayout_21.addWidget(self.hide_dataloader_settings)


        self.verticalLayout_11.addWidget(self.frame_17)

        self.frame_3 = QFrame(self.scrollAreaWidgetContents)
        self.frame_3.setObjectName(u"frame_3")
        sizePolicy1.setHeightForWidth(self.frame_3.sizePolicy().hasHeightForWidth())
        self.frame_3.setSizePolicy(sizePolicy1)
        self.frame_3.setFrameShape(QFrame.StyledPanel)
        self.frame_3.setFrameShadow(QFrame.Raised)
        self.formLayout_3 = QFormLayout(self.frame_3)
        self.formLayout_3.setObjectName(u"formLayout_3")
        self.formLayout_3.setContentsMargins(0, 0, 0, 0)
        self.label = QLabel(self.frame_3)
        self.label.setObjectName(u"label")

        self.formLayout_3.setWidget(0, QFormLayout.ItemRole.LabelRole, self.label)

        self.horizontalLayout_2 = QHBoxLayout()
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.dataroot = QLineEdit(self.frame_3)
        self.dataroot.setObjectName(u"dataroot")

        self.horizontalLayout_2.addWidget(self.dataroot)

        self.browse_dataset = QPushButton(self.frame_3)
        self.browse_dataset.setObjectName(u"browse_dataset")

        self.horizontalLayout_2.addWidget(self.browse_dataset)


        self.formLayout_3.setLayout(0, QFormLayout.ItemRole.FieldRole, self.horizontalLayout_2)


        self.verticalLayout_11.addWidget(self.frame_3)

        self.frame_4 = QFrame(self.scrollAreaWidgetContents)
        self.frame_4.setObjectName(u"frame_4")
        sizePolicy1.setHeightForWidth(self.frame_4.sizePolicy().hasHeightForWidth())
        self.frame_4.setSizePolicy(sizePolicy1)
        self.frame_4.setFrameShape(QFrame.StyledPanel)
        self.frame_4.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_9 = QHBoxLayout(self.frame_4)
        self.horizontalLayout_9.setObjectName(u"horizontalLayout_9")
        self.horizontalLayout_9.setContentsMargins(0, 0, 0, 0)
        self.frame_5 = QFrame(self.frame_4)
        self.frame_5.setObjectName(u"frame_5")
        self.frame_5.setFrameShape(QFrame.StyledPanel)
        self.frame_5.setFrameShadow(QFrame.Raised)
        self.formLayout_4 = QFormLayout(self.frame_5)
        self.formLayout_4.setObjectName(u"formLayout_4")
        self.formLayout_4.setContentsMargins(0, 0, 0, 0)
        self.label_10 = QLabel(self.frame_5)
        self.label_10.setObjectName(u"label_10")

        self.formLayout_4.setWidget(0, QFormLayout.ItemRole.LabelRole, self.label_10)

        self.load_size = QSpinBox(self.frame_5)
        self.load_size.setObjectName(u"load_size")
        sizePolicy6 = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        sizePolicy6.setHorizontalStretch(0)
        sizePolicy6.setVerticalStretch(0)
        sizePolicy6.setHeightForWidth(self.load_size.sizePolicy().hasHeightForWidth())
        self.load_size.setSizePolicy(sizePolicy6)
        self.load_size.setMinimum(64)
        self.load_size.setMaximum(2048)
        self.load_size.setSingleStep(64)
        self.load_size.setValue(256)

        self.formLayout_4.setWidget(0, QFormLayout.ItemRole.FieldRole, self.load_size)


        self.horizontalLayout_9.addWidget(self.frame_5)

        self.frame_6 = QFrame(self.frame_4)
        self.frame_6.setObjectName(u"frame_6")
        self.frame_6.setFrameShape(QFrame.StyledPanel)
        self.frame_6.setFrameShadow(QFrame.Raised)
        self.formLayout_5 = QFormLayout(self.frame_6)
        self.formLayout_5.setObjectName(u"formLayout_5")
        self.formLayout_5.setContentsMargins(0, 0, 0, 0)
        self.label_11 = QLabel(self.frame_6)
        self.label_11.setObjectName(u"label_11")

        self.formLayout_5.setWidget(0, QFormLayout.ItemRole.LabelRole, self.label_11)

        self.crop_size = QSpinBox(self.frame_6)
        self.crop_size.setObjectName(u"crop_size")
        sizePolicy6.setHeightForWidth(self.crop_size.sizePolicy().hasHeightForWidth())
        self.crop_size.setSizePolicy(sizePolicy6)
        self.crop_size.setMinimum(64)
        self.crop_size.setMaximum(1024)
        self.crop_size.setSingleStep(64)
        self.crop_size.setValue(256)

        self.formLayout_5.setWidget(0, QFormLayout.ItemRole.FieldRole, self.crop_size)


        self.horizontalLayout_9.addWidget(self.frame_6)

        self.label_9 = QLabel(self.frame_4)
        self.label_9.setObjectName(u"label_9")
        sizePolicy3.setHeightForWidth(self.label_9.sizePolicy().hasHeightForWidth())
        self.label_9.setSizePolicy(sizePolicy3)

        self.horizontalLayout_9.addWidget(self.label_9)

        self.input_nc = QSpinBox(self.frame_4)
        self.input_nc.setObjectName(u"input_nc")
        sizePolicy7 = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        sizePolicy7.setHorizontalStretch(0)
        sizePolicy7.setVerticalStretch(0)
        sizePolicy7.setHeightForWidth(self.input_nc.sizePolicy().hasHeightForWidth())
        self.input_nc.setSizePolicy(sizePolicy7)
        self.input_nc.setMinimum(1)
        self.input_nc.setMaximum(3)
        self.input_nc.setValue(3)

        self.horizontalLayout_9.addWidget(self.input_nc)


        self.verticalLayout_11.addWidget(self.frame_4)

        self.frame_13 = QFrame(self.scrollAreaWidgetContents)
        self.frame_13.setObjectName(u"frame_13")
        sizePolicy1.setHeightForWidth(self.frame_13.sizePolicy().hasHeightForWidth())
        self.frame_13.setSizePolicy(sizePolicy1)
        self.frame_13.setFrameShape(QFrame.StyledPanel)
        self.frame_13.setFrameShadow(QFrame.Raised)
        self.formLayout_7 = QFormLayout(self.frame_13)
        self.formLayout_7.setObjectName(u"formLayout_7")
        self.formLayout_7.setContentsMargins(0, 0, 0, 0)
        self.label_5 = QLabel(self.frame_13)
        self.label_5.setObjectName(u"label_5")

        self.formLayout_7.setWidget(0, QFormLayout.ItemRole.LabelRole, self.label_5)

        self.direction = QComboBox(self.frame_13)
        self.direction.setObjectName(u"direction")
        self.direction.setEditable(False)
        self.direction.setFrame(True)

        self.formLayout_7.setWidget(0, QFormLayout.ItemRole.FieldRole, self.direction)

        self.label_8 = QLabel(self.frame_13)
        self.label_8.setObjectName(u"label_8")

        self.formLayout_7.setWidget(1, QFormLayout.ItemRole.LabelRole, self.label_8)

        self.horizontalLayout_7 = QHBoxLayout()
        self.horizontalLayout_7.setObjectName(u"horizontalLayout_7")
        self.batch_size_slider = QSlider(self.frame_13)
        self.batch_size_slider.setObjectName(u"batch_size_slider")
        self.batch_size_slider.setMinimum(1)
        self.batch_size_slider.setMaximum(32)
        self.batch_size_slider.setValue(1)
        self.batch_size_slider.setOrientation(Qt.Horizontal)

        self.horizontalLayout_7.addWidget(self.batch_size_slider)

        self.batch_size = QSpinBox(self.frame_13)
        self.batch_size.setObjectName(u"batch_size")
        self.batch_size.setMinimum(1)
        self.batch_size.setMaximum(32)

        self.horizontalLayout_7.addWidget(self.batch_size)


        self.formLayout_7.setLayout(1, QFormLayout.ItemRole.FieldRole, self.horizontalLayout_7)


        self.verticalLayout_11.addWidget(self.frame_13)

        self.frame_23 = QFrame(self.scrollAreaWidgetContents)
        self.frame_23.setObjectName(u"frame_23")
        self.frame_23.setFrameShape(QFrame.StyledPanel)
        self.frame_23.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_30 = QHBoxLayout(self.frame_23)
        self.horizontalLayout_30.setObjectName(u"horizontalLayout_30")
        self.horizontalLayout_30.setContentsMargins(0, 0, 0, 0)
        self.label_40 = QLabel(self.frame_23)
        self.label_40.setObjectName(u"label_40")
        sizePolicy.setHeightForWidth(self.label_40.sizePolicy().hasHeightForWidth())
        self.label_40.setSizePolicy(sizePolicy)
        self.label_40.setFont(font3)
        self.label_40.setFrameShape(QFrame.NoFrame)
        self.label_40.setFrameShadow(QFrame.Raised)

        self.horizontalLayout_30.addWidget(self.label_40)

        self.line_6 = QFrame(self.frame_23)
        self.line_6.setObjectName(u"line_6")
        self.line_6.setFrameShadow(QFrame.Raised)
        self.line_6.setFrameShape(QFrame.Shape.HLine)

        self.horizontalLayout_30.addWidget(self.line_6)


        self.verticalLayout_11.addWidget(self.frame_23)

        self.tabWidget_2 = QTabWidget(self.scrollAreaWidgetContents)
        self.tabWidget_2.setObjectName(u"tabWidget_2")
        sizePolicy8 = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        sizePolicy8.setHorizontalStretch(0)
        sizePolicy8.setVerticalStretch(0)
        sizePolicy8.setHeightForWidth(self.tabWidget_2.sizePolicy().hasHeightForWidth())
        self.tabWidget_2.setSizePolicy(sizePolicy8)
        self.tabWidget_2.setTabPosition(QTabWidget.North)
        self.tab = QWidget()
        self.tab.setObjectName(u"tab")
        sizePolicy3.setHeightForWidth(self.tab.sizePolicy().hasHeightForWidth())
        self.tab.setSizePolicy(sizePolicy3)
        self.verticalLayout_2 = QVBoxLayout(self.tab)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.scrollArea_2 = QScrollArea(self.tab)
        self.scrollArea_2.setObjectName(u"scrollArea_2")
        self.scrollArea_2.setWidgetResizable(True)
        self.scrollAreaWidgetContents_2 = QWidget()
        self.scrollAreaWidgetContents_2.setObjectName(u"scrollAreaWidgetContents_2")
        self.scrollAreaWidgetContents_2.setGeometry(QRect(0, -167, 306, 618))
        self.verticalLayout_15 = QVBoxLayout(self.scrollAreaWidgetContents_2)
        self.verticalLayout_15.setObjectName(u"verticalLayout_15")
        self.continue_train = QGroupBox(self.scrollAreaWidgetContents_2)
        self.continue_train.setObjectName(u"continue_train")
        sizePolicy1.setHeightForWidth(self.continue_train.sizePolicy().hasHeightForWidth())
        self.continue_train.setSizePolicy(sizePolicy1)
        self.continue_train.setCheckable(True)
        self.continue_train.setChecked(False)
        self.horizontalLayout_16 = QHBoxLayout(self.continue_train)
        self.horizontalLayout_16.setObjectName(u"horizontalLayout_16")
        self.horizontalLayout_16.setContentsMargins(5, 5, 5, 5)
        self.label_16 = QLabel(self.continue_train)
        self.label_16.setObjectName(u"label_16")
        sizePolicy.setHeightForWidth(self.label_16.sizePolicy().hasHeightForWidth())
        self.label_16.setSizePolicy(sizePolicy)

        self.horizontalLayout_16.addWidget(self.label_16)

        self.load_epoch = QSpinBox(self.continue_train)
        self.load_epoch.setObjectName(u"load_epoch")
        self.load_epoch.setMinimum(1)

        self.horizontalLayout_16.addWidget(self.load_epoch)

        self.horizontalSpacer = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_16.addItem(self.horizontalSpacer)


        self.verticalLayout_15.addWidget(self.continue_train)

        self.separate_lr_schedules = QCheckBox(self.scrollAreaWidgetContents_2)
        self.separate_lr_schedules.setObjectName(u"separate_lr_schedules")

        self.verticalLayout_15.addWidget(self.separate_lr_schedules)

        self.gen_schedule_box = QGroupBox(self.scrollAreaWidgetContents_2)
        self.gen_schedule_box.setObjectName(u"gen_schedule_box")
        self.verticalLayout_19 = QVBoxLayout(self.gen_schedule_box)
        self.verticalLayout_19.setObjectName(u"verticalLayout_19")
        self.verticalLayout_19.setContentsMargins(2, 2, 2, 2)
        self.groupBox_4 = QGroupBox(self.gen_schedule_box)
        self.groupBox_4.setObjectName(u"groupBox_4")
        self.verticalLayout_12 = QVBoxLayout(self.groupBox_4)
        self.verticalLayout_12.setObjectName(u"verticalLayout_12")
        self.verticalLayout_12.setContentsMargins(5, 5, 5, 5)
        self.frame_19 = QFrame(self.groupBox_4)
        self.frame_19.setObjectName(u"frame_19")
        sizePolicy1.setHeightForWidth(self.frame_19.sizePolicy().hasHeightForWidth())
        self.frame_19.setSizePolicy(sizePolicy1)
        self.frame_19.setFrameShape(QFrame.StyledPanel)
        self.frame_19.setFrameShadow(QFrame.Raised)
        self.formLayout_8 = QFormLayout(self.frame_19)
        self.formLayout_8.setObjectName(u"formLayout_8")
        self.formLayout_8.setContentsMargins(0, 0, 0, 0)
        self.label_32 = QLabel(self.frame_19)
        self.label_32.setObjectName(u"label_32")

        self.formLayout_8.setWidget(0, QFormLayout.ItemRole.LabelRole, self.label_32)

        self.gen_epochs = QSpinBox(self.frame_19)
        self.gen_epochs.setObjectName(u"gen_epochs")
        self.gen_epochs.setMinimum(1)
        self.gen_epochs.setMaximum(999999)
        self.gen_epochs.setValue(100)

        self.formLayout_8.setWidget(0, QFormLayout.ItemRole.FieldRole, self.gen_epochs)

        self.label_31 = QLabel(self.frame_19)
        self.label_31.setObjectName(u"label_31")

        self.formLayout_8.setWidget(1, QFormLayout.ItemRole.LabelRole, self.label_31)

        self.gen_epochs_decay = QSpinBox(self.frame_19)
        self.gen_epochs_decay.setObjectName(u"gen_epochs_decay")
        self.gen_epochs_decay.setMinimum(1)
        self.gen_epochs_decay.setMaximum(999999)
        self.gen_epochs_decay.setValue(100)

        self.formLayout_8.setWidget(1, QFormLayout.ItemRole.FieldRole, self.gen_epochs_decay)


        self.verticalLayout_12.addWidget(self.frame_19)


        self.verticalLayout_19.addWidget(self.groupBox_4)

        self.groupBox_2 = QGroupBox(self.gen_schedule_box)
        self.groupBox_2.setObjectName(u"groupBox_2")
        self.formLayout_10 = QFormLayout(self.groupBox_2)
        self.formLayout_10.setObjectName(u"formLayout_10")
        self.formLayout_10.setContentsMargins(5, 5, 5, 5)
        self.label_6 = QLabel(self.groupBox_2)
        self.label_6.setObjectName(u"label_6")

        self.formLayout_10.setWidget(0, QFormLayout.ItemRole.LabelRole, self.label_6)

        self.horizontalLayout_22 = QHBoxLayout()
        self.horizontalLayout_22.setObjectName(u"horizontalLayout_22")
        self.gen_lr_initial_slider = QSlider(self.groupBox_2)
        self.gen_lr_initial_slider.setObjectName(u"gen_lr_initial_slider")
        self.gen_lr_initial_slider.setMaximum(100)
        self.gen_lr_initial_slider.setValue(2)
        self.gen_lr_initial_slider.setOrientation(Qt.Horizontal)

        self.horizontalLayout_22.addWidget(self.gen_lr_initial_slider)

        self.gen_lr_initial = QDoubleSpinBox(self.groupBox_2)
        self.gen_lr_initial.setObjectName(u"gen_lr_initial")
        self.gen_lr_initial.setDecimals(5)
        self.gen_lr_initial.setMinimum(0.000000000000000)
        self.gen_lr_initial.setMaximum(0.100000000000000)
        self.gen_lr_initial.setSingleStep(0.000100000000000)
        self.gen_lr_initial.setValue(0.000200000000000)

        self.horizontalLayout_22.addWidget(self.gen_lr_initial)


        self.formLayout_10.setLayout(0, QFormLayout.ItemRole.FieldRole, self.horizontalLayout_22)

        self.label_33 = QLabel(self.groupBox_2)
        self.label_33.setObjectName(u"label_33")

        self.formLayout_10.setWidget(1, QFormLayout.ItemRole.LabelRole, self.label_33)

        self.horizontalLayout_23 = QHBoxLayout()
        self.horizontalLayout_23.setObjectName(u"horizontalLayout_23")
        self.gen_lr_target_slider = QSlider(self.groupBox_2)
        self.gen_lr_target_slider.setObjectName(u"gen_lr_target_slider")
        self.gen_lr_target_slider.setMaximum(100)
        self.gen_lr_target_slider.setValue(2)
        self.gen_lr_target_slider.setOrientation(Qt.Horizontal)

        self.horizontalLayout_23.addWidget(self.gen_lr_target_slider)

        self.gen_lr_target = QDoubleSpinBox(self.groupBox_2)
        self.gen_lr_target.setObjectName(u"gen_lr_target")
        self.gen_lr_target.setDecimals(5)
        self.gen_lr_target.setMinimum(0.000000000000000)
        self.gen_lr_target.setMaximum(0.100000000000000)
        self.gen_lr_target.setSingleStep(0.000100000000000)
        self.gen_lr_target.setValue(0.000200000000000)

        self.horizontalLayout_23.addWidget(self.gen_lr_target)


        self.formLayout_10.setLayout(1, QFormLayout.ItemRole.FieldRole, self.horizontalLayout_23)


        self.verticalLayout_19.addWidget(self.groupBox_2)

        self.groupBox_3 = QGroupBox(self.gen_schedule_box)
        self.groupBox_3.setObjectName(u"groupBox_3")
        self.verticalLayout_14 = QVBoxLayout(self.groupBox_3)
        self.verticalLayout_14.setObjectName(u"verticalLayout_14")
        self.verticalLayout_14.setContentsMargins(5, 5, 5, 5)
        self.formLayout_12 = QFormLayout()
        self.formLayout_12.setObjectName(u"formLayout_12")
        self.label_14 = QLabel(self.groupBox_3)
        self.label_14.setObjectName(u"label_14")

        self.formLayout_12.setWidget(0, QFormLayout.ItemRole.LabelRole, self.label_14)

        self.horizontalLayout_14 = QHBoxLayout()
        self.horizontalLayout_14.setObjectName(u"horizontalLayout_14")
        self.gen_optim_beta1_slider = QSlider(self.groupBox_3)
        self.gen_optim_beta1_slider.setObjectName(u"gen_optim_beta1_slider")
        self.gen_optim_beta1_slider.setMaximum(250)
        self.gen_optim_beta1_slider.setOrientation(Qt.Horizontal)

        self.horizontalLayout_14.addWidget(self.gen_optim_beta1_slider)

        self.gen_optim_beta1 = QDoubleSpinBox(self.groupBox_3)
        self.gen_optim_beta1.setObjectName(u"gen_optim_beta1")
        self.gen_optim_beta1.setMaximum(2.500000000000000)
        self.gen_optim_beta1.setSingleStep(0.100000000000000)
        self.gen_optim_beta1.setValue(0.500000000000000)

        self.horizontalLayout_14.addWidget(self.gen_optim_beta1)


        self.formLayout_12.setLayout(0, QFormLayout.ItemRole.FieldRole, self.horizontalLayout_14)


        self.verticalLayout_14.addLayout(self.formLayout_12)


        self.verticalLayout_19.addWidget(self.groupBox_3)


        self.verticalLayout_15.addWidget(self.gen_schedule_box)

        self.disc_schedule_box = QGroupBox(self.scrollAreaWidgetContents_2)
        self.disc_schedule_box.setObjectName(u"disc_schedule_box")
        self.verticalLayout_20 = QVBoxLayout(self.disc_schedule_box)
        self.verticalLayout_20.setObjectName(u"verticalLayout_20")
        self.verticalLayout_20.setContentsMargins(2, 2, 2, 2)
        self.groupBox_5 = QGroupBox(self.disc_schedule_box)
        self.groupBox_5.setObjectName(u"groupBox_5")
        self.verticalLayout_21 = QVBoxLayout(self.groupBox_5)
        self.verticalLayout_21.setObjectName(u"verticalLayout_21")
        self.verticalLayout_21.setContentsMargins(5, 5, 5, 5)
        self.frame_21 = QFrame(self.groupBox_5)
        self.frame_21.setObjectName(u"frame_21")
        sizePolicy1.setHeightForWidth(self.frame_21.sizePolicy().hasHeightForWidth())
        self.frame_21.setSizePolicy(sizePolicy1)
        self.frame_21.setFrameShape(QFrame.StyledPanel)
        self.frame_21.setFrameShadow(QFrame.Raised)
        self.formLayout_11 = QFormLayout(self.frame_21)
        self.formLayout_11.setObjectName(u"formLayout_11")
        self.formLayout_11.setContentsMargins(0, 0, 0, 0)
        self.label_34 = QLabel(self.frame_21)
        self.label_34.setObjectName(u"label_34")

        self.formLayout_11.setWidget(0, QFormLayout.ItemRole.LabelRole, self.label_34)

        self.disc_epochs = QSpinBox(self.frame_21)
        self.disc_epochs.setObjectName(u"disc_epochs")
        self.disc_epochs.setMinimum(1)
        self.disc_epochs.setMaximum(999999)
        self.disc_epochs.setValue(100)

        self.formLayout_11.setWidget(0, QFormLayout.ItemRole.FieldRole, self.disc_epochs)

        self.label_35 = QLabel(self.frame_21)
        self.label_35.setObjectName(u"label_35")

        self.formLayout_11.setWidget(1, QFormLayout.ItemRole.LabelRole, self.label_35)

        self.disc_epochs_decay = QSpinBox(self.frame_21)
        self.disc_epochs_decay.setObjectName(u"disc_epochs_decay")
        self.disc_epochs_decay.setMinimum(1)
        self.disc_epochs_decay.setMaximum(999999)
        self.disc_epochs_decay.setValue(100)

        self.formLayout_11.setWidget(1, QFormLayout.ItemRole.FieldRole, self.disc_epochs_decay)


        self.verticalLayout_21.addWidget(self.frame_21)


        self.verticalLayout_20.addWidget(self.groupBox_5)

        self.groupBox_8 = QGroupBox(self.disc_schedule_box)
        self.groupBox_8.setObjectName(u"groupBox_8")
        self.formLayout_13 = QFormLayout(self.groupBox_8)
        self.formLayout_13.setObjectName(u"formLayout_13")
        self.formLayout_13.setContentsMargins(5, 5, 5, 5)
        self.label_7 = QLabel(self.groupBox_8)
        self.label_7.setObjectName(u"label_7")

        self.formLayout_13.setWidget(0, QFormLayout.ItemRole.LabelRole, self.label_7)

        self.horizontalLayout_24 = QHBoxLayout()
        self.horizontalLayout_24.setObjectName(u"horizontalLayout_24")
        self.disc_lr_initial_slider = QSlider(self.groupBox_8)
        self.disc_lr_initial_slider.setObjectName(u"disc_lr_initial_slider")
        self.disc_lr_initial_slider.setMaximum(100)
        self.disc_lr_initial_slider.setValue(2)
        self.disc_lr_initial_slider.setOrientation(Qt.Horizontal)

        self.horizontalLayout_24.addWidget(self.disc_lr_initial_slider)

        self.disc_lr_initial = QDoubleSpinBox(self.groupBox_8)
        self.disc_lr_initial.setObjectName(u"disc_lr_initial")
        self.disc_lr_initial.setDecimals(5)
        self.disc_lr_initial.setMinimum(0.000000000000000)
        self.disc_lr_initial.setMaximum(0.100000000000000)
        self.disc_lr_initial.setSingleStep(0.000100000000000)
        self.disc_lr_initial.setValue(0.000200000000000)

        self.horizontalLayout_24.addWidget(self.disc_lr_initial)


        self.formLayout_13.setLayout(0, QFormLayout.ItemRole.FieldRole, self.horizontalLayout_24)

        self.label_36 = QLabel(self.groupBox_8)
        self.label_36.setObjectName(u"label_36")

        self.formLayout_13.setWidget(1, QFormLayout.ItemRole.LabelRole, self.label_36)

        self.horizontalLayout_25 = QHBoxLayout()
        self.horizontalLayout_25.setObjectName(u"horizontalLayout_25")
        self.disc_lr_target_slider = QSlider(self.groupBox_8)
        self.disc_lr_target_slider.setObjectName(u"disc_lr_target_slider")
        self.disc_lr_target_slider.setMaximum(100)
        self.disc_lr_target_slider.setValue(2)
        self.disc_lr_target_slider.setOrientation(Qt.Horizontal)

        self.horizontalLayout_25.addWidget(self.disc_lr_target_slider)

        self.disc_lr_target = QDoubleSpinBox(self.groupBox_8)
        self.disc_lr_target.setObjectName(u"disc_lr_target")
        self.disc_lr_target.setDecimals(5)
        self.disc_lr_target.setMinimum(0.000000000000000)
        self.disc_lr_target.setMaximum(0.100000000000000)
        self.disc_lr_target.setSingleStep(0.000100000000000)
        self.disc_lr_target.setValue(0.000200000000000)

        self.horizontalLayout_25.addWidget(self.disc_lr_target)


        self.formLayout_13.setLayout(1, QFormLayout.ItemRole.FieldRole, self.horizontalLayout_25)


        self.verticalLayout_20.addWidget(self.groupBox_8)

        self.groupBox_9 = QGroupBox(self.disc_schedule_box)
        self.groupBox_9.setObjectName(u"groupBox_9")
        self.verticalLayout_22 = QVBoxLayout(self.groupBox_9)
        self.verticalLayout_22.setObjectName(u"verticalLayout_22")
        self.verticalLayout_22.setContentsMargins(5, 5, 5, 5)
        self.formLayout_14 = QFormLayout()
        self.formLayout_14.setObjectName(u"formLayout_14")
        self.label_15 = QLabel(self.groupBox_9)
        self.label_15.setObjectName(u"label_15")

        self.formLayout_14.setWidget(0, QFormLayout.ItemRole.LabelRole, self.label_15)

        self.horizontalLayout_26 = QHBoxLayout()
        self.horizontalLayout_26.setObjectName(u"horizontalLayout_26")
        self.disc_optim_beta1_slider = QSlider(self.groupBox_9)
        self.disc_optim_beta1_slider.setObjectName(u"disc_optim_beta1_slider")
        self.disc_optim_beta1_slider.setMaximum(250)
        self.disc_optim_beta1_slider.setOrientation(Qt.Horizontal)

        self.horizontalLayout_26.addWidget(self.disc_optim_beta1_slider)

        self.disc_optim_beta1 = QDoubleSpinBox(self.groupBox_9)
        self.disc_optim_beta1.setObjectName(u"disc_optim_beta1")
        self.disc_optim_beta1.setMaximum(2.500000000000000)
        self.disc_optim_beta1.setSingleStep(0.100000000000000)
        self.disc_optim_beta1.setValue(0.500000000000000)

        self.horizontalLayout_26.addWidget(self.disc_optim_beta1)


        self.formLayout_14.setLayout(0, QFormLayout.ItemRole.FieldRole, self.horizontalLayout_26)


        self.verticalLayout_22.addLayout(self.formLayout_14)


        self.verticalLayout_20.addWidget(self.groupBox_9)


        self.verticalLayout_15.addWidget(self.disc_schedule_box)

        self.verticalSpacer_2 = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.verticalLayout_15.addItem(self.verticalSpacer_2)

        self.scrollArea_2.setWidget(self.scrollAreaWidgetContents_2)

        self.verticalLayout_2.addWidget(self.scrollArea_2)

        self.tabWidget_2.addTab(self.tab, "")
        self.tab_3 = QWidget()
        self.tab_3.setObjectName(u"tab_3")
        self.verticalLayout_3 = QVBoxLayout(self.tab_3)
        self.verticalLayout_3.setObjectName(u"verticalLayout_3")
        self.verticalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.scrollArea_3 = QScrollArea(self.tab_3)
        self.scrollArea_3.setObjectName(u"scrollArea_3")
        self.scrollArea_3.setWidgetResizable(True)
        self.scrollAreaWidgetContents_3 = QWidget()
        self.scrollAreaWidgetContents_3.setObjectName(u"scrollAreaWidgetContents_3")
        self.scrollAreaWidgetContents_3.setGeometry(QRect(0, 0, 335, 296))
        self.verticalLayout_16 = QVBoxLayout(self.scrollAreaWidgetContents_3)
        self.verticalLayout_16.setObjectName(u"verticalLayout_16")
        self.frame_18 = QFrame(self.scrollAreaWidgetContents_3)
        self.frame_18.setObjectName(u"frame_18")
        self.frame_18.setFrameShape(QFrame.StyledPanel)
        self.frame_18.setFrameShadow(QFrame.Raised)
        self.formLayout_2 = QFormLayout(self.frame_18)
        self.formLayout_2.setObjectName(u"formLayout_2")
        self.formLayout_2.setContentsMargins(0, 0, 0, 0)
        self.label_12 = QLabel(self.frame_18)
        self.label_12.setObjectName(u"label_12")

        self.formLayout_2.setWidget(0, QFormLayout.ItemRole.LabelRole, self.label_12)

        self.horizontalLayout_10 = QHBoxLayout()
        self.horizontalLayout_10.setObjectName(u"horizontalLayout_10")
        self.lambda_l1_slider = QSlider(self.frame_18)
        self.lambda_l1_slider.setObjectName(u"lambda_l1_slider")
        self.lambda_l1_slider.setMaximum(500)
        self.lambda_l1_slider.setValue(100)
        self.lambda_l1_slider.setOrientation(Qt.Horizontal)

        self.horizontalLayout_10.addWidget(self.lambda_l1_slider)

        self.lambda_l1 = QDoubleSpinBox(self.frame_18)
        self.lambda_l1.setObjectName(u"lambda_l1")
        self.lambda_l1.setMaximum(500.000000000000000)
        self.lambda_l1.setSingleStep(5.000000000000000)
        self.lambda_l1.setValue(100.000000000000000)

        self.horizontalLayout_10.addWidget(self.lambda_l1)


        self.formLayout_2.setLayout(0, QFormLayout.ItemRole.FieldRole, self.horizontalLayout_10)

        self.label_13 = QLabel(self.frame_18)
        self.label_13.setObjectName(u"label_13")

        self.formLayout_2.setWidget(1, QFormLayout.ItemRole.LabelRole, self.label_13)

        self.horizontalLayout_11 = QHBoxLayout()
        self.horizontalLayout_11.setObjectName(u"horizontalLayout_11")
        self.lambda_sobel_slider = QSlider(self.frame_18)
        self.lambda_sobel_slider.setObjectName(u"lambda_sobel_slider")
        self.lambda_sobel_slider.setMaximum(100)
        self.lambda_sobel_slider.setOrientation(Qt.Horizontal)

        self.horizontalLayout_11.addWidget(self.lambda_sobel_slider)

        self.lambda_sobel = QDoubleSpinBox(self.frame_18)
        self.lambda_sobel.setObjectName(u"lambda_sobel")
        self.lambda_sobel.setMaximum(500.000000000000000)
        self.lambda_sobel.setValue(0.000000000000000)

        self.horizontalLayout_11.addWidget(self.lambda_sobel)


        self.formLayout_2.setLayout(1, QFormLayout.ItemRole.FieldRole, self.horizontalLayout_11)

        self.label_37 = QLabel(self.frame_18)
        self.label_37.setObjectName(u"label_37")

        self.formLayout_2.setWidget(2, QFormLayout.ItemRole.LabelRole, self.label_37)

        self.label_38 = QLabel(self.frame_18)
        self.label_38.setObjectName(u"label_38")

        self.formLayout_2.setWidget(3, QFormLayout.ItemRole.LabelRole, self.label_38)

        self.horizontalLayout_28 = QHBoxLayout()
        self.horizontalLayout_28.setObjectName(u"horizontalLayout_28")
        self.lambda_laplacian_slider = QSlider(self.frame_18)
        self.lambda_laplacian_slider.setObjectName(u"lambda_laplacian_slider")
        self.lambda_laplacian_slider.setMaximum(100)
        self.lambda_laplacian_slider.setOrientation(Qt.Horizontal)

        self.horizontalLayout_28.addWidget(self.lambda_laplacian_slider)

        self.lambda_laplacian = QDoubleSpinBox(self.frame_18)
        self.lambda_laplacian.setObjectName(u"lambda_laplacian")
        self.lambda_laplacian.setMaximum(500.000000000000000)
        self.lambda_laplacian.setValue(0.000000000000000)

        self.horizontalLayout_28.addWidget(self.lambda_laplacian)


        self.formLayout_2.setLayout(2, QFormLayout.ItemRole.FieldRole, self.horizontalLayout_28)

        self.horizontalLayout_27 = QHBoxLayout()
        self.horizontalLayout_27.setObjectName(u"horizontalLayout_27")
        self.lambda_vgg_slider = QSlider(self.frame_18)
        self.lambda_vgg_slider.setObjectName(u"lambda_vgg_slider")
        self.lambda_vgg_slider.setMaximum(100)
        self.lambda_vgg_slider.setValue(10)
        self.lambda_vgg_slider.setOrientation(Qt.Horizontal)

        self.horizontalLayout_27.addWidget(self.lambda_vgg_slider)

        self.lambda_vgg = QDoubleSpinBox(self.frame_18)
        self.lambda_vgg.setObjectName(u"lambda_vgg")
        self.lambda_vgg.setMaximum(500.000000000000000)
        self.lambda_vgg.setValue(10.000000000000000)

        self.horizontalLayout_27.addWidget(self.lambda_vgg)


        self.formLayout_2.setLayout(3, QFormLayout.ItemRole.FieldRole, self.horizontalLayout_27)


        self.verticalLayout_16.addWidget(self.frame_18)

        self.verticalSpacer_3 = QSpacerItem(20, 153, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.verticalLayout_16.addItem(self.verticalSpacer_3)

        self.scrollArea_3.setWidget(self.scrollAreaWidgetContents_3)

        self.verticalLayout_3.addWidget(self.scrollArea_3)

        self.tabWidget_2.addTab(self.tab_3, "")
        self.tab_4 = QWidget()
        self.tab_4.setObjectName(u"tab_4")
        self.verticalLayout_4 = QVBoxLayout(self.tab_4)
        self.verticalLayout_4.setObjectName(u"verticalLayout_4")
        self.verticalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.scrollArea_4 = QScrollArea(self.tab_4)
        self.scrollArea_4.setObjectName(u"scrollArea_4")
        self.scrollArea_4.setWidgetResizable(True)
        self.scrollAreaWidgetContents_4 = QWidget()
        self.scrollAreaWidgetContents_4.setObjectName(u"scrollAreaWidgetContents_4")
        self.scrollAreaWidgetContents_4.setGeometry(QRect(0, 0, 335, 296))
        self.verticalLayout_17 = QVBoxLayout(self.scrollAreaWidgetContents_4)
        self.verticalLayout_17.setObjectName(u"verticalLayout_17")
        self.groupBox_6 = QGroupBox(self.scrollAreaWidgetContents_4)
        self.groupBox_6.setObjectName(u"groupBox_6")
        self.verticalLayout_24 = QVBoxLayout(self.groupBox_6)
        self.verticalLayout_24.setObjectName(u"verticalLayout_24")
        self.verticalLayout_24.setContentsMargins(9, 9, 9, 9)
        self.frame_25 = QFrame(self.groupBox_6)
        self.frame_25.setObjectName(u"frame_25")
        self.frame_25.setFrameShape(QFrame.StyledPanel)
        self.frame_25.setFrameShadow(QFrame.Raised)
        self.formLayout_16 = QFormLayout(self.frame_25)
        self.formLayout_16.setObjectName(u"formLayout_16")
        self.formLayout_16.setContentsMargins(0, 0, 0, 0)
        self.label_17 = QLabel(self.frame_25)
        self.label_17.setObjectName(u"label_17")
        sizePolicy.setHeightForWidth(self.label_17.sizePolicy().hasHeightForWidth())
        self.label_17.setSizePolicy(sizePolicy)

        self.formLayout_16.setWidget(0, QFormLayout.ItemRole.LabelRole, self.label_17)

        self.save_model = QCheckBox(self.frame_25)
        self.save_model.setObjectName(u"save_model")
        sizePolicy5.setHeightForWidth(self.save_model.sizePolicy().hasHeightForWidth())
        self.save_model.setSizePolicy(sizePolicy5)
        self.save_model.setChecked(True)

        self.formLayout_16.setWidget(0, QFormLayout.ItemRole.FieldRole, self.save_model)

        self.label_18 = QLabel(self.frame_25)
        self.label_18.setObjectName(u"label_18")
        sizePolicy.setHeightForWidth(self.label_18.sizePolicy().hasHeightForWidth())
        self.label_18.setSizePolicy(sizePolicy)

        self.formLayout_16.setWidget(1, QFormLayout.ItemRole.LabelRole, self.label_18)

        self.horizontalLayout_17 = QHBoxLayout()
        self.horizontalLayout_17.setObjectName(u"horizontalLayout_17")
        self.model_save_rate = QSpinBox(self.frame_25)
        self.model_save_rate.setObjectName(u"model_save_rate")
        sizePolicy5.setHeightForWidth(self.model_save_rate.sizePolicy().hasHeightForWidth())
        self.model_save_rate.setSizePolicy(sizePolicy5)
        self.model_save_rate.setMinimum(1)
        self.model_save_rate.setValue(5)

        self.horizontalLayout_17.addWidget(self.model_save_rate)

        self.label_42 = QLabel(self.frame_25)
        self.label_42.setObjectName(u"label_42")

        self.horizontalLayout_17.addWidget(self.label_42)

        self.horizontalSpacer_2 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_17.addItem(self.horizontalSpacer_2)


        self.formLayout_16.setLayout(1, QFormLayout.ItemRole.FieldRole, self.horizontalLayout_17)


        self.verticalLayout_24.addWidget(self.frame_25)


        self.verticalLayout_17.addWidget(self.groupBox_6)

        self.groupBox = QGroupBox(self.scrollAreaWidgetContents_4)
        self.groupBox.setObjectName(u"groupBox")
        self.formLayout_6 = QFormLayout(self.groupBox)
        self.formLayout_6.setObjectName(u"formLayout_6")
        self.label_24 = QLabel(self.groupBox)
        self.label_24.setObjectName(u"label_24")

        self.formLayout_6.setWidget(0, QFormLayout.ItemRole.LabelRole, self.label_24)

        self.save_examples = QCheckBox(self.groupBox)
        self.save_examples.setObjectName(u"save_examples")
        self.save_examples.setChecked(True)

        self.formLayout_6.setWidget(0, QFormLayout.ItemRole.FieldRole, self.save_examples)

        self.label_25 = QLabel(self.groupBox)
        self.label_25.setObjectName(u"label_25")

        self.formLayout_6.setWidget(1, QFormLayout.ItemRole.LabelRole, self.label_25)

        self.horizontalLayout_18 = QHBoxLayout()
        self.horizontalLayout_18.setObjectName(u"horizontalLayout_18")
        self.example_save_rate = QSpinBox(self.groupBox)
        self.example_save_rate.setObjectName(u"example_save_rate")

        self.horizontalLayout_18.addWidget(self.example_save_rate)

        self.label_43 = QLabel(self.groupBox)
        self.label_43.setObjectName(u"label_43")

        self.horizontalLayout_18.addWidget(self.label_43)

        self.horizontalSpacer_3 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_18.addItem(self.horizontalSpacer_3)


        self.formLayout_6.setLayout(1, QFormLayout.ItemRole.FieldRole, self.horizontalLayout_18)

        self.label_26 = QLabel(self.groupBox)
        self.label_26.setObjectName(u"label_26")

        self.formLayout_6.setWidget(2, QFormLayout.ItemRole.LabelRole, self.label_26)

        self.horizontalLayout_19 = QHBoxLayout()
        self.horizontalLayout_19.setObjectName(u"horizontalLayout_19")
        self.num_examples = QSpinBox(self.groupBox)
        self.num_examples.setObjectName(u"num_examples")

        self.horizontalLayout_19.addWidget(self.num_examples)

        self.horizontalSpacer_4 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_19.addItem(self.horizontalSpacer_4)


        self.formLayout_6.setLayout(2, QFormLayout.ItemRole.FieldRole, self.horizontalLayout_19)


        self.verticalLayout_17.addWidget(self.groupBox)

        self.verticalSpacer_4 = QSpacerItem(20, 67, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.verticalLayout_17.addItem(self.verticalSpacer_4)

        self.scrollArea_4.setWidget(self.scrollAreaWidgetContents_4)

        self.verticalLayout_4.addWidget(self.scrollArea_4)

        self.tabWidget_2.addTab(self.tab_4, "")
        self.tab_2 = QWidget()
        self.tab_2.setObjectName(u"tab_2")
        self.verticalLayout_10 = QVBoxLayout(self.tab_2)
        self.verticalLayout_10.setObjectName(u"verticalLayout_10")
        self.verticalLayout_10.setContentsMargins(3, 0, 0, 0)
        self.scrollArea_5 = QScrollArea(self.tab_2)
        self.scrollArea_5.setObjectName(u"scrollArea_5")
        self.scrollArea_5.setWidgetResizable(True)
        self.scrollAreaWidgetContents_5 = QWidget()
        self.scrollAreaWidgetContents_5.setObjectName(u"scrollAreaWidgetContents_5")
        self.scrollAreaWidgetContents_5.setGeometry(QRect(0, 0, 332, 296))
        self.verticalLayout_18 = QVBoxLayout(self.scrollAreaWidgetContents_5)
        self.verticalLayout_18.setObjectName(u"verticalLayout_18")
        self.visdom_enable = QGroupBox(self.scrollAreaWidgetContents_5)
        self.visdom_enable.setObjectName(u"visdom_enable")
        sizePolicy1.setHeightForWidth(self.visdom_enable.sizePolicy().hasHeightForWidth())
        self.visdom_enable.setSizePolicy(sizePolicy1)
        self.visdom_enable.setCheckable(True)
        self.visdom_enable.setChecked(False)
        self.formLayout_9 = QFormLayout(self.visdom_enable)
        self.formLayout_9.setObjectName(u"formLayout_9")
        self.formLayout_9.setContentsMargins(2, 2, 2, 2)
        self.label_27 = QLabel(self.visdom_enable)
        self.label_27.setObjectName(u"label_27")

        self.formLayout_9.setWidget(0, QFormLayout.ItemRole.LabelRole, self.label_27)

        self.label_28 = QLabel(self.visdom_enable)
        self.label_28.setObjectName(u"label_28")

        self.formLayout_9.setWidget(1, QFormLayout.ItemRole.LabelRole, self.label_28)

        self.label_29 = QLabel(self.visdom_enable)
        self.label_29.setObjectName(u"label_29")

        self.formLayout_9.setWidget(2, QFormLayout.ItemRole.LabelRole, self.label_29)

        self.label_30 = QLabel(self.visdom_enable)
        self.label_30.setObjectName(u"label_30")

        self.formLayout_9.setWidget(3, QFormLayout.ItemRole.LabelRole, self.label_30)

        self.visdom_env_name = QLineEdit(self.visdom_enable)
        self.visdom_env_name.setObjectName(u"visdom_env_name")

        self.formLayout_9.setWidget(0, QFormLayout.ItemRole.FieldRole, self.visdom_env_name)

        self.visdom_port = QSpinBox(self.visdom_enable)
        self.visdom_port.setObjectName(u"visdom_port")
        self.visdom_port.setMinimum(1)
        self.visdom_port.setMaximum(65535)
        self.visdom_port.setValue(8097)

        self.formLayout_9.setWidget(1, QFormLayout.ItemRole.FieldRole, self.visdom_port)

        self.visdom_image_size = QSpinBox(self.visdom_enable)
        self.visdom_image_size.setObjectName(u"visdom_image_size")
        self.visdom_image_size.setMinimum(32)
        self.visdom_image_size.setMaximum(4096)
        self.visdom_image_size.setValue(400)

        self.formLayout_9.setWidget(2, QFormLayout.ItemRole.FieldRole, self.visdom_image_size)

        self.visdom_update_frequency = QSpinBox(self.visdom_enable)
        self.visdom_update_frequency.setObjectName(u"visdom_update_frequency")
        self.visdom_update_frequency.setMinimum(1)
        self.visdom_update_frequency.setMaximum(99999)
        self.visdom_update_frequency.setValue(100)

        self.formLayout_9.setWidget(3, QFormLayout.ItemRole.FieldRole, self.visdom_update_frequency)


        self.verticalLayout_18.addWidget(self.visdom_enable)

        self.verticalSpacer_5 = QSpacerItem(20, 141, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.verticalLayout_18.addItem(self.verticalSpacer_5)

        self.scrollArea_5.setWidget(self.scrollAreaWidgetContents_5)

        self.verticalLayout_10.addWidget(self.scrollArea_5)

        self.tabWidget_2.addTab(self.tab_2, "")

        self.verticalLayout_11.addWidget(self.tabWidget_2)

        self.scrollArea.setWidget(self.scrollAreaWidgetContents)

        self.verticalLayout_13.addWidget(self.scrollArea)


        self.verticalLayout_6.addWidget(self.train_settings)

        self.frame_8 = QFrame(self.dockWidgetContents)
        self.frame_8.setObjectName(u"frame_8")
        sizePolicy1.setHeightForWidth(self.frame_8.sizePolicy().hasHeightForWidth())
        self.frame_8.setSizePolicy(sizePolicy1)
        self.frame_8.setFrameShape(QFrame.StyledPanel)
        self.frame_8.setFrameShadow(QFrame.Sunken)
        self.frame_8.setLineWidth(2)
        self.verticalLayout_5 = QVBoxLayout(self.frame_8)
        self.verticalLayout_5.setObjectName(u"verticalLayout_5")
        self.verticalLayout_5.setContentsMargins(3, 3, 3, 3)
        self.train_start = QPushButton(self.frame_8)
        self.train_start.setObjectName(u"train_start")

        self.verticalLayout_5.addWidget(self.train_start)

        self.train_pause = QPushButton(self.frame_8)
        self.train_pause.setObjectName(u"train_pause")

        self.verticalLayout_5.addWidget(self.train_pause)

        self.train_stop = QPushButton(self.frame_8)
        self.train_stop.setObjectName(u"train_stop")

        self.verticalLayout_5.addWidget(self.train_stop)


        self.verticalLayout_6.addWidget(self.frame_8)

        self.experiment_settings.setWidget(self.dockWidgetContents)
        MainWindow.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.experiment_settings)
        self.dockWidget_2 = QDockWidget(MainWindow)
        self.dockWidget_2.setObjectName(u"dockWidget_2")
        self.dockWidget_2.setFloating(False)
        self.dockWidget_2.setFeatures(QDockWidget.NoDockWidgetFeatures)
        self.dockWidget_2.setAllowedAreas(Qt.BottomDockWidgetArea)
        self.dockWidgetContents_2 = QWidget()
        self.dockWidgetContents_2.setObjectName(u"dockWidgetContents_2")
        sizePolicy3.setHeightForWidth(self.dockWidgetContents_2.sizePolicy().hasHeightForWidth())
        self.dockWidgetContents_2.setSizePolicy(sizePolicy3)
        self.verticalLayout_7 = QVBoxLayout(self.dockWidgetContents_2)
        self.verticalLayout_7.setSpacing(2)
        self.verticalLayout_7.setObjectName(u"verticalLayout_7")
        self.verticalLayout_7.setContentsMargins(2, 2, 2, 2)
        self.frame_24 = QFrame(self.dockWidgetContents_2)
        self.frame_24.setObjectName(u"frame_24")
        sizePolicy9 = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Ignored)
        sizePolicy9.setHorizontalStretch(0)
        sizePolicy9.setVerticalStretch(0)
        sizePolicy9.setHeightForWidth(self.frame_24.sizePolicy().hasHeightForWidth())
        self.frame_24.setSizePolicy(sizePolicy9)
        self.frame_24.setFrameShape(QFrame.StyledPanel)
        self.frame_24.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_32 = QHBoxLayout(self.frame_24)
        self.horizontalLayout_32.setSpacing(2)
        self.horizontalLayout_32.setObjectName(u"horizontalLayout_32")
        self.horizontalLayout_32.setContentsMargins(0, 0, 0, 0)
        self.groupBox_10 = QGroupBox(self.frame_24)
        self.groupBox_10.setObjectName(u"groupBox_10")
        sizePolicy.setHeightForWidth(self.groupBox_10.sizePolicy().hasHeightForWidth())
        self.groupBox_10.setSizePolicy(sizePolicy)
        self.verticalLayout_30 = QVBoxLayout(self.groupBox_10)
        self.verticalLayout_30.setSpacing(2)
        self.verticalLayout_30.setObjectName(u"verticalLayout_30")
        self.verticalLayout_30.setContentsMargins(3, 3, 3, 3)
        self.frame_36 = QFrame(self.groupBox_10)
        self.frame_36.setObjectName(u"frame_36")
        self.frame_36.setFrameShape(QFrame.StyledPanel)
        self.frame_36.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_36 = QHBoxLayout(self.frame_36)
        self.horizontalLayout_36.setObjectName(u"horizontalLayout_36")
        self.horizontalLayout_36.setContentsMargins(0, 0, 0, 0)
        self.label_55 = QLabel(self.frame_36)
        self.label_55.setObjectName(u"label_55")
        sizePolicy.setHeightForWidth(self.label_55.sizePolicy().hasHeightForWidth())
        self.label_55.setSizePolicy(sizePolicy)
        font4 = QFont()
        font4.setBold(True)
        self.label_55.setFont(font4)

        self.horizontalLayout_36.addWidget(self.label_55)

        self.line_10 = QFrame(self.frame_36)
        self.line_10.setObjectName(u"line_10")
        self.line_10.setFrameShadow(QFrame.Raised)
        self.line_10.setFrameShape(QFrame.Shape.HLine)

        self.horizontalLayout_36.addWidget(self.line_10)


        self.verticalLayout_30.addWidget(self.frame_36)

        self.frame_37 = QFrame(self.groupBox_10)
        self.frame_37.setObjectName(u"frame_37")
        sizePolicy1.setHeightForWidth(self.frame_37.sizePolicy().hasHeightForWidth())
        self.frame_37.setSizePolicy(sizePolicy1)
        self.frame_37.setFrameShape(QFrame.StyledPanel)
        self.frame_37.setFrameShadow(QFrame.Raised)
        self.formLayout_19 = QFormLayout(self.frame_37)
        self.formLayout_19.setObjectName(u"formLayout_19")
        self.formLayout_19.setHorizontalSpacing(6)
        self.formLayout_19.setVerticalSpacing(2)
        self.formLayout_19.setContentsMargins(0, 0, 0, 0)
        self.label_56 = QLabel(self.frame_37)
        self.label_56.setObjectName(u"label_56")

        self.formLayout_19.setWidget(0, QFormLayout.ItemRole.LabelRole, self.label_56)

        self.performance_epoch_fastest_2 = QLabel(self.frame_37)
        self.performance_epoch_fastest_2.setObjectName(u"performance_epoch_fastest_2")

        self.formLayout_19.setWidget(0, QFormLayout.ItemRole.FieldRole, self.performance_epoch_fastest_2)


        self.verticalLayout_30.addWidget(self.frame_37)

        self.frame_34 = QFrame(self.groupBox_10)
        self.frame_34.setObjectName(u"frame_34")
        self.frame_34.setFrameShape(QFrame.StyledPanel)
        self.frame_34.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_34 = QHBoxLayout(self.frame_34)
        self.horizontalLayout_34.setObjectName(u"horizontalLayout_34")
        self.horizontalLayout_34.setContentsMargins(0, 0, 0, 0)
        self.label_48 = QLabel(self.frame_34)
        self.label_48.setObjectName(u"label_48")
        sizePolicy.setHeightForWidth(self.label_48.sizePolicy().hasHeightForWidth())
        self.label_48.setSizePolicy(sizePolicy)
        self.label_48.setFont(font4)

        self.horizontalLayout_34.addWidget(self.label_48)

        self.line_8 = QFrame(self.frame_34)
        self.line_8.setObjectName(u"line_8")
        self.line_8.setFrameShadow(QFrame.Raised)
        self.line_8.setFrameShape(QFrame.Shape.HLine)

        self.horizontalLayout_34.addWidget(self.line_8)


        self.verticalLayout_30.addWidget(self.frame_34)

        self.frame_30 = QFrame(self.groupBox_10)
        self.frame_30.setObjectName(u"frame_30")
        sizePolicy1.setHeightForWidth(self.frame_30.sizePolicy().hasHeightForWidth())
        self.frame_30.setSizePolicy(sizePolicy1)
        self.frame_30.setFrameShape(QFrame.StyledPanel)
        self.frame_30.setFrameShadow(QFrame.Raised)
        self.formLayout_17 = QFormLayout(self.frame_30)
        self.formLayout_17.setObjectName(u"formLayout_17")
        self.formLayout_17.setHorizontalSpacing(6)
        self.formLayout_17.setVerticalSpacing(2)
        self.formLayout_17.setContentsMargins(0, 0, 0, 0)
        self.label_47 = QLabel(self.frame_30)
        self.label_47.setObjectName(u"label_47")

        self.formLayout_17.setWidget(0, QFormLayout.ItemRole.LabelRole, self.label_47)

        self.performance_epoch_fastest = QLabel(self.frame_30)
        self.performance_epoch_fastest.setObjectName(u"performance_epoch_fastest")

        self.formLayout_17.setWidget(0, QFormLayout.ItemRole.FieldRole, self.performance_epoch_fastest)

        self.label_50 = QLabel(self.frame_30)
        self.label_50.setObjectName(u"label_50")

        self.formLayout_17.setWidget(1, QFormLayout.ItemRole.LabelRole, self.label_50)

        self.performance_epoch_slowest = QLabel(self.frame_30)
        self.performance_epoch_slowest.setObjectName(u"performance_epoch_slowest")

        self.formLayout_17.setWidget(1, QFormLayout.ItemRole.FieldRole, self.performance_epoch_slowest)

        self.label_51 = QLabel(self.frame_30)
        self.label_51.setObjectName(u"label_51")

        self.formLayout_17.setWidget(2, QFormLayout.ItemRole.LabelRole, self.label_51)

        self.performance_epoch_average = QLabel(self.frame_30)
        self.performance_epoch_average.setObjectName(u"performance_epoch_average")

        self.formLayout_17.setWidget(2, QFormLayout.ItemRole.FieldRole, self.performance_epoch_average)


        self.verticalLayout_30.addWidget(self.frame_30)

        self.frame_35 = QFrame(self.groupBox_10)
        self.frame_35.setObjectName(u"frame_35")
        self.frame_35.setFrameShape(QFrame.StyledPanel)
        self.frame_35.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_35 = QHBoxLayout(self.frame_35)
        self.horizontalLayout_35.setObjectName(u"horizontalLayout_35")
        self.horizontalLayout_35.setContentsMargins(0, 0, 0, 0)
        self.label_49 = QLabel(self.frame_35)
        self.label_49.setObjectName(u"label_49")
        sizePolicy.setHeightForWidth(self.label_49.sizePolicy().hasHeightForWidth())
        self.label_49.setSizePolicy(sizePolicy)
        self.label_49.setFont(font4)

        self.horizontalLayout_35.addWidget(self.label_49)

        self.line_9 = QFrame(self.frame_35)
        self.line_9.setObjectName(u"line_9")
        self.line_9.setFrameShadow(QFrame.Raised)
        self.line_9.setFrameShape(QFrame.Shape.HLine)

        self.horizontalLayout_35.addWidget(self.line_9)


        self.verticalLayout_30.addWidget(self.frame_35)

        self.frame_32 = QFrame(self.groupBox_10)
        self.frame_32.setObjectName(u"frame_32")
        sizePolicy1.setHeightForWidth(self.frame_32.sizePolicy().hasHeightForWidth())
        self.frame_32.setSizePolicy(sizePolicy1)
        self.frame_32.setFrameShape(QFrame.StyledPanel)
        self.frame_32.setFrameShadow(QFrame.Raised)
        self.formLayout_18 = QFormLayout(self.frame_32)
        self.formLayout_18.setObjectName(u"formLayout_18")
        self.formLayout_18.setHorizontalSpacing(6)
        self.formLayout_18.setVerticalSpacing(2)
        self.formLayout_18.setContentsMargins(0, 0, 0, 0)
        self.label_52 = QLabel(self.frame_32)
        self.label_52.setObjectName(u"label_52")

        self.formLayout_18.setWidget(0, QFormLayout.ItemRole.LabelRole, self.label_52)

        self.performance_iter_fastest = QLabel(self.frame_32)
        self.performance_iter_fastest.setObjectName(u"performance_iter_fastest")

        self.formLayout_18.setWidget(0, QFormLayout.ItemRole.FieldRole, self.performance_iter_fastest)

        self.label_53 = QLabel(self.frame_32)
        self.label_53.setObjectName(u"label_53")

        self.formLayout_18.setWidget(1, QFormLayout.ItemRole.LabelRole, self.label_53)

        self.performance_iter_slowest = QLabel(self.frame_32)
        self.performance_iter_slowest.setObjectName(u"performance_iter_slowest")

        self.formLayout_18.setWidget(1, QFormLayout.ItemRole.FieldRole, self.performance_iter_slowest)

        self.label_54 = QLabel(self.frame_32)
        self.label_54.setObjectName(u"label_54")

        self.formLayout_18.setWidget(2, QFormLayout.ItemRole.LabelRole, self.label_54)

        self.performance_iter_average = QLabel(self.frame_32)
        self.performance_iter_average.setObjectName(u"performance_iter_average")

        self.formLayout_18.setWidget(2, QFormLayout.ItemRole.FieldRole, self.performance_iter_average)


        self.verticalLayout_30.addWidget(self.frame_32)

        self.verticalSpacer_7 = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.verticalLayout_30.addItem(self.verticalSpacer_7)


        self.horizontalLayout_32.addWidget(self.groupBox_10)

        self.performance_monitor_group = QGroupBox(self.frame_24)
        self.performance_monitor_group.setObjectName(u"performance_monitor_group")
        self.verticalLayout_26 = QVBoxLayout(self.performance_monitor_group)
        self.verticalLayout_26.setSpacing(0)
        self.verticalLayout_26.setObjectName(u"verticalLayout_26")
        self.verticalLayout_26.setContentsMargins(3, 3, 3, 3)
        self.frame_33 = QFrame(self.performance_monitor_group)
        self.frame_33.setObjectName(u"frame_33")
        self.frame_33.setFrameShape(QFrame.StyledPanel)
        self.frame_33.setFrameShadow(QFrame.Raised)
        self.verticalLayout_29 = QVBoxLayout(self.frame_33)
        self.verticalLayout_29.setSpacing(2)
        self.verticalLayout_29.setObjectName(u"verticalLayout_29")
        self.verticalLayout_29.setContentsMargins(0, 0, 0, 0)
        self.frame_27 = QFrame(self.frame_33)
        self.frame_27.setObjectName(u"frame_27")
        self.frame_27.setFrameShape(QFrame.StyledPanel)
        self.frame_27.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_37 = QHBoxLayout(self.frame_27)
        self.horizontalLayout_37.setSpacing(2)
        self.horizontalLayout_37.setObjectName(u"horizontalLayout_37")
        self.horizontalLayout_37.setContentsMargins(0, 0, 0, 0)
        self.perf_graph_time_label = QLabel(self.frame_27)
        self.perf_graph_time_label.setObjectName(u"perf_graph_time_label")
        sizePolicy.setHeightForWidth(self.perf_graph_time_label.sizePolicy().hasHeightForWidth())
        self.perf_graph_time_label.setSizePolicy(sizePolicy)
        self.perf_graph_time_label.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_37.addWidget(self.perf_graph_time_label)

        self.perf_graph_layout_frame = QFrame(self.frame_27)
        self.perf_graph_layout_frame.setObjectName(u"perf_graph_layout_frame")
        self.perf_graph_layout_frame.setFrameShape(QFrame.StyledPanel)
        self.perf_graph_layout_frame.setFrameShadow(QFrame.Raised)
        self.performance_graph_layout = QVBoxLayout(self.perf_graph_layout_frame)
        self.performance_graph_layout.setObjectName(u"performance_graph_layout")
        self.performance_graph_layout.setContentsMargins(0, 0, 0, 0)

        self.horizontalLayout_37.addWidget(self.perf_graph_layout_frame)


        self.verticalLayout_29.addWidget(self.frame_27)

        self.perf_graph_epoch_label = QLabel(self.frame_33)
        self.perf_graph_epoch_label.setObjectName(u"perf_graph_epoch_label")
        sizePolicy1.setHeightForWidth(self.perf_graph_epoch_label.sizePolicy().hasHeightForWidth())
        self.perf_graph_epoch_label.setSizePolicy(sizePolicy1)
        self.perf_graph_epoch_label.setAlignment(Qt.AlignCenter)

        self.verticalLayout_29.addWidget(self.perf_graph_epoch_label)


        self.verticalLayout_26.addWidget(self.frame_33)

        self.frame_26 = QFrame(self.performance_monitor_group)
        self.frame_26.setObjectName(u"frame_26")
        sizePolicy1.setHeightForWidth(self.frame_26.sizePolicy().hasHeightForWidth())
        self.frame_26.setSizePolicy(sizePolicy1)
        self.frame_26.setFrameShape(QFrame.Box)
        self.frame_26.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_33 = QHBoxLayout(self.frame_26)
        self.horizontalLayout_33.setObjectName(u"horizontalLayout_33")
        self.horizontalLayout_33.setContentsMargins(2, 2, 2, 2)
        self.performance_graph_reset_framing = QPushButton(self.frame_26)
        self.performance_graph_reset_framing.setObjectName(u"performance_graph_reset_framing")

        self.horizontalLayout_33.addWidget(self.performance_graph_reset_framing)

        self.performance_graph_clear = QPushButton(self.frame_26)
        self.performance_graph_clear.setObjectName(u"performance_graph_clear")

        self.horizontalLayout_33.addWidget(self.performance_graph_clear)

        self.horizontalSpacer_6 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_33.addItem(self.horizontalSpacer_6)


        self.verticalLayout_26.addWidget(self.frame_26)


        self.horizontalLayout_32.addWidget(self.performance_monitor_group)

        self.groupBox_7 = QGroupBox(self.frame_24)
        self.groupBox_7.setObjectName(u"groupBox_7")
        sizePolicy9.setHeightForWidth(self.groupBox_7.sizePolicy().hasHeightForWidth())
        self.groupBox_7.setSizePolicy(sizePolicy9)
        self.verticalLayout_9 = QVBoxLayout(self.groupBox_7)
        self.verticalLayout_9.setSpacing(2)
        self.verticalLayout_9.setObjectName(u"verticalLayout_9")
        self.verticalLayout_9.setContentsMargins(3, 3, 3, 3)
        self.log_output_frozen_header = QLabel(self.groupBox_7)
        self.log_output_frozen_header.setObjectName(u"log_output_frozen_header")
        font5 = QFont()
        font5.setBold(False)
        font5.setItalic(True)
        self.log_output_frozen_header.setFont(font5)
        self.log_output_frozen_header.setAlignment(Qt.AlignCenter)

        self.verticalLayout_9.addWidget(self.log_output_frozen_header)

        self.frame_9 = QFrame(self.groupBox_7)
        self.frame_9.setObjectName(u"frame_9")
        self.frame_9.setFrameShape(QFrame.StyledPanel)
        self.frame_9.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_29 = QHBoxLayout(self.frame_9)
        self.horizontalLayout_29.setObjectName(u"horizontalLayout_29")
        self.horizontalLayout_29.setContentsMargins(0, 0, 0, 0)
        self.output_log = QTextEdit(self.frame_9)
        self.output_log.setObjectName(u"output_log")
        sizePolicy10 = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        sizePolicy10.setHorizontalStretch(0)
        sizePolicy10.setVerticalStretch(0)
        sizePolicy10.setHeightForWidth(self.output_log.sizePolicy().hasHeightForWidth())
        self.output_log.setSizePolicy(sizePolicy10)
        self.output_log.setFrameShape(QFrame.Box)
        self.output_log.setLineWidth(2)
        self.output_log.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.output_log.setSizeAdjustPolicy(QAbstractScrollArea.AdjustIgnored)
        self.output_log.setLineWrapMode(QTextEdit.NoWrap)
        self.output_log.setReadOnly(True)

        self.horizontalLayout_29.addWidget(self.output_log)

        self.frame_15 = QFrame(self.frame_9)
        self.frame_15.setObjectName(u"frame_15")
        self.frame_15.setFrameShape(QFrame.StyledPanel)
        self.frame_15.setFrameShadow(QFrame.Raised)
        self.verticalLayout_23 = QVBoxLayout(self.frame_15)
        self.verticalLayout_23.setSpacing(2)
        self.verticalLayout_23.setObjectName(u"verticalLayout_23")
        self.verticalLayout_23.setContentsMargins(0, 0, 0, 0)
        self.label_39 = QLabel(self.frame_15)
        self.label_39.setObjectName(u"label_39")

        self.verticalLayout_23.addWidget(self.label_39)

        self.line_5 = QFrame(self.frame_15)
        self.line_5.setObjectName(u"line_5")
        self.line_5.setFrameShape(QFrame.Shape.HLine)
        self.line_5.setFrameShadow(QFrame.Shadow.Sunken)

        self.verticalLayout_23.addWidget(self.line_5)

        self.export_output_log = QPushButton(self.frame_15)
        self.export_output_log.setObjectName(u"export_output_log")

        self.verticalLayout_23.addWidget(self.export_output_log)

        self.freeze_output_log = QPushButton(self.frame_15)
        self.freeze_output_log.setObjectName(u"freeze_output_log")

        self.verticalLayout_23.addWidget(self.freeze_output_log)

        self.clear_output_log = QPushButton(self.frame_15)
        self.clear_output_log.setObjectName(u"clear_output_log")

        self.verticalLayout_23.addWidget(self.clear_output_log)

        self.verticalSpacer_6 = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.verticalLayout_23.addItem(self.verticalSpacer_6)


        self.horizontalLayout_29.addWidget(self.frame_15)


        self.verticalLayout_9.addWidget(self.frame_9)

        self.frame_14 = QFrame(self.groupBox_7)
        self.frame_14.setObjectName(u"frame_14")
        sizePolicy1.setHeightForWidth(self.frame_14.sizePolicy().hasHeightForWidth())
        self.frame_14.setSizePolicy(sizePolicy1)
        self.frame_14.setFrameShape(QFrame.StyledPanel)
        self.frame_14.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_8 = QHBoxLayout(self.frame_14)
        self.horizontalLayout_8.setObjectName(u"horizontalLayout_8")
        self.horizontalLayout_8.setContentsMargins(0, 0, 0, 0)
        self.autoscroll_log = QCheckBox(self.frame_14)
        self.autoscroll_log.setObjectName(u"autoscroll_log")
        sizePolicy5.setHeightForWidth(self.autoscroll_log.sizePolicy().hasHeightForWidth())
        self.autoscroll_log.setSizePolicy(sizePolicy5)
        self.autoscroll_log.setChecked(True)

        self.horizontalLayout_8.addWidget(self.autoscroll_log)

        self.show_epoch_progress = QCheckBox(self.frame_14)
        self.show_epoch_progress.setObjectName(u"show_epoch_progress")
        sizePolicy5.setHeightForWidth(self.show_epoch_progress.sizePolicy().hasHeightForWidth())
        self.show_epoch_progress.setSizePolicy(sizePolicy5)
        self.show_epoch_progress.setChecked(True)

        self.horizontalLayout_8.addWidget(self.show_epoch_progress)

        self.show_train_progress = QCheckBox(self.frame_14)
        self.show_train_progress.setObjectName(u"show_train_progress")
        self.show_train_progress.setChecked(True)

        self.horizontalLayout_8.addWidget(self.show_train_progress)


        self.verticalLayout_9.addWidget(self.frame_14)


        self.horizontalLayout_32.addWidget(self.groupBox_7)


        self.verticalLayout_7.addWidget(self.frame_24)

        self.frame_10 = QFrame(self.dockWidgetContents_2)
        self.frame_10.setObjectName(u"frame_10")
        sizePolicy1.setHeightForWidth(self.frame_10.sizePolicy().hasHeightForWidth())
        self.frame_10.setSizePolicy(sizePolicy1)
        self.frame_10.setFrameShape(QFrame.Box)
        self.frame_10.setFrameShadow(QFrame.Sunken)
        self.horizontalLayout_12 = QHBoxLayout(self.frame_10)
        self.horizontalLayout_12.setObjectName(u"horizontalLayout_12")
        self.horizontalLayout_12.setContentsMargins(0, 0, 0, 0)
        self.frame_28 = QFrame(self.frame_10)
        self.frame_28.setObjectName(u"frame_28")
        sizePolicy11 = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        sizePolicy11.setHorizontalStretch(2)
        sizePolicy11.setVerticalStretch(0)
        sizePolicy11.setHeightForWidth(self.frame_28.sizePolicy().hasHeightForWidth())
        self.frame_28.setSizePolicy(sizePolicy11)
        self.frame_28.setFrameShape(QFrame.StyledPanel)
        self.frame_28.setFrameShadow(QFrame.Raised)
        self.verticalLayout_27 = QVBoxLayout(self.frame_28)
        self.verticalLayout_27.setSpacing(1)
        self.verticalLayout_27.setObjectName(u"verticalLayout_27")
        self.verticalLayout_27.setContentsMargins(0, 0, 0, 0)
        self.epoch_progress = QProgressBar(self.frame_28)
        self.epoch_progress.setObjectName(u"epoch_progress")
        sizePolicy6.setHeightForWidth(self.epoch_progress.sizePolicy().hasHeightForWidth())
        self.epoch_progress.setSizePolicy(sizePolicy6)
        self.epoch_progress.setValue(0)
        self.epoch_progress.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)
        self.epoch_progress.setOrientation(Qt.Horizontal)
        self.epoch_progress.setInvertedAppearance(False)
        self.epoch_progress.setTextDirection(QProgressBar.TopToBottom)

        self.verticalLayout_27.addWidget(self.epoch_progress)

        self.label_45 = QLabel(self.frame_28)
        self.label_45.setObjectName(u"label_45")
        font6 = QFont()
        font6.setPointSize(7)
        self.label_45.setFont(font6)
        self.label_45.setAlignment(Qt.AlignCenter)

        self.verticalLayout_27.addWidget(self.label_45)


        self.horizontalLayout_12.addWidget(self.frame_28)

        self.frame_29 = QFrame(self.frame_10)
        self.frame_29.setObjectName(u"frame_29")
        sizePolicy12 = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        sizePolicy12.setHorizontalStretch(5)
        sizePolicy12.setVerticalStretch(0)
        sizePolicy12.setHeightForWidth(self.frame_29.sizePolicy().hasHeightForWidth())
        self.frame_29.setSizePolicy(sizePolicy12)
        self.frame_29.setFrameShape(QFrame.StyledPanel)
        self.frame_29.setFrameShadow(QFrame.Raised)
        self.verticalLayout_28 = QVBoxLayout(self.frame_29)
        self.verticalLayout_28.setSpacing(1)
        self.verticalLayout_28.setObjectName(u"verticalLayout_28")
        self.verticalLayout_28.setContentsMargins(0, 0, 0, 0)
        self.train_progress = QProgressBar(self.frame_29)
        self.train_progress.setObjectName(u"train_progress")
        sizePolicy6.setHeightForWidth(self.train_progress.sizePolicy().hasHeightForWidth())
        self.train_progress.setSizePolicy(sizePolicy6)
        self.train_progress.setValue(0)
        self.train_progress.setAlignment(Qt.AlignBottom|Qt.AlignRight|Qt.AlignTrailing)

        self.verticalLayout_28.addWidget(self.train_progress)

        self.label_46 = QLabel(self.frame_29)
        self.label_46.setObjectName(u"label_46")
        self.label_46.setFont(font6)
        self.label_46.setAlignment(Qt.AlignCenter)

        self.verticalLayout_28.addWidget(self.label_46)


        self.horizontalLayout_12.addWidget(self.frame_29)

        self.frame_22 = QFrame(self.frame_10)
        self.frame_22.setObjectName(u"frame_22")
        self.frame_22.setFrameShape(QFrame.Panel)
        self.frame_22.setFrameShadow(QFrame.Sunken)
        self.verticalLayout_31 = QVBoxLayout(self.frame_22)
        self.verticalLayout_31.setSpacing(1)
        self.verticalLayout_31.setObjectName(u"verticalLayout_31")
        self.verticalLayout_31.setContentsMargins(2, 2, 2, 2)
        self.current_epoch = QLCDNumber(self.frame_22)
        self.current_epoch.setObjectName(u"current_epoch")
        self.current_epoch.setFrameShadow(QFrame.Raised)
        self.current_epoch.setSegmentStyle(QLCDNumber.Filled)

        self.verticalLayout_31.addWidget(self.current_epoch)

        self.label_41 = QLabel(self.frame_22)
        self.label_41.setObjectName(u"label_41")
        font7 = QFont()
        font7.setPointSize(7)
        font7.setBold(True)
        self.label_41.setFont(font7)
        self.label_41.setAlignment(Qt.AlignCenter)

        self.verticalLayout_31.addWidget(self.label_41)


        self.horizontalLayout_12.addWidget(self.frame_22)


        self.verticalLayout_7.addWidget(self.frame_10)

        self.dockWidget_2.setWidget(self.dockWidgetContents_2)
        MainWindow.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.dockWidget_2)

        self.menubar.addAction(self.menuFile.menuAction())
        self.menuFile.addAction(self.actionLoad_From_Config)

        self.retranslateUi(MainWindow)

        self.direction.setCurrentIndex(-1)
        self.tabWidget_2.setCurrentIndex(0)


        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"MainWindow", None))
        self.actionLoad_From_Config.setText(QCoreApplication.translate("MainWindow", u"Load From Config", None))
        self.close_experiment_settings.setText("")
        self.label_19.setText(QCoreApplication.translate("MainWindow", u"A_real", None))
        self.label_20.setText(QCoreApplication.translate("MainWindow", u"B_fake", None))
        self.label_21.setText(QCoreApplication.translate("MainWindow", u"B_real", None))
        self.x_label.setText("")
        self.y_fake_label.setText("")
        self.y_label.setText("")
        self.label_44.setText(QCoreApplication.translate("MainWindow", u"Update Rate", None))
        self.menuFile.setTitle(QCoreApplication.translate("MainWindow", u"File", None))
        self.label_22.setText(QCoreApplication.translate("MainWindow", u"Experiment", None))
        self.hide_experiment_settings.setText(QCoreApplication.translate("MainWindow", u"Hide", None))
        self.label_2.setText(QCoreApplication.translate("MainWindow", u"Output Root", None))
        self.browse_output.setText(QCoreApplication.translate("MainWindow", u"Browse", None))
        self.label_3.setText(QCoreApplication.translate("MainWindow", u"Experiment Name", None))
        self.label_4.setText(QCoreApplication.translate("MainWindow", u"Version", None))
        self.init_from_last_train.setText(QCoreApplication.translate("MainWindow", u"Initialize settings from most recent train", None))
        self.label_23.setText(QCoreApplication.translate("MainWindow", u"Dataloader", None))
        self.hide_dataloader_settings.setText(QCoreApplication.translate("MainWindow", u"Hide", None))
        self.label.setText(QCoreApplication.translate("MainWindow", u"Dataset Root", None))
        self.browse_dataset.setText(QCoreApplication.translate("MainWindow", u"Browse", None))
        self.label_10.setText(QCoreApplication.translate("MainWindow", u"Load Size", None))
        self.label_11.setText(QCoreApplication.translate("MainWindow", u"Crop Size", None))
        self.label_9.setText(QCoreApplication.translate("MainWindow", u"Input Channels", None))
        self.label_5.setText(QCoreApplication.translate("MainWindow", u"Direction", None))
        self.direction.setCurrentText("")
        self.label_8.setText(QCoreApplication.translate("MainWindow", u"Batch Size", None))
        self.label_40.setText(QCoreApplication.translate("MainWindow", u"Train Settings", None))
        self.continue_train.setTitle(QCoreApplication.translate("MainWindow", u"Continue Train", None))
        self.label_16.setText(QCoreApplication.translate("MainWindow", u"Load Epoch", None))
        self.separate_lr_schedules.setText(QCoreApplication.translate("MainWindow", u"Separate Learning Rate Schedules", None))
        self.gen_schedule_box.setTitle(QCoreApplication.translate("MainWindow", u"Schedule", None))
        self.groupBox_4.setTitle(QCoreApplication.translate("MainWindow", u"Epochs", None))
        self.label_32.setText(QCoreApplication.translate("MainWindow", u"Epochs", None))
        self.label_31.setText(QCoreApplication.translate("MainWindow", u"Epochs Decay", None))
        self.groupBox_2.setTitle(QCoreApplication.translate("MainWindow", u"Learning rate", None))
        self.label_6.setText(QCoreApplication.translate("MainWindow", u"Initial", None))
        self.label_33.setText(QCoreApplication.translate("MainWindow", u"Target", None))
        self.groupBox_3.setTitle(QCoreApplication.translate("MainWindow", u"Optimizer", None))
        self.label_14.setText(QCoreApplication.translate("MainWindow", u"Beta1", None))
        self.disc_schedule_box.setTitle(QCoreApplication.translate("MainWindow", u"Discriminator", None))
        self.groupBox_5.setTitle(QCoreApplication.translate("MainWindow", u"Epochs", None))
        self.label_34.setText(QCoreApplication.translate("MainWindow", u"Epochs", None))
        self.label_35.setText(QCoreApplication.translate("MainWindow", u"Epochs Decay", None))
        self.groupBox_8.setTitle(QCoreApplication.translate("MainWindow", u"Learning rate", None))
        self.label_7.setText(QCoreApplication.translate("MainWindow", u"Initial", None))
        self.label_36.setText(QCoreApplication.translate("MainWindow", u"Target", None))
        self.groupBox_9.setTitle(QCoreApplication.translate("MainWindow", u"Optimizer", None))
        self.label_15.setText(QCoreApplication.translate("MainWindow", u"Beta1", None))
        self.tabWidget_2.setTabText(self.tabWidget_2.indexOf(self.tab), QCoreApplication.translate("MainWindow", u"Training", None))
        self.label_12.setText(QCoreApplication.translate("MainWindow", u"L1", None))
        self.label_13.setText(QCoreApplication.translate("MainWindow", u"Sobel", None))
        self.label_37.setText(QCoreApplication.translate("MainWindow", u"Laplacian", None))
        self.label_38.setText(QCoreApplication.translate("MainWindow", u"VGG", None))
        self.tabWidget_2.setTabText(self.tabWidget_2.indexOf(self.tab_3), QCoreApplication.translate("MainWindow", u"Loss", None))
        self.groupBox_6.setTitle(QCoreApplication.translate("MainWindow", u"Save Checkpoints", None))
        self.label_17.setText(QCoreApplication.translate("MainWindow", u"Enable", None))
        self.save_model.setText("")
        self.label_18.setText(QCoreApplication.translate("MainWindow", u"Frequency", None))
        self.label_42.setText(QCoreApplication.translate("MainWindow", u"Epochs", None))
        self.groupBox.setTitle(QCoreApplication.translate("MainWindow", u"Save Example Images", None))
        self.label_24.setText(QCoreApplication.translate("MainWindow", u"Enable", None))
        self.save_examples.setText("")
        self.label_25.setText(QCoreApplication.translate("MainWindow", u"Frequency", None))
        self.label_43.setText(QCoreApplication.translate("MainWindow", u"Epochs", None))
        self.label_26.setText(QCoreApplication.translate("MainWindow", u"Count", None))
        self.tabWidget_2.setTabText(self.tabWidget_2.indexOf(self.tab_4), QCoreApplication.translate("MainWindow", u"Saving", None))
        self.visdom_enable.setTitle(QCoreApplication.translate("MainWindow", u"Enable Visdom", None))
        self.label_27.setText(QCoreApplication.translate("MainWindow", u"Environment Name", None))
        self.label_28.setText(QCoreApplication.translate("MainWindow", u"Port", None))
        self.label_29.setText(QCoreApplication.translate("MainWindow", u"Image Size", None))
        self.label_30.setText(QCoreApplication.translate("MainWindow", u"Update Frequency", None))
        self.visdom_env_name.setText(QCoreApplication.translate("MainWindow", u"main", None))
        self.tabWidget_2.setTabText(self.tabWidget_2.indexOf(self.tab_2), QCoreApplication.translate("MainWindow", u"Visdom", None))
        self.train_start.setText(QCoreApplication.translate("MainWindow", u"Begin Training", None))
        self.train_pause.setText(QCoreApplication.translate("MainWindow", u"Pause Training", None))
        self.train_stop.setText(QCoreApplication.translate("MainWindow", u"Stop Training", None))
        self.dockWidget_2.setWindowTitle(QCoreApplication.translate("MainWindow", u"\u2191 Performance Monitor / Output Log", None))
        self.groupBox_10.setTitle(QCoreApplication.translate("MainWindow", u"Times", None))
        self.label_55.setText(QCoreApplication.translate("MainWindow", u"Train", None))
        self.label_56.setText(QCoreApplication.translate("MainWindow", u"Since Start:", None))
        self.performance_epoch_fastest_2.setText(QCoreApplication.translate("MainWindow", u"00.000", None))
        self.label_48.setText(QCoreApplication.translate("MainWindow", u"Epoch", None))
        self.label_47.setText(QCoreApplication.translate("MainWindow", u"Fastest:", None))
        self.performance_epoch_fastest.setText(QCoreApplication.translate("MainWindow", u"00.000", None))
        self.label_50.setText(QCoreApplication.translate("MainWindow", u"Slowest:", None))
        self.performance_epoch_slowest.setText(QCoreApplication.translate("MainWindow", u"00.000", None))
        self.label_51.setText(QCoreApplication.translate("MainWindow", u"Average:", None))
        self.performance_epoch_average.setText(QCoreApplication.translate("MainWindow", u"00.000", None))
        self.label_49.setText(QCoreApplication.translate("MainWindow", u"Iteration", None))
        self.label_52.setText(QCoreApplication.translate("MainWindow", u"Fastest:", None))
        self.performance_iter_fastest.setText(QCoreApplication.translate("MainWindow", u"00.000", None))
        self.label_53.setText(QCoreApplication.translate("MainWindow", u"Slowest:", None))
        self.performance_iter_slowest.setText(QCoreApplication.translate("MainWindow", u"00.000", None))
        self.label_54.setText(QCoreApplication.translate("MainWindow", u"Average:", None))
        self.performance_iter_average.setText(QCoreApplication.translate("MainWindow", u"00.000", None))
        self.performance_monitor_group.setTitle(QCoreApplication.translate("MainWindow", u"Performance Monitor", None))
        self.perf_graph_time_label.setText("")
        self.perf_graph_epoch_label.setText("")
        self.performance_graph_reset_framing.setText(QCoreApplication.translate("MainWindow", u"Reset Framing", None))
        self.performance_graph_clear.setText(QCoreApplication.translate("MainWindow", u"Clear Graph", None))
        self.groupBox_7.setTitle(QCoreApplication.translate("MainWindow", u"Log", None))
        self.log_output_frozen_header.setText(QCoreApplication.translate("MainWindow", u"* OUTPUT FROZEN *", None))
        self.output_log.setDocumentTitle("")
        self.label_39.setText(QCoreApplication.translate("MainWindow", u"Log Functions", None))
        self.export_output_log.setText(QCoreApplication.translate("MainWindow", u"Export to File", None))
        self.freeze_output_log.setText(QCoreApplication.translate("MainWindow", u"Freeze Ouput", None))
        self.clear_output_log.setText(QCoreApplication.translate("MainWindow", u"Clear", None))
        self.autoscroll_log.setText(QCoreApplication.translate("MainWindow", u"Scroll Log Automatically", None))
        self.show_epoch_progress.setText(QCoreApplication.translate("MainWindow", u"Show Current Epoch Progress", None))
        self.show_train_progress.setText(QCoreApplication.translate("MainWindow", u"Show Train Progress", None))
        self.label_45.setText(QCoreApplication.translate("MainWindow", u"Current Epoch Progress", None))
        self.label_46.setText(QCoreApplication.translate("MainWindow", u"Current Train Progress", None))
        self.label_41.setText(QCoreApplication.translate("MainWindow", u"Current Epoch", None))
    # retranslateUi

