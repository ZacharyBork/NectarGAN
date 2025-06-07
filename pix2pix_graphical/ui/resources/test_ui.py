# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'test.ui'
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
    QLineEdit, QMainWindow, QProgressBar, QPushButton,
    QScrollArea, QSizePolicy, QSlider, QSpacerItem,
    QSpinBox, QSplitter, QStackedWidget, QStatusBar,
    QTabWidget, QTextEdit, QVBoxLayout, QWidget)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(2006, 1140)
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
        self.close_experiment_settings.setMaximumSize(QSize(6, 16777215))
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
        self.frame_7.setLineWidth(2)
        self.verticalLayout_8 = QVBoxLayout(self.frame_7)
        self.verticalLayout_8.setSpacing(0)
        self.verticalLayout_8.setObjectName(u"verticalLayout_8")
        self.verticalLayout_8.setContentsMargins(3, 3, 3, 3)
        self.frame_16 = QFrame(self.frame_7)
        self.frame_16.setObjectName(u"frame_16")
        self.frame_16.setFrameShape(QFrame.StyledPanel)
        self.frame_16.setFrameShadow(QFrame.Raised)
        self.verticalLayout_6 = QVBoxLayout(self.frame_16)
        self.verticalLayout_6.setObjectName(u"verticalLayout_6")
        self.verticalLayout_6.setContentsMargins(0, 0, 0, 0)
        self.visualizer_splitter = QSplitter(self.frame_16)
        self.visualizer_splitter.setObjectName(u"visualizer_splitter")
        self.visualizer_splitter.setOrientation(Qt.Vertical)
        self.visualizer_splitter.setHandleWidth(3)
        self.frame_4 = QFrame(self.visualizer_splitter)
        self.frame_4.setObjectName(u"frame_4")
        sizePolicy1 = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.frame_4.sizePolicy().hasHeightForWidth())
        self.frame_4.setSizePolicy(sizePolicy1)
        self.frame_4.setFrameShape(QFrame.StyledPanel)
        self.frame_4.setFrameShadow(QFrame.Raised)
        self.verticalLayout_32 = QVBoxLayout(self.frame_4)
        self.verticalLayout_32.setSpacing(3)
        self.verticalLayout_32.setObjectName(u"verticalLayout_32")
        self.verticalLayout_32.setContentsMargins(0, 0, 0, 0)
        self.frame_12 = QFrame(self.frame_4)
        self.frame_12.setObjectName(u"frame_12")
        sizePolicy2 = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        sizePolicy2.setHorizontalStretch(0)
        sizePolicy2.setVerticalStretch(0)
        sizePolicy2.setHeightForWidth(self.frame_12.sizePolicy().hasHeightForWidth())
        self.frame_12.setSizePolicy(sizePolicy2)
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


        self.verticalLayout_32.addWidget(self.frame_12)

        self.frame_11 = QFrame(self.frame_4)
        self.frame_11.setObjectName(u"frame_11")
        sizePolicy1.setHeightForWidth(self.frame_11.sizePolicy().hasHeightForWidth())
        self.frame_11.setSizePolicy(sizePolicy1)
        self.frame_11.setFrameShape(QFrame.Panel)
        self.frame_11.setFrameShadow(QFrame.Sunken)
        self.horizontalLayout_13 = QHBoxLayout(self.frame_11)
        self.horizontalLayout_13.setObjectName(u"horizontalLayout_13")
        self.x_label = QLabel(self.frame_11)
        self.x_label.setObjectName(u"x_label")
        sizePolicy3 = QSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        sizePolicy3.setHorizontalStretch(0)
        sizePolicy3.setVerticalStretch(0)
        sizePolicy3.setHeightForWidth(self.x_label.sizePolicy().hasHeightForWidth())
        self.x_label.setSizePolicy(sizePolicy3)
        self.x_label.setFrameShape(QFrame.Panel)
        self.x_label.setFrameShadow(QFrame.Sunken)
        self.x_label.setAlignment(Qt.AlignHCenter|Qt.AlignTop)

        self.horizontalLayout_13.addWidget(self.x_label)

        self.y_fake_label = QLabel(self.frame_11)
        self.y_fake_label.setObjectName(u"y_fake_label")
        sizePolicy3.setHeightForWidth(self.y_fake_label.sizePolicy().hasHeightForWidth())
        self.y_fake_label.setSizePolicy(sizePolicy3)
        self.y_fake_label.setFrameShape(QFrame.Panel)
        self.y_fake_label.setFrameShadow(QFrame.Sunken)
        self.y_fake_label.setAlignment(Qt.AlignHCenter|Qt.AlignTop)

        self.horizontalLayout_13.addWidget(self.y_fake_label)

        self.y_label = QLabel(self.frame_11)
        self.y_label.setObjectName(u"y_label")
        sizePolicy3.setHeightForWidth(self.y_label.sizePolicy().hasHeightForWidth())
        self.y_label.setSizePolicy(sizePolicy3)
        self.y_label.setFrameShape(QFrame.Panel)
        self.y_label.setFrameShadow(QFrame.Sunken)
        self.y_label.setAlignment(Qt.AlignHCenter|Qt.AlignTop)

        self.horizontalLayout_13.addWidget(self.y_label)


        self.verticalLayout_32.addWidget(self.frame_11)

        self.visualizer_splitter.addWidget(self.frame_4)
        self.frame_14 = QFrame(self.visualizer_splitter)
        self.frame_14.setObjectName(u"frame_14")
        sizePolicy2.setHeightForWidth(self.frame_14.sizePolicy().hasHeightForWidth())
        self.frame_14.setSizePolicy(sizePolicy2)
        self.frame_14.setFrameShape(QFrame.StyledPanel)
        self.frame_14.setFrameShadow(QFrame.Raised)
        self.verticalLayout_34 = QVBoxLayout(self.frame_14)
        self.verticalLayout_34.setObjectName(u"verticalLayout_34")
        self.verticalLayout_34.setContentsMargins(0, 0, 0, 0)
        self.splitter_2 = QSplitter(self.frame_14)
        self.splitter_2.setObjectName(u"splitter_2")
        sizePolicy4 = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
        sizePolicy4.setHorizontalStretch(0)
        sizePolicy4.setVerticalStretch(0)
        sizePolicy4.setHeightForWidth(self.splitter_2.sizePolicy().hasHeightForWidth())
        self.splitter_2.setSizePolicy(sizePolicy4)
        self.splitter_2.setOrientation(Qt.Horizontal)
        self.splitter_2.setOpaqueResize(True)
        self.splitter_2.setHandleWidth(3)
        self.frame_17 = QFrame(self.splitter_2)
        self.frame_17.setObjectName(u"frame_17")
        self.frame_17.setFrameShape(QFrame.StyledPanel)
        self.frame_17.setFrameShadow(QFrame.Raised)
        self.verticalLayout_37 = QVBoxLayout(self.frame_17)
        self.verticalLayout_37.setSpacing(1)
        self.verticalLayout_37.setObjectName(u"verticalLayout_37")
        self.verticalLayout_37.setContentsMargins(1, 1, 1, 1)
        self.label_6 = QLabel(self.frame_17)
        self.label_6.setObjectName(u"label_6")
        sizePolicy2.setHeightForWidth(self.label_6.sizePolicy().hasHeightForWidth())
        self.label_6.setSizePolicy(sizePolicy2)
        font2 = QFont()
        font2.setPointSize(12)
        self.label_6.setFont(font2)
        self.label_6.setAlignment(Qt.AlignCenter)

        self.verticalLayout_37.addWidget(self.label_6)

        self.loss_g_graph_frame = QFrame(self.frame_17)
        self.loss_g_graph_frame.setObjectName(u"loss_g_graph_frame")
        self.loss_g_graph_frame.setFrameShape(QFrame.StyledPanel)
        self.loss_g_graph_frame.setFrameShadow(QFrame.Raised)
        self.loss_g_graph_layout = QVBoxLayout(self.loss_g_graph_frame)
        self.loss_g_graph_layout.setSpacing(2)
        self.loss_g_graph_layout.setObjectName(u"loss_g_graph_layout")
        self.loss_g_graph_layout.setContentsMargins(0, 0, 0, 0)

        self.verticalLayout_37.addWidget(self.loss_g_graph_frame)

        self.splitter_2.addWidget(self.frame_17)
        self.frame_47 = QFrame(self.splitter_2)
        self.frame_47.setObjectName(u"frame_47")
        self.frame_47.setFrameShape(QFrame.StyledPanel)
        self.frame_47.setFrameShadow(QFrame.Raised)
        self.verticalLayout_39 = QVBoxLayout(self.frame_47)
        self.verticalLayout_39.setSpacing(1)
        self.verticalLayout_39.setObjectName(u"verticalLayout_39")
        self.verticalLayout_39.setContentsMargins(1, 1, 1, 1)
        self.label_22 = QLabel(self.frame_47)
        self.label_22.setObjectName(u"label_22")
        sizePolicy2.setHeightForWidth(self.label_22.sizePolicy().hasHeightForWidth())
        self.label_22.setSizePolicy(sizePolicy2)
        self.label_22.setFont(font2)
        self.label_22.setAlignment(Qt.AlignCenter)

        self.verticalLayout_39.addWidget(self.label_22)

        self.loss_d_graph_frame = QFrame(self.frame_47)
        self.loss_d_graph_frame.setObjectName(u"loss_d_graph_frame")
        self.loss_d_graph_frame.setFrameShape(QFrame.StyledPanel)
        self.loss_d_graph_frame.setFrameShadow(QFrame.Raised)
        self.loss_d_graph_layout = QVBoxLayout(self.loss_d_graph_frame)
        self.loss_d_graph_layout.setObjectName(u"loss_d_graph_layout")
        self.loss_d_graph_layout.setContentsMargins(0, 0, 0, 0)

        self.verticalLayout_39.addWidget(self.loss_d_graph_frame)

        self.splitter_2.addWidget(self.frame_47)

        self.verticalLayout_34.addWidget(self.splitter_2)

        self.visualizer_splitter.addWidget(self.frame_14)

        self.verticalLayout_6.addWidget(self.visualizer_splitter)


        self.verticalLayout_8.addWidget(self.frame_16)


        self.horizontalLayout.addWidget(self.frame_7)


        self.verticalLayout.addWidget(self.frame)

        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.utility_dock = QDockWidget(MainWindow)
        self.utility_dock.setObjectName(u"utility_dock")
        self.utility_dock.setFloating(False)
        self.utility_dock.setFeatures(QDockWidget.NoDockWidgetFeatures)
        self.utility_dock.setAllowedAreas(Qt.BottomDockWidgetArea)
        self.dockWidgetContents_2 = QWidget()
        self.dockWidgetContents_2.setObjectName(u"dockWidgetContents_2")
        sizePolicy1.setHeightForWidth(self.dockWidgetContents_2.sizePolicy().hasHeightForWidth())
        self.dockWidgetContents_2.setSizePolicy(sizePolicy1)
        self.verticalLayout_7 = QVBoxLayout(self.dockWidgetContents_2)
        self.verticalLayout_7.setSpacing(0)
        self.verticalLayout_7.setObjectName(u"verticalLayout_7")
        self.verticalLayout_7.setContentsMargins(0, 0, 0, 0)
        self.frame_24 = QFrame(self.dockWidgetContents_2)
        self.frame_24.setObjectName(u"frame_24")
        sizePolicy5 = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Ignored)
        sizePolicy5.setHorizontalStretch(0)
        sizePolicy5.setVerticalStretch(0)
        sizePolicy5.setHeightForWidth(self.frame_24.sizePolicy().hasHeightForWidth())
        self.frame_24.setSizePolicy(sizePolicy5)
        self.frame_24.setFrameShape(QFrame.StyledPanel)
        self.frame_24.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_32 = QHBoxLayout(self.frame_24)
        self.horizontalLayout_32.setSpacing(0)
        self.horizontalLayout_32.setObjectName(u"horizontalLayout_32")
        self.horizontalLayout_32.setContentsMargins(0, 0, 0, 0)
        self.times_group = QGroupBox(self.frame_24)
        self.times_group.setObjectName(u"times_group")
        sizePolicy6 = QSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Ignored)
        sizePolicy6.setHorizontalStretch(0)
        sizePolicy6.setVerticalStretch(0)
        sizePolicy6.setHeightForWidth(self.times_group.sizePolicy().hasHeightForWidth())
        self.times_group.setSizePolicy(sizePolicy6)
        self.verticalLayout_30 = QVBoxLayout(self.times_group)
        self.verticalLayout_30.setSpacing(2)
        self.verticalLayout_30.setObjectName(u"verticalLayout_30")
        self.verticalLayout_30.setContentsMargins(6, 6, 6, 6)
        self.frame_36 = QFrame(self.times_group)
        self.frame_36.setObjectName(u"frame_36")
        self.frame_36.setFrameShape(QFrame.StyledPanel)
        self.frame_36.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_36 = QHBoxLayout(self.frame_36)
        self.horizontalLayout_36.setObjectName(u"horizontalLayout_36")
        self.horizontalLayout_36.setContentsMargins(0, 0, 0, 0)
        self.label_55 = QLabel(self.frame_36)
        self.label_55.setObjectName(u"label_55")
        sizePolicy7 = QSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
        sizePolicy7.setHorizontalStretch(0)
        sizePolicy7.setVerticalStretch(0)
        sizePolicy7.setHeightForWidth(self.label_55.sizePolicy().hasHeightForWidth())
        self.label_55.setSizePolicy(sizePolicy7)
        font3 = QFont()
        font3.setBold(True)
        self.label_55.setFont(font3)

        self.horizontalLayout_36.addWidget(self.label_55)

        self.line_10 = QFrame(self.frame_36)
        self.line_10.setObjectName(u"line_10")
        self.line_10.setFrameShadow(QFrame.Raised)
        self.line_10.setFrameShape(QFrame.Shape.HLine)

        self.horizontalLayout_36.addWidget(self.line_10)


        self.verticalLayout_30.addWidget(self.frame_36)

        self.frame_37 = QFrame(self.times_group)
        self.frame_37.setObjectName(u"frame_37")
        sizePolicy2.setHeightForWidth(self.frame_37.sizePolicy().hasHeightForWidth())
        self.frame_37.setSizePolicy(sizePolicy2)
        self.frame_37.setFrameShape(QFrame.StyledPanel)
        self.frame_37.setFrameShadow(QFrame.Raised)
        self.formLayout_19 = QFormLayout(self.frame_37)
        self.formLayout_19.setObjectName(u"formLayout_19")
        self.formLayout_19.setHorizontalSpacing(6)
        self.formLayout_19.setVerticalSpacing(2)
        self.formLayout_19.setContentsMargins(0, 0, 0, 0)
        self.performance_time_total = QLabel(self.frame_37)
        self.performance_time_total.setObjectName(u"performance_time_total")

        self.formLayout_19.setWidget(0, QFormLayout.ItemRole.FieldRole, self.performance_time_total)

        self.label_56 = QLabel(self.frame_37)
        self.label_56.setObjectName(u"label_56")
        sizePolicy8 = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        sizePolicy8.setHorizontalStretch(0)
        sizePolicy8.setVerticalStretch(0)
        sizePolicy8.setHeightForWidth(self.label_56.sizePolicy().hasHeightForWidth())
        self.label_56.setSizePolicy(sizePolicy8)

        self.formLayout_19.setWidget(0, QFormLayout.ItemRole.LabelRole, self.label_56)


        self.verticalLayout_30.addWidget(self.frame_37)

        self.frame_34 = QFrame(self.times_group)
        self.frame_34.setObjectName(u"frame_34")
        self.frame_34.setFrameShape(QFrame.StyledPanel)
        self.frame_34.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_34 = QHBoxLayout(self.frame_34)
        self.horizontalLayout_34.setObjectName(u"horizontalLayout_34")
        self.horizontalLayout_34.setContentsMargins(0, 0, 0, 0)
        self.label_48 = QLabel(self.frame_34)
        self.label_48.setObjectName(u"label_48")
        sizePolicy7.setHeightForWidth(self.label_48.sizePolicy().hasHeightForWidth())
        self.label_48.setSizePolicy(sizePolicy7)
        self.label_48.setFont(font3)

        self.horizontalLayout_34.addWidget(self.label_48)

        self.line_8 = QFrame(self.frame_34)
        self.line_8.setObjectName(u"line_8")
        self.line_8.setFrameShadow(QFrame.Raised)
        self.line_8.setFrameShape(QFrame.Shape.HLine)

        self.horizontalLayout_34.addWidget(self.line_8)


        self.verticalLayout_30.addWidget(self.frame_34)

        self.frame_30 = QFrame(self.times_group)
        self.frame_30.setObjectName(u"frame_30")
        sizePolicy2.setHeightForWidth(self.frame_30.sizePolicy().hasHeightForWidth())
        self.frame_30.setSizePolicy(sizePolicy2)
        self.frame_30.setFrameShape(QFrame.StyledPanel)
        self.frame_30.setFrameShadow(QFrame.Raised)
        self.formLayout_17 = QFormLayout(self.frame_30)
        self.formLayout_17.setObjectName(u"formLayout_17")
        self.formLayout_17.setHorizontalSpacing(6)
        self.formLayout_17.setVerticalSpacing(2)
        self.formLayout_17.setContentsMargins(0, 0, 0, 0)
        self.label_47 = QLabel(self.frame_30)
        self.label_47.setObjectName(u"label_47")
        sizePolicy8.setHeightForWidth(self.label_47.sizePolicy().hasHeightForWidth())
        self.label_47.setSizePolicy(sizePolicy8)

        self.formLayout_17.setWidget(0, QFormLayout.ItemRole.LabelRole, self.label_47)

        self.performance_epoch_fastest = QLabel(self.frame_30)
        self.performance_epoch_fastest.setObjectName(u"performance_epoch_fastest")

        self.formLayout_17.setWidget(0, QFormLayout.ItemRole.FieldRole, self.performance_epoch_fastest)

        self.label_50 = QLabel(self.frame_30)
        self.label_50.setObjectName(u"label_50")
        sizePolicy8.setHeightForWidth(self.label_50.sizePolicy().hasHeightForWidth())
        self.label_50.setSizePolicy(sizePolicy8)

        self.formLayout_17.setWidget(1, QFormLayout.ItemRole.LabelRole, self.label_50)

        self.performance_epoch_slowest = QLabel(self.frame_30)
        self.performance_epoch_slowest.setObjectName(u"performance_epoch_slowest")

        self.formLayout_17.setWidget(1, QFormLayout.ItemRole.FieldRole, self.performance_epoch_slowest)

        self.label_51 = QLabel(self.frame_30)
        self.label_51.setObjectName(u"label_51")
        sizePolicy8.setHeightForWidth(self.label_51.sizePolicy().hasHeightForWidth())
        self.label_51.setSizePolicy(sizePolicy8)

        self.formLayout_17.setWidget(2, QFormLayout.ItemRole.LabelRole, self.label_51)

        self.performance_epoch_average = QLabel(self.frame_30)
        self.performance_epoch_average.setObjectName(u"performance_epoch_average")

        self.formLayout_17.setWidget(2, QFormLayout.ItemRole.FieldRole, self.performance_epoch_average)


        self.verticalLayout_30.addWidget(self.frame_30)

        self.frame_35 = QFrame(self.times_group)
        self.frame_35.setObjectName(u"frame_35")
        self.frame_35.setFrameShape(QFrame.StyledPanel)
        self.frame_35.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_35 = QHBoxLayout(self.frame_35)
        self.horizontalLayout_35.setObjectName(u"horizontalLayout_35")
        self.horizontalLayout_35.setContentsMargins(0, 0, 0, 0)
        self.label_49 = QLabel(self.frame_35)
        self.label_49.setObjectName(u"label_49")
        sizePolicy7.setHeightForWidth(self.label_49.sizePolicy().hasHeightForWidth())
        self.label_49.setSizePolicy(sizePolicy7)
        self.label_49.setFont(font3)

        self.horizontalLayout_35.addWidget(self.label_49)

        self.line_9 = QFrame(self.frame_35)
        self.line_9.setObjectName(u"line_9")
        self.line_9.setFrameShadow(QFrame.Raised)
        self.line_9.setFrameShape(QFrame.Shape.HLine)

        self.horizontalLayout_35.addWidget(self.line_9)


        self.verticalLayout_30.addWidget(self.frame_35)

        self.frame_32 = QFrame(self.times_group)
        self.frame_32.setObjectName(u"frame_32")
        sizePolicy2.setHeightForWidth(self.frame_32.sizePolicy().hasHeightForWidth())
        self.frame_32.setSizePolicy(sizePolicy2)
        self.frame_32.setFrameShape(QFrame.StyledPanel)
        self.frame_32.setFrameShadow(QFrame.Raised)
        self.formLayout_18 = QFormLayout(self.frame_32)
        self.formLayout_18.setObjectName(u"formLayout_18")
        self.formLayout_18.setHorizontalSpacing(6)
        self.formLayout_18.setVerticalSpacing(2)
        self.formLayout_18.setContentsMargins(0, 0, 0, 0)
        self.label_52 = QLabel(self.frame_32)
        self.label_52.setObjectName(u"label_52")
        sizePolicy8.setHeightForWidth(self.label_52.sizePolicy().hasHeightForWidth())
        self.label_52.setSizePolicy(sizePolicy8)

        self.formLayout_18.setWidget(0, QFormLayout.ItemRole.LabelRole, self.label_52)

        self.performance_iter_fastest = QLabel(self.frame_32)
        self.performance_iter_fastest.setObjectName(u"performance_iter_fastest")

        self.formLayout_18.setWidget(0, QFormLayout.ItemRole.FieldRole, self.performance_iter_fastest)

        self.label_53 = QLabel(self.frame_32)
        self.label_53.setObjectName(u"label_53")
        sizePolicy8.setHeightForWidth(self.label_53.sizePolicy().hasHeightForWidth())
        self.label_53.setSizePolicy(sizePolicy8)

        self.formLayout_18.setWidget(1, QFormLayout.ItemRole.LabelRole, self.label_53)

        self.performance_iter_slowest = QLabel(self.frame_32)
        self.performance_iter_slowest.setObjectName(u"performance_iter_slowest")

        self.formLayout_18.setWidget(1, QFormLayout.ItemRole.FieldRole, self.performance_iter_slowest)

        self.label_54 = QLabel(self.frame_32)
        self.label_54.setObjectName(u"label_54")
        sizePolicy8.setHeightForWidth(self.label_54.sizePolicy().hasHeightForWidth())
        self.label_54.setSizePolicy(sizePolicy8)

        self.formLayout_18.setWidget(2, QFormLayout.ItemRole.LabelRole, self.label_54)

        self.performance_iter_average = QLabel(self.frame_32)
        self.performance_iter_average.setObjectName(u"performance_iter_average")

        self.formLayout_18.setWidget(2, QFormLayout.ItemRole.FieldRole, self.performance_iter_average)


        self.verticalLayout_30.addWidget(self.frame_32)

        self.verticalSpacer_7 = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.verticalLayout_30.addItem(self.verticalSpacer_7)


        self.horizontalLayout_32.addWidget(self.times_group)

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
        sizePolicy2.setHeightForWidth(self.perf_graph_epoch_label.sizePolicy().hasHeightForWidth())
        self.perf_graph_epoch_label.setSizePolicy(sizePolicy2)
        self.perf_graph_epoch_label.setAlignment(Qt.AlignCenter)

        self.verticalLayout_29.addWidget(self.perf_graph_epoch_label)


        self.verticalLayout_26.addWidget(self.frame_33)

        self.frame_26 = QFrame(self.performance_monitor_group)
        self.frame_26.setObjectName(u"frame_26")
        sizePolicy2.setHeightForWidth(self.frame_26.sizePolicy().hasHeightForWidth())
        self.frame_26.setSizePolicy(sizePolicy2)
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

        self.output_log_group = QGroupBox(self.frame_24)
        self.output_log_group.setObjectName(u"output_log_group")
        sizePolicy5.setHeightForWidth(self.output_log_group.sizePolicy().hasHeightForWidth())
        self.output_log_group.setSizePolicy(sizePolicy5)
        self.verticalLayout_9 = QVBoxLayout(self.output_log_group)
        self.verticalLayout_9.setSpacing(2)
        self.verticalLayout_9.setObjectName(u"verticalLayout_9")
        self.verticalLayout_9.setContentsMargins(3, 3, 3, 3)
        self.log_output_frozen_header = QLabel(self.output_log_group)
        self.log_output_frozen_header.setObjectName(u"log_output_frozen_header")
        font4 = QFont()
        font4.setBold(False)
        font4.setItalic(True)
        self.log_output_frozen_header.setFont(font4)
        self.log_output_frozen_header.setAlignment(Qt.AlignCenter)

        self.verticalLayout_9.addWidget(self.log_output_frozen_header)

        self.frame_9 = QFrame(self.output_log_group)
        self.frame_9.setObjectName(u"frame_9")
        self.frame_9.setFrameShape(QFrame.StyledPanel)
        self.frame_9.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_29 = QHBoxLayout(self.frame_9)
        self.horizontalLayout_29.setObjectName(u"horizontalLayout_29")
        self.horizontalLayout_29.setContentsMargins(0, 0, 0, 0)
        self.output_log = QTextEdit(self.frame_9)
        self.output_log.setObjectName(u"output_log")
        sizePolicy9 = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        sizePolicy9.setHorizontalStretch(0)
        sizePolicy9.setVerticalStretch(0)
        sizePolicy9.setHeightForWidth(self.output_log.sizePolicy().hasHeightForWidth())
        self.output_log.setSizePolicy(sizePolicy9)
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

        self.line_7 = QFrame(self.frame_15)
        self.line_7.setObjectName(u"line_7")
        self.line_7.setFrameShadow(QFrame.Raised)
        self.line_7.setFrameShape(QFrame.Shape.HLine)

        self.verticalLayout_23.addWidget(self.line_7)

        self.autoscroll_log = QCheckBox(self.frame_15)
        self.autoscroll_log.setObjectName(u"autoscroll_log")
        sizePolicy7.setHeightForWidth(self.autoscroll_log.sizePolicy().hasHeightForWidth())
        self.autoscroll_log.setSizePolicy(sizePolicy7)
        self.autoscroll_log.setChecked(True)

        self.verticalLayout_23.addWidget(self.autoscroll_log)

        self.verticalSpacer_6 = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.verticalLayout_23.addItem(self.verticalSpacer_6)


        self.horizontalLayout_29.addWidget(self.frame_15)


        self.verticalLayout_9.addWidget(self.frame_9)


        self.horizontalLayout_32.addWidget(self.output_log_group)


        self.verticalLayout_7.addWidget(self.frame_24)

        self.frame_10 = QFrame(self.dockWidgetContents_2)
        self.frame_10.setObjectName(u"frame_10")
        sizePolicy2.setHeightForWidth(self.frame_10.sizePolicy().hasHeightForWidth())
        self.frame_10.setSizePolicy(sizePolicy2)
        self.frame_10.setFrameShape(QFrame.Box)
        self.frame_10.setFrameShadow(QFrame.Sunken)
        self.horizontalLayout_12 = QHBoxLayout(self.frame_10)
        self.horizontalLayout_12.setSpacing(3)
        self.horizontalLayout_12.setObjectName(u"horizontalLayout_12")
        self.horizontalLayout_12.setContentsMargins(0, 0, 0, 0)
        self.frame_28 = QFrame(self.frame_10)
        self.frame_28.setObjectName(u"frame_28")
        sizePolicy10 = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        sizePolicy10.setHorizontalStretch(2)
        sizePolicy10.setVerticalStretch(0)
        sizePolicy10.setHeightForWidth(self.frame_28.sizePolicy().hasHeightForWidth())
        self.frame_28.setSizePolicy(sizePolicy10)
        self.frame_28.setFrameShape(QFrame.StyledPanel)
        self.frame_28.setFrameShadow(QFrame.Raised)
        self.verticalLayout_27 = QVBoxLayout(self.frame_28)
        self.verticalLayout_27.setSpacing(1)
        self.verticalLayout_27.setObjectName(u"verticalLayout_27")
        self.verticalLayout_27.setContentsMargins(0, 0, 0, 0)
        self.epoch_progress = QProgressBar(self.frame_28)
        self.epoch_progress.setObjectName(u"epoch_progress")
        sizePolicy11 = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        sizePolicy11.setHorizontalStretch(0)
        sizePolicy11.setVerticalStretch(0)
        sizePolicy11.setHeightForWidth(self.epoch_progress.sizePolicy().hasHeightForWidth())
        self.epoch_progress.setSizePolicy(sizePolicy11)
        self.epoch_progress.setValue(0)
        self.epoch_progress.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)
        self.epoch_progress.setOrientation(Qt.Horizontal)
        self.epoch_progress.setInvertedAppearance(False)
        self.epoch_progress.setTextDirection(QProgressBar.TopToBottom)

        self.verticalLayout_27.addWidget(self.epoch_progress)

        self.label_45 = QLabel(self.frame_28)
        self.label_45.setObjectName(u"label_45")
        font5 = QFont()
        font5.setPointSize(7)
        self.label_45.setFont(font5)
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
        sizePolicy11.setHeightForWidth(self.train_progress.sizePolicy().hasHeightForWidth())
        self.train_progress.setSizePolicy(sizePolicy11)
        self.train_progress.setValue(0)
        self.train_progress.setAlignment(Qt.AlignBottom|Qt.AlignRight|Qt.AlignTrailing)

        self.verticalLayout_28.addWidget(self.train_progress)

        self.label_46 = QLabel(self.frame_29)
        self.label_46.setObjectName(u"label_46")
        self.label_46.setFont(font5)
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
        font6 = QFont()
        font6.setPointSize(7)
        font6.setBold(True)
        self.label_41.setFont(font6)
        self.label_41.setAlignment(Qt.AlignCenter)

        self.verticalLayout_31.addWidget(self.label_41)


        self.horizontalLayout_12.addWidget(self.frame_22)


        self.verticalLayout_7.addWidget(self.frame_10)

        self.utility_dock.setWidget(self.dockWidgetContents_2)
        MainWindow.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.utility_dock)
        self.settings_dock = QDockWidget(MainWindow)
        self.settings_dock.setObjectName(u"settings_dock")
        self.settings_dock.setFeatures(QDockWidget.NoDockWidgetFeatures)
        self.settings_dock.setAllowedAreas(Qt.NoDockWidgetArea)
        self.dockWidgetContents_8 = QWidget()
        self.dockWidgetContents_8.setObjectName(u"dockWidgetContents_8")
        self.horizontalLayout_5 = QHBoxLayout(self.dockWidgetContents_8)
        self.horizontalLayout_5.setSpacing(0)
        self.horizontalLayout_5.setObjectName(u"horizontalLayout_5")
        self.horizontalLayout_5.setContentsMargins(0, 0, 0, 0)
        self.settings_btns_frame = QFrame(self.dockWidgetContents_8)
        self.settings_btns_frame.setObjectName(u"settings_btns_frame")
        sizePolicy.setHeightForWidth(self.settings_btns_frame.sizePolicy().hasHeightForWidth())
        self.settings_btns_frame.setSizePolicy(sizePolicy)
        self.settings_btns_frame.setFrameShape(QFrame.Panel)
        self.settings_btns_frame.setFrameShadow(QFrame.Raised)
        self.settings_btns_frame.setLineWidth(1)
        self.verticalLayout_4 = QVBoxLayout(self.settings_btns_frame)
        self.verticalLayout_4.setSpacing(12)
        self.verticalLayout_4.setObjectName(u"verticalLayout_4")
        self.verticalLayout_4.setContentsMargins(4, 4, 4, 4)
        self.settings_dock_main_icon = QLabel(self.settings_btns_frame)
        self.settings_dock_main_icon.setObjectName(u"settings_dock_main_icon")
        sizePolicy1.setHeightForWidth(self.settings_dock_main_icon.sizePolicy().hasHeightForWidth())
        self.settings_dock_main_icon.setSizePolicy(sizePolicy1)
        self.settings_dock_main_icon.setMinimumSize(QSize(50, 50))
        self.settings_dock_main_icon.setMaximumSize(QSize(100, 100))
        self.settings_dock_main_icon.setFont(font3)
        self.settings_dock_main_icon.setScaledContents(False)
        self.settings_dock_main_icon.setAlignment(Qt.AlignLeading|Qt.AlignLeft|Qt.AlignVCenter)

        self.verticalLayout_4.addWidget(self.settings_dock_main_icon)

        self.verticalSpacer_12 = QSpacerItem(20, 2000, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.verticalLayout_4.addItem(self.verticalSpacer_12)

        self.frame_3 = QFrame(self.settings_btns_frame)
        self.frame_3.setObjectName(u"frame_3")
        sizePolicy2.setHeightForWidth(self.frame_3.sizePolicy().hasHeightForWidth())
        self.frame_3.setSizePolicy(sizePolicy2)
        self.frame_3.setFrameShape(QFrame.StyledPanel)
        self.frame_3.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_6 = QHBoxLayout(self.frame_3)
        self.horizontalLayout_6.setSpacing(0)
        self.horizontalLayout_6.setObjectName(u"horizontalLayout_6")
        self.horizontalLayout_6.setContentsMargins(0, 0, 0, 0)
        self.experiment_settings_btn = QPushButton(self.frame_3)
        self.experiment_settings_btn.setObjectName(u"experiment_settings_btn")
        self.experiment_settings_btn.setEnabled(True)
        sizePolicy13 = QSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        sizePolicy13.setHorizontalStretch(0)
        sizePolicy13.setVerticalStretch(0)
        sizePolicy13.setHeightForWidth(self.experiment_settings_btn.sizePolicy().hasHeightForWidth())
        self.experiment_settings_btn.setSizePolicy(sizePolicy13)
        self.experiment_settings_btn.setMinimumSize(QSize(50, 50))
        self.experiment_settings_btn.setMaximumSize(QSize(50, 50))
        self.experiment_settings_btn.setBaseSize(QSize(0, 0))
        self.experiment_settings_btn.setIconSize(QSize(35, 35))
        self.experiment_settings_btn.setCheckable(True)
        self.experiment_settings_btn.setChecked(True)

        self.horizontalLayout_6.addWidget(self.experiment_settings_btn)

        self.experiment_settings_label = QPushButton(self.frame_3)
        self.experiment_settings_label.setObjectName(u"experiment_settings_label")
        sizePolicy14 = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        sizePolicy14.setHorizontalStretch(0)
        sizePolicy14.setVerticalStretch(0)
        sizePolicy14.setHeightForWidth(self.experiment_settings_label.sizePolicy().hasHeightForWidth())
        self.experiment_settings_label.setSizePolicy(sizePolicy14)
        font7 = QFont()
        font7.setPointSize(16)
        self.experiment_settings_label.setFont(font7)
        self.experiment_settings_label.setFlat(True)

        self.horizontalLayout_6.addWidget(self.experiment_settings_label)

        self.horizontalSpacer_7 = QSpacerItem(0, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_6.addItem(self.horizontalSpacer_7)


        self.verticalLayout_4.addWidget(self.frame_3)

        self.frame_54 = QFrame(self.settings_btns_frame)
        self.frame_54.setObjectName(u"frame_54")
        sizePolicy2.setHeightForWidth(self.frame_54.sizePolicy().hasHeightForWidth())
        self.frame_54.setSizePolicy(sizePolicy2)
        self.frame_54.setFrameShape(QFrame.StyledPanel)
        self.frame_54.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_8 = QHBoxLayout(self.frame_54)
        self.horizontalLayout_8.setSpacing(0)
        self.horizontalLayout_8.setObjectName(u"horizontalLayout_8")
        self.horizontalLayout_8.setContentsMargins(0, 0, 0, 0)
        self.dataset_settings_btn = QPushButton(self.frame_54)
        self.dataset_settings_btn.setObjectName(u"dataset_settings_btn")
        self.dataset_settings_btn.setEnabled(True)
        sizePolicy13.setHeightForWidth(self.dataset_settings_btn.sizePolicy().hasHeightForWidth())
        self.dataset_settings_btn.setSizePolicy(sizePolicy13)
        self.dataset_settings_btn.setMinimumSize(QSize(50, 50))
        self.dataset_settings_btn.setMaximumSize(QSize(50, 50))
        self.dataset_settings_btn.setBaseSize(QSize(0, 0))
        self.dataset_settings_btn.setIconSize(QSize(35, 35))
        self.dataset_settings_btn.setCheckable(True)

        self.horizontalLayout_8.addWidget(self.dataset_settings_btn)

        self.dataset_settings_label = QPushButton(self.frame_54)
        self.dataset_settings_label.setObjectName(u"dataset_settings_label")
        sizePolicy14.setHeightForWidth(self.dataset_settings_label.sizePolicy().hasHeightForWidth())
        self.dataset_settings_label.setSizePolicy(sizePolicy14)
        self.dataset_settings_label.setFont(font7)
        self.dataset_settings_label.setFlat(True)

        self.horizontalLayout_8.addWidget(self.dataset_settings_label)

        self.horizontalSpacer_8 = QSpacerItem(0, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_8.addItem(self.horizontalSpacer_8)


        self.verticalLayout_4.addWidget(self.frame_54)

        self.frame_55 = QFrame(self.settings_btns_frame)
        self.frame_55.setObjectName(u"frame_55")
        sizePolicy2.setHeightForWidth(self.frame_55.sizePolicy().hasHeightForWidth())
        self.frame_55.setSizePolicy(sizePolicy2)
        self.frame_55.setFrameShape(QFrame.StyledPanel)
        self.frame_55.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_46 = QHBoxLayout(self.frame_55)
        self.horizontalLayout_46.setSpacing(0)
        self.horizontalLayout_46.setObjectName(u"horizontalLayout_46")
        self.horizontalLayout_46.setContentsMargins(0, 0, 0, 0)
        self.training_settings_btn = QPushButton(self.frame_55)
        self.training_settings_btn.setObjectName(u"training_settings_btn")
        self.training_settings_btn.setEnabled(True)
        sizePolicy13.setHeightForWidth(self.training_settings_btn.sizePolicy().hasHeightForWidth())
        self.training_settings_btn.setSizePolicy(sizePolicy13)
        self.training_settings_btn.setMinimumSize(QSize(50, 50))
        self.training_settings_btn.setMaximumSize(QSize(50, 50))
        self.training_settings_btn.setBaseSize(QSize(0, 0))
        self.training_settings_btn.setIconSize(QSize(35, 35))
        self.training_settings_btn.setCheckable(True)

        self.horizontalLayout_46.addWidget(self.training_settings_btn)

        self.training_settings_label = QPushButton(self.frame_55)
        self.training_settings_label.setObjectName(u"training_settings_label")
        sizePolicy14.setHeightForWidth(self.training_settings_label.sizePolicy().hasHeightForWidth())
        self.training_settings_label.setSizePolicy(sizePolicy14)
        self.training_settings_label.setFont(font7)
        self.training_settings_label.setFlat(True)

        self.horizontalLayout_46.addWidget(self.training_settings_label)

        self.horizontalSpacer_5 = QSpacerItem(0, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_46.addItem(self.horizontalSpacer_5)


        self.verticalLayout_4.addWidget(self.frame_55)

        self.frame_56 = QFrame(self.settings_btns_frame)
        self.frame_56.setObjectName(u"frame_56")
        sizePolicy2.setHeightForWidth(self.frame_56.sizePolicy().hasHeightForWidth())
        self.frame_56.setSizePolicy(sizePolicy2)
        self.frame_56.setFrameShape(QFrame.StyledPanel)
        self.frame_56.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_47 = QHBoxLayout(self.frame_56)
        self.horizontalLayout_47.setSpacing(0)
        self.horizontalLayout_47.setObjectName(u"horizontalLayout_47")
        self.horizontalLayout_47.setContentsMargins(0, 0, 0, 0)
        self.testing_settings_btn = QPushButton(self.frame_56)
        self.testing_settings_btn.setObjectName(u"testing_settings_btn")
        self.testing_settings_btn.setEnabled(True)
        sizePolicy13.setHeightForWidth(self.testing_settings_btn.sizePolicy().hasHeightForWidth())
        self.testing_settings_btn.setSizePolicy(sizePolicy13)
        self.testing_settings_btn.setMinimumSize(QSize(50, 50))
        self.testing_settings_btn.setMaximumSize(QSize(50, 50))
        self.testing_settings_btn.setBaseSize(QSize(0, 0))
        self.testing_settings_btn.setIconSize(QSize(35, 35))
        self.testing_settings_btn.setCheckable(True)

        self.horizontalLayout_47.addWidget(self.testing_settings_btn)

        self.testing_settings_label = QPushButton(self.frame_56)
        self.testing_settings_label.setObjectName(u"testing_settings_label")
        sizePolicy14.setHeightForWidth(self.testing_settings_label.sizePolicy().hasHeightForWidth())
        self.testing_settings_label.setSizePolicy(sizePolicy14)
        self.testing_settings_label.setFont(font7)
        self.testing_settings_label.setFlat(True)

        self.horizontalLayout_47.addWidget(self.testing_settings_label)

        self.horizontalSpacer_9 = QSpacerItem(0, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_47.addItem(self.horizontalSpacer_9)


        self.verticalLayout_4.addWidget(self.frame_56)

        self.frame_59 = QFrame(self.settings_btns_frame)
        self.frame_59.setObjectName(u"frame_59")
        sizePolicy2.setHeightForWidth(self.frame_59.sizePolicy().hasHeightForWidth())
        self.frame_59.setSizePolicy(sizePolicy2)
        self.frame_59.setFrameShape(QFrame.StyledPanel)
        self.frame_59.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_50 = QHBoxLayout(self.frame_59)
        self.horizontalLayout_50.setSpacing(0)
        self.horizontalLayout_50.setObjectName(u"horizontalLayout_50")
        self.horizontalLayout_50.setContentsMargins(0, 0, 0, 0)
        self.review_settings_btn = QPushButton(self.frame_59)
        self.review_settings_btn.setObjectName(u"review_settings_btn")
        self.review_settings_btn.setEnabled(True)
        sizePolicy13.setHeightForWidth(self.review_settings_btn.sizePolicy().hasHeightForWidth())
        self.review_settings_btn.setSizePolicy(sizePolicy13)
        self.review_settings_btn.setMinimumSize(QSize(50, 50))
        self.review_settings_btn.setMaximumSize(QSize(50, 50))
        self.review_settings_btn.setBaseSize(QSize(0, 0))
        self.review_settings_btn.setIconSize(QSize(35, 35))
        self.review_settings_btn.setCheckable(True)

        self.horizontalLayout_50.addWidget(self.review_settings_btn)

        self.review_settings_label = QPushButton(self.frame_59)
        self.review_settings_label.setObjectName(u"review_settings_label")
        sizePolicy14.setHeightForWidth(self.review_settings_label.sizePolicy().hasHeightForWidth())
        self.review_settings_label.setSizePolicy(sizePolicy14)
        self.review_settings_label.setFont(font7)
        self.review_settings_label.setFlat(True)

        self.horizontalLayout_50.addWidget(self.review_settings_label)

        self.horizontalSpacer_10 = QSpacerItem(0, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_50.addItem(self.horizontalSpacer_10)


        self.verticalLayout_4.addWidget(self.frame_59)

        self.frame_57 = QFrame(self.settings_btns_frame)
        self.frame_57.setObjectName(u"frame_57")
        sizePolicy2.setHeightForWidth(self.frame_57.sizePolicy().hasHeightForWidth())
        self.frame_57.setSizePolicy(sizePolicy2)
        self.frame_57.setFrameShape(QFrame.StyledPanel)
        self.frame_57.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_48 = QHBoxLayout(self.frame_57)
        self.horizontalLayout_48.setSpacing(0)
        self.horizontalLayout_48.setObjectName(u"horizontalLayout_48")
        self.horizontalLayout_48.setContentsMargins(0, 0, 0, 0)
        self.utils_settings_btn = QPushButton(self.frame_57)
        self.utils_settings_btn.setObjectName(u"utils_settings_btn")
        self.utils_settings_btn.setEnabled(True)
        sizePolicy13.setHeightForWidth(self.utils_settings_btn.sizePolicy().hasHeightForWidth())
        self.utils_settings_btn.setSizePolicy(sizePolicy13)
        self.utils_settings_btn.setMinimumSize(QSize(50, 50))
        self.utils_settings_btn.setMaximumSize(QSize(50, 50))
        self.utils_settings_btn.setBaseSize(QSize(0, 0))
        self.utils_settings_btn.setIconSize(QSize(35, 35))
        self.utils_settings_btn.setCheckable(True)

        self.horizontalLayout_48.addWidget(self.utils_settings_btn)

        self.utils_settings_label = QPushButton(self.frame_57)
        self.utils_settings_label.setObjectName(u"utils_settings_label")
        sizePolicy14.setHeightForWidth(self.utils_settings_label.sizePolicy().hasHeightForWidth())
        self.utils_settings_label.setSizePolicy(sizePolicy14)
        self.utils_settings_label.setFont(font7)
        self.utils_settings_label.setFlat(True)

        self.horizontalLayout_48.addWidget(self.utils_settings_label)

        self.horizontalSpacer_11 = QSpacerItem(0, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_48.addItem(self.horizontalSpacer_11)


        self.verticalLayout_4.addWidget(self.frame_57)

        self.frame_58 = QFrame(self.settings_btns_frame)
        self.frame_58.setObjectName(u"frame_58")
        sizePolicy2.setHeightForWidth(self.frame_58.sizePolicy().hasHeightForWidth())
        self.frame_58.setSizePolicy(sizePolicy2)
        self.frame_58.setFrameShape(QFrame.StyledPanel)
        self.frame_58.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_49 = QHBoxLayout(self.frame_58)
        self.horizontalLayout_49.setSpacing(0)
        self.horizontalLayout_49.setObjectName(u"horizontalLayout_49")
        self.horizontalLayout_49.setContentsMargins(0, 0, 0, 0)
        self.settings_settings_btn = QPushButton(self.frame_58)
        self.settings_settings_btn.setObjectName(u"settings_settings_btn")
        self.settings_settings_btn.setEnabled(True)
        sizePolicy13.setHeightForWidth(self.settings_settings_btn.sizePolicy().hasHeightForWidth())
        self.settings_settings_btn.setSizePolicy(sizePolicy13)
        self.settings_settings_btn.setMinimumSize(QSize(50, 50))
        self.settings_settings_btn.setMaximumSize(QSize(50, 50))
        self.settings_settings_btn.setBaseSize(QSize(0, 0))
        self.settings_settings_btn.setIconSize(QSize(35, 35))
        self.settings_settings_btn.setCheckable(True)

        self.horizontalLayout_49.addWidget(self.settings_settings_btn)

        self.settings_settings_label = QPushButton(self.frame_58)
        self.settings_settings_label.setObjectName(u"settings_settings_label")
        sizePolicy14.setHeightForWidth(self.settings_settings_label.sizePolicy().hasHeightForWidth())
        self.settings_settings_label.setSizePolicy(sizePolicy14)
        self.settings_settings_label.setFont(font7)
        self.settings_settings_label.setFlat(True)

        self.horizontalLayout_49.addWidget(self.settings_settings_label)

        self.horizontalSpacer_12 = QSpacerItem(0, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_49.addItem(self.horizontalSpacer_12)


        self.verticalLayout_4.addWidget(self.frame_58)

        self.verticalSpacer_2 = QSpacerItem(20, 2000, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.verticalLayout_4.addItem(self.verticalSpacer_2)

        self.frame_49 = QFrame(self.settings_btns_frame)
        self.frame_49.setObjectName(u"frame_49")
        self.frame_49.setFrameShape(QFrame.StyledPanel)
        self.frame_49.setFrameShadow(QFrame.Raised)
        self.verticalLayout_40 = QVBoxLayout(self.frame_49)
        self.verticalLayout_40.setSpacing(1)
        self.verticalLayout_40.setObjectName(u"verticalLayout_40")
        self.verticalLayout_40.setContentsMargins(0, 0, 0, 0)
        self.title_tag = QLabel(self.frame_49)
        self.title_tag.setObjectName(u"title_tag")
        font8 = QFont()
        font8.setPointSize(8)
        self.title_tag.setFont(font8)
        self.title_tag.setAlignment(Qt.AlignCenter)

        self.verticalLayout_40.addWidget(self.title_tag)

        self.creator_tag = QLabel(self.frame_49)
        self.creator_tag.setObjectName(u"creator_tag")
        sizePolicy2.setHeightForWidth(self.creator_tag.sizePolicy().hasHeightForWidth())
        self.creator_tag.setSizePolicy(sizePolicy2)
        font9 = QFont()
        font9.setPointSize(6)
        self.creator_tag.setFont(font9)
        self.creator_tag.setAlignment(Qt.AlignCenter)

        self.verticalLayout_40.addWidget(self.creator_tag)


        self.verticalLayout_4.addWidget(self.frame_49)


        self.horizontalLayout_5.addWidget(self.settings_btns_frame, 0, Qt.AlignLeft)

        self.frame_2 = QFrame(self.dockWidgetContents_8)
        self.frame_2.setObjectName(u"frame_2")
        self.frame_2.setFrameShape(QFrame.NoFrame)
        self.frame_2.setFrameShadow(QFrame.Raised)
        self.frame_2.setLineWidth(1)
        self.verticalLayout_2 = QVBoxLayout(self.frame_2)
        self.verticalLayout_2.setSpacing(0)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.settings_section_header_frame = QFrame(self.frame_2)
        self.settings_section_header_frame.setObjectName(u"settings_section_header_frame")
        self.settings_section_header_frame.setFrameShape(QFrame.StyledPanel)
        self.settings_section_header_frame.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_20 = QHBoxLayout(self.settings_section_header_frame)
        self.horizontalLayout_20.setSpacing(3)
        self.horizontalLayout_20.setObjectName(u"horizontalLayout_20")
        self.horizontalLayout_20.setContentsMargins(3, 3, 3, 3)
        self.settings_section_header_icon = QLabel(self.settings_section_header_frame)
        self.settings_section_header_icon.setObjectName(u"settings_section_header_icon")
        sizePolicy.setHeightForWidth(self.settings_section_header_icon.sizePolicy().hasHeightForWidth())
        self.settings_section_header_icon.setSizePolicy(sizePolicy)

        self.horizontalLayout_20.addWidget(self.settings_section_header_icon)

        self.settings_section_header = QLabel(self.settings_section_header_frame)
        self.settings_section_header.setObjectName(u"settings_section_header")
        self.settings_section_header.setMinimumSize(QSize(0, 80))
        font10 = QFont()
        font10.setPointSize(22)
        font10.setBold(False)
        font10.setItalic(False)
        self.settings_section_header.setFont(font10)

        self.horizontalLayout_20.addWidget(self.settings_section_header)


        self.verticalLayout_2.addWidget(self.settings_section_header_frame)

        self.train_settings = QFrame(self.frame_2)
        self.train_settings.setObjectName(u"train_settings")
        self.train_settings.setFrameShape(QFrame.NoFrame)
        self.train_settings.setFrameShadow(QFrame.Raised)
        self.train_settings.setLineWidth(1)
        self.verticalLayout_13 = QVBoxLayout(self.train_settings)
        self.verticalLayout_13.setObjectName(u"verticalLayout_13")
        self.verticalLayout_13.setContentsMargins(5, 5, 5, 5)
        self.settings_pages = QStackedWidget(self.train_settings)
        self.settings_pages.setObjectName(u"settings_pages")
        self.experiment_page = QWidget()
        self.experiment_page.setObjectName(u"experiment_page")
        self.verticalLayout_3 = QVBoxLayout(self.experiment_page)
        self.verticalLayout_3.setObjectName(u"verticalLayout_3")
        self.verticalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.scrollArea = QScrollArea(self.experiment_page)
        self.scrollArea.setObjectName(u"scrollArea")
        self.scrollArea.setFrameShadow(QFrame.Raised)
        self.scrollArea.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scrollArea.setWidgetResizable(True)
        self.experiment_settings_container = QWidget()
        self.experiment_settings_container.setObjectName(u"experiment_settings_container")
        self.experiment_settings_container.setGeometry(QRect(0, 0, 331, 544))
        self.verticalLayout_11 = QVBoxLayout(self.experiment_settings_container)
        self.verticalLayout_11.setSpacing(6)
        self.verticalLayout_11.setObjectName(u"verticalLayout_11")
        self.verticalLayout_11.setContentsMargins(3, 3, 3, 3)
        self.frame_8 = QFrame(self.experiment_settings_container)
        self.frame_8.setObjectName(u"frame_8")
        self.frame_8.setFrameShape(QFrame.StyledPanel)
        self.frame_8.setFrameShadow(QFrame.Raised)
        self.formLayout_13 = QFormLayout(self.frame_8)
        self.formLayout_13.setObjectName(u"formLayout_13")
        self.formLayout_13.setContentsMargins(0, 0, 0, 0)
        self.label_2 = QLabel(self.frame_8)
        self.label_2.setObjectName(u"label_2")

        self.formLayout_13.setWidget(0, QFormLayout.ItemRole.LabelRole, self.label_2)

        self.horizontalLayout_3 = QHBoxLayout()
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.output_directory = QLineEdit(self.frame_8)
        self.output_directory.setObjectName(u"output_directory")

        self.horizontalLayout_3.addWidget(self.output_directory)

        self.browse_output = QPushButton(self.frame_8)
        self.browse_output.setObjectName(u"browse_output")

        self.horizontalLayout_3.addWidget(self.browse_output)


        self.formLayout_13.setLayout(0, QFormLayout.ItemRole.FieldRole, self.horizontalLayout_3)

        self.label_3 = QLabel(self.frame_8)
        self.label_3.setObjectName(u"label_3")

        self.formLayout_13.setWidget(1, QFormLayout.ItemRole.LabelRole, self.label_3)

        self.horizontalLayout_4 = QHBoxLayout()
        self.horizontalLayout_4.setObjectName(u"horizontalLayout_4")
        self.experiment_name = QLineEdit(self.frame_8)
        self.experiment_name.setObjectName(u"experiment_name")

        self.horizontalLayout_4.addWidget(self.experiment_name)

        self.label_4 = QLabel(self.frame_8)
        self.label_4.setObjectName(u"label_4")

        self.horizontalLayout_4.addWidget(self.label_4)

        self.experiment_version = QSpinBox(self.frame_8)
        self.experiment_version.setObjectName(u"experiment_version")
        self.experiment_version.setMinimum(1)

        self.horizontalLayout_4.addWidget(self.experiment_version)


        self.formLayout_13.setLayout(1, QFormLayout.ItemRole.FieldRole, self.horizontalLayout_4)


        self.verticalLayout_11.addWidget(self.frame_8)

        self.init_from_last_train = QPushButton(self.experiment_settings_container)
        self.init_from_last_train.setObjectName(u"init_from_last_train")

        self.verticalLayout_11.addWidget(self.init_from_last_train)

        self.verticalSpacer_8 = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.verticalLayout_11.addItem(self.verticalSpacer_8)

        self.scrollArea.setWidget(self.experiment_settings_container)

        self.verticalLayout_3.addWidget(self.scrollArea)

        self.settings_pages.addWidget(self.experiment_page)
        self.dataloader_page = QWidget()
        self.dataloader_page.setObjectName(u"dataloader_page")
        self.verticalLayout_5 = QVBoxLayout(self.dataloader_page)
        self.verticalLayout_5.setObjectName(u"verticalLayout_5")
        self.verticalLayout_5.setContentsMargins(0, 0, 0, 0)
        self.scrollArea_2 = QScrollArea(self.dataloader_page)
        self.scrollArea_2.setObjectName(u"scrollArea_2")
        self.scrollArea_2.setFrameShadow(QFrame.Raised)
        self.scrollArea_2.setWidgetResizable(True)
        self.dataloader_settings_container = QWidget()
        self.dataloader_settings_container.setObjectName(u"dataloader_settings_container")
        self.dataloader_settings_container.setGeometry(QRect(0, 0, 331, 544))
        self.verticalLayout_10 = QVBoxLayout(self.dataloader_settings_container)
        self.verticalLayout_10.setObjectName(u"verticalLayout_10")
        self.verticalLayout_10.setContentsMargins(0, 0, 0, 0)
        self.frame_5 = QFrame(self.dataloader_settings_container)
        self.frame_5.setObjectName(u"frame_5")
        sizePolicy2.setHeightForWidth(self.frame_5.sizePolicy().hasHeightForWidth())
        self.frame_5.setSizePolicy(sizePolicy2)
        self.frame_5.setFrameShape(QFrame.StyledPanel)
        self.frame_5.setFrameShadow(QFrame.Raised)
        self.formLayout_3 = QFormLayout(self.frame_5)
        self.formLayout_3.setObjectName(u"formLayout_3")
        self.formLayout_3.setContentsMargins(0, 0, 0, 0)
        self.label = QLabel(self.frame_5)
        self.label.setObjectName(u"label")

        self.formLayout_3.setWidget(0, QFormLayout.ItemRole.LabelRole, self.label)

        self.horizontalLayout_2 = QHBoxLayout()
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.dataroot = QLineEdit(self.frame_5)
        self.dataroot.setObjectName(u"dataroot")

        self.horizontalLayout_2.addWidget(self.dataroot)

        self.browse_dataset = QPushButton(self.frame_5)
        self.browse_dataset.setObjectName(u"browse_dataset")

        self.horizontalLayout_2.addWidget(self.browse_dataset)


        self.formLayout_3.setLayout(0, QFormLayout.ItemRole.FieldRole, self.horizontalLayout_2)


        self.verticalLayout_10.addWidget(self.frame_5)

        self.frame_6 = QFrame(self.dataloader_settings_container)
        self.frame_6.setObjectName(u"frame_6")
        sizePolicy2.setHeightForWidth(self.frame_6.sizePolicy().hasHeightForWidth())
        self.frame_6.setSizePolicy(sizePolicy2)
        self.frame_6.setFrameShape(QFrame.StyledPanel)
        self.frame_6.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_9 = QHBoxLayout(self.frame_6)
        self.horizontalLayout_9.setObjectName(u"horizontalLayout_9")
        self.horizontalLayout_9.setContentsMargins(0, 0, 0, 0)
        self.frame_13 = QFrame(self.frame_6)
        self.frame_13.setObjectName(u"frame_13")
        self.frame_13.setFrameShape(QFrame.StyledPanel)
        self.frame_13.setFrameShadow(QFrame.Raised)
        self.formLayout_4 = QFormLayout(self.frame_13)
        self.formLayout_4.setObjectName(u"formLayout_4")
        self.formLayout_4.setContentsMargins(0, 0, 0, 0)
        self.label_10 = QLabel(self.frame_13)
        self.label_10.setObjectName(u"label_10")

        self.formLayout_4.setWidget(0, QFormLayout.ItemRole.LabelRole, self.label_10)

        self.load_size = QSpinBox(self.frame_13)
        self.load_size.setObjectName(u"load_size")
        sizePolicy11.setHeightForWidth(self.load_size.sizePolicy().hasHeightForWidth())
        self.load_size.setSizePolicy(sizePolicy11)
        self.load_size.setMinimum(64)
        self.load_size.setMaximum(2048)
        self.load_size.setSingleStep(64)
        self.load_size.setValue(256)

        self.formLayout_4.setWidget(0, QFormLayout.ItemRole.FieldRole, self.load_size)


        self.horizontalLayout_9.addWidget(self.frame_13)

        self.frame_18 = QFrame(self.frame_6)
        self.frame_18.setObjectName(u"frame_18")
        self.frame_18.setFrameShape(QFrame.StyledPanel)
        self.frame_18.setFrameShadow(QFrame.Raised)
        self.formLayout_5 = QFormLayout(self.frame_18)
        self.formLayout_5.setObjectName(u"formLayout_5")
        self.formLayout_5.setContentsMargins(0, 0, 0, 0)
        self.label_11 = QLabel(self.frame_18)
        self.label_11.setObjectName(u"label_11")

        self.formLayout_5.setWidget(0, QFormLayout.ItemRole.LabelRole, self.label_11)

        self.crop_size = QSpinBox(self.frame_18)
        self.crop_size.setObjectName(u"crop_size")
        sizePolicy11.setHeightForWidth(self.crop_size.sizePolicy().hasHeightForWidth())
        self.crop_size.setSizePolicy(sizePolicy11)
        self.crop_size.setMinimum(64)
        self.crop_size.setMaximum(1024)
        self.crop_size.setSingleStep(64)
        self.crop_size.setValue(256)

        self.formLayout_5.setWidget(0, QFormLayout.ItemRole.FieldRole, self.crop_size)


        self.horizontalLayout_9.addWidget(self.frame_18)

        self.label_9 = QLabel(self.frame_6)
        self.label_9.setObjectName(u"label_9")
        sizePolicy1.setHeightForWidth(self.label_9.sizePolicy().hasHeightForWidth())
        self.label_9.setSizePolicy(sizePolicy1)

        self.horizontalLayout_9.addWidget(self.label_9)

        self.input_nc = QSpinBox(self.frame_6)
        self.input_nc.setObjectName(u"input_nc")
        sizePolicy8.setHeightForWidth(self.input_nc.sizePolicy().hasHeightForWidth())
        self.input_nc.setSizePolicy(sizePolicy8)
        self.input_nc.setMinimum(1)
        self.input_nc.setMaximum(3)
        self.input_nc.setValue(3)

        self.horizontalLayout_9.addWidget(self.input_nc)


        self.verticalLayout_10.addWidget(self.frame_6)

        self.frame_19 = QFrame(self.dataloader_settings_container)
        self.frame_19.setObjectName(u"frame_19")
        sizePolicy2.setHeightForWidth(self.frame_19.sizePolicy().hasHeightForWidth())
        self.frame_19.setSizePolicy(sizePolicy2)
        self.frame_19.setFrameShape(QFrame.StyledPanel)
        self.frame_19.setFrameShadow(QFrame.Raised)
        self.formLayout_7 = QFormLayout(self.frame_19)
        self.formLayout_7.setObjectName(u"formLayout_7")
        self.formLayout_7.setContentsMargins(0, 0, 0, 0)
        self.label_5 = QLabel(self.frame_19)
        self.label_5.setObjectName(u"label_5")

        self.formLayout_7.setWidget(0, QFormLayout.ItemRole.LabelRole, self.label_5)

        self.direction = QComboBox(self.frame_19)
        self.direction.setObjectName(u"direction")
        self.direction.setEditable(False)
        self.direction.setFrame(True)

        self.formLayout_7.setWidget(0, QFormLayout.ItemRole.FieldRole, self.direction)

        self.label_8 = QLabel(self.frame_19)
        self.label_8.setObjectName(u"label_8")

        self.formLayout_7.setWidget(1, QFormLayout.ItemRole.LabelRole, self.label_8)

        self.horizontalLayout_7 = QHBoxLayout()
        self.horizontalLayout_7.setObjectName(u"horizontalLayout_7")
        self.batch_size_slider = QSlider(self.frame_19)
        self.batch_size_slider.setObjectName(u"batch_size_slider")
        self.batch_size_slider.setMinimum(1)
        self.batch_size_slider.setMaximum(32)
        self.batch_size_slider.setValue(1)
        self.batch_size_slider.setOrientation(Qt.Horizontal)

        self.horizontalLayout_7.addWidget(self.batch_size_slider)

        self.batch_size = QSpinBox(self.frame_19)
        self.batch_size.setObjectName(u"batch_size")
        self.batch_size.setMinimum(1)
        self.batch_size.setMaximum(32)

        self.horizontalLayout_7.addWidget(self.batch_size)


        self.formLayout_7.setLayout(1, QFormLayout.ItemRole.FieldRole, self.horizontalLayout_7)


        self.verticalLayout_10.addWidget(self.frame_19)

        self.verticalSpacer_3 = QSpacerItem(20, 565, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.verticalLayout_10.addItem(self.verticalSpacer_3)

        self.scrollArea_2.setWidget(self.dataloader_settings_container)

        self.verticalLayout_5.addWidget(self.scrollArea_2)

        self.settings_pages.addWidget(self.dataloader_page)
        self.train_page = QWidget()
        self.train_page.setObjectName(u"train_page")
        self.verticalLayout_22 = QVBoxLayout(self.train_page)
        self.verticalLayout_22.setObjectName(u"verticalLayout_22")
        self.verticalLayout_22.setContentsMargins(0, 0, 0, 0)
        self.scrollArea_3 = QScrollArea(self.train_page)
        self.scrollArea_3.setObjectName(u"scrollArea_3")
        self.scrollArea_3.setFrameShadow(QFrame.Raised)
        self.scrollArea_3.setWidgetResizable(True)
        self.train_settings_container = QWidget()
        self.train_settings_container.setObjectName(u"train_settings_container")
        self.train_settings_container.setGeometry(QRect(0, 0, 331, 544))
        self.verticalLayout_20 = QVBoxLayout(self.train_settings_container)
        self.verticalLayout_20.setObjectName(u"verticalLayout_20")
        self.verticalLayout_20.setContentsMargins(0, 0, 0, 0)
        self.continue_train = QGroupBox(self.train_settings_container)
        self.continue_train.setObjectName(u"continue_train")
        sizePolicy2.setHeightForWidth(self.continue_train.sizePolicy().hasHeightForWidth())
        self.continue_train.setSizePolicy(sizePolicy2)
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


        self.verticalLayout_20.addWidget(self.continue_train)

        self.tabWidget_2 = QTabWidget(self.train_settings_container)
        self.tabWidget_2.setObjectName(u"tabWidget_2")
        sizePolicy15 = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        sizePolicy15.setHorizontalStretch(0)
        sizePolicy15.setVerticalStretch(0)
        sizePolicy15.setHeightForWidth(self.tabWidget_2.sizePolicy().hasHeightForWidth())
        self.tabWidget_2.setSizePolicy(sizePolicy15)
        self.tabWidget_2.setTabPosition(QTabWidget.North)
        self.tab = QWidget()
        self.tab.setObjectName(u"tab")
        sizePolicy1.setHeightForWidth(self.tab.sizePolicy().hasHeightForWidth())
        self.tab.setSizePolicy(sizePolicy1)
        self.verticalLayout_12 = QVBoxLayout(self.tab)
        self.verticalLayout_12.setObjectName(u"verticalLayout_12")
        self.verticalLayout_12.setContentsMargins(0, 0, 0, 0)
        self.scrollArea_4 = QScrollArea(self.tab)
        self.scrollArea_4.setObjectName(u"scrollArea_4")
        self.scrollArea_4.setFrameShadow(QFrame.Raised)
        self.scrollArea_4.setWidgetResizable(True)
        self.scrollAreaWidgetContents_4 = QWidget()
        self.scrollAreaWidgetContents_4.setObjectName(u"scrollAreaWidgetContents_4")
        self.scrollAreaWidgetContents_4.setGeometry(QRect(0, 0, 323, 457))
        self.verticalLayout_15 = QVBoxLayout(self.scrollAreaWidgetContents_4)
        self.verticalLayout_15.setObjectName(u"verticalLayout_15")
        self.verticalLayout_15.setContentsMargins(6, 6, 6, 6)
        self.frame_31 = QFrame(self.scrollAreaWidgetContents_4)
        self.frame_31.setObjectName(u"frame_31")
        self.frame_31.setFrameShape(QFrame.StyledPanel)
        self.frame_31.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_38 = QHBoxLayout(self.frame_31)
        self.horizontalLayout_38.setObjectName(u"horizontalLayout_38")
        self.horizontalLayout_38.setContentsMargins(0, 0, 0, 0)
        self.label_57 = QLabel(self.frame_31)
        self.label_57.setObjectName(u"label_57")
        sizePolicy.setHeightForWidth(self.label_57.sizePolicy().hasHeightForWidth())
        self.label_57.setSizePolicy(sizePolicy)
        font11 = QFont()
        font11.setPointSize(11)
        self.label_57.setFont(font11)
        self.label_57.setFrameShape(QFrame.NoFrame)
        self.label_57.setFrameShadow(QFrame.Raised)

        self.horizontalLayout_38.addWidget(self.label_57)

        self.line_11 = QFrame(self.frame_31)
        self.line_11.setObjectName(u"line_11")
        self.line_11.setFrameShadow(QFrame.Raised)
        self.line_11.setFrameShape(QFrame.Shape.HLine)

        self.horizontalLayout_38.addWidget(self.line_11)


        self.verticalLayout_15.addWidget(self.frame_31)

        self.frame_21 = QFrame(self.scrollAreaWidgetContents_4)
        self.frame_21.setObjectName(u"frame_21")
        sizePolicy2.setHeightForWidth(self.frame_21.sizePolicy().hasHeightForWidth())
        self.frame_21.setSizePolicy(sizePolicy2)
        self.frame_21.setFrameShape(QFrame.StyledPanel)
        self.frame_21.setFrameShadow(QFrame.Raised)
        self.formLayout_8 = QFormLayout(self.frame_21)
        self.formLayout_8.setObjectName(u"formLayout_8")
        self.formLayout_8.setContentsMargins(0, 0, 0, 0)
        self.label_32 = QLabel(self.frame_21)
        self.label_32.setObjectName(u"label_32")

        self.formLayout_8.setWidget(0, QFormLayout.ItemRole.LabelRole, self.label_32)

        self.gen_epochs = QSpinBox(self.frame_21)
        self.gen_epochs.setObjectName(u"gen_epochs")
        self.gen_epochs.setMinimum(1)
        self.gen_epochs.setMaximum(999999)
        self.gen_epochs.setValue(100)

        self.formLayout_8.setWidget(0, QFormLayout.ItemRole.FieldRole, self.gen_epochs)

        self.label_31 = QLabel(self.frame_21)
        self.label_31.setObjectName(u"label_31")

        self.formLayout_8.setWidget(1, QFormLayout.ItemRole.LabelRole, self.label_31)

        self.gen_epochs_decay = QSpinBox(self.frame_21)
        self.gen_epochs_decay.setObjectName(u"gen_epochs_decay")
        self.gen_epochs_decay.setMinimum(1)
        self.gen_epochs_decay.setMaximum(999999)
        self.gen_epochs_decay.setValue(100)

        self.formLayout_8.setWidget(1, QFormLayout.ItemRole.FieldRole, self.gen_epochs_decay)


        self.verticalLayout_15.addWidget(self.frame_21)

        self.frame_40 = QFrame(self.scrollAreaWidgetContents_4)
        self.frame_40.setObjectName(u"frame_40")
        self.frame_40.setFrameShape(QFrame.StyledPanel)
        self.frame_40.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_41 = QHBoxLayout(self.frame_40)
        self.horizontalLayout_41.setObjectName(u"horizontalLayout_41")
        self.horizontalLayout_41.setContentsMargins(0, 0, 0, 0)
        self.label_60 = QLabel(self.frame_40)
        self.label_60.setObjectName(u"label_60")
        sizePolicy.setHeightForWidth(self.label_60.sizePolicy().hasHeightForWidth())
        self.label_60.setSizePolicy(sizePolicy)
        self.label_60.setFont(font11)
        self.label_60.setFrameShape(QFrame.NoFrame)
        self.label_60.setFrameShadow(QFrame.Raised)

        self.horizontalLayout_41.addWidget(self.label_60)

        self.line_14 = QFrame(self.frame_40)
        self.line_14.setObjectName(u"line_14")
        self.line_14.setFrameShadow(QFrame.Raised)
        self.line_14.setFrameShape(QFrame.Shape.HLine)

        self.horizontalLayout_41.addWidget(self.line_14)


        self.verticalLayout_15.addWidget(self.frame_40)

        self.frame_41 = QFrame(self.scrollAreaWidgetContents_4)
        self.frame_41.setObjectName(u"frame_41")
        self.frame_41.setFrameShape(QFrame.StyledPanel)
        self.frame_41.setFrameShadow(QFrame.Raised)
        self.formLayout_15 = QFormLayout(self.frame_41)
        self.formLayout_15.setObjectName(u"formLayout_15")
        self.formLayout_15.setContentsMargins(0, 0, 0, 0)
        self.label_7 = QLabel(self.frame_41)
        self.label_7.setObjectName(u"label_7")

        self.formLayout_15.setWidget(0, QFormLayout.ItemRole.LabelRole, self.label_7)

        self.label_33 = QLabel(self.frame_41)
        self.label_33.setObjectName(u"label_33")

        self.formLayout_15.setWidget(1, QFormLayout.ItemRole.LabelRole, self.label_33)

        self.horizontalLayout_22 = QHBoxLayout()
        self.horizontalLayout_22.setObjectName(u"horizontalLayout_22")
        self.gen_lr_initial_slider = QSlider(self.frame_41)
        self.gen_lr_initial_slider.setObjectName(u"gen_lr_initial_slider")
        self.gen_lr_initial_slider.setMaximum(100)
        self.gen_lr_initial_slider.setValue(2)
        self.gen_lr_initial_slider.setOrientation(Qt.Horizontal)

        self.horizontalLayout_22.addWidget(self.gen_lr_initial_slider)

        self.gen_lr_initial = QDoubleSpinBox(self.frame_41)
        self.gen_lr_initial.setObjectName(u"gen_lr_initial")
        self.gen_lr_initial.setDecimals(5)
        self.gen_lr_initial.setMinimum(0.000000000000000)
        self.gen_lr_initial.setMaximum(0.100000000000000)
        self.gen_lr_initial.setSingleStep(0.000100000000000)
        self.gen_lr_initial.setValue(0.000200000000000)

        self.horizontalLayout_22.addWidget(self.gen_lr_initial)


        self.formLayout_15.setLayout(0, QFormLayout.ItemRole.FieldRole, self.horizontalLayout_22)

        self.horizontalLayout_23 = QHBoxLayout()
        self.horizontalLayout_23.setObjectName(u"horizontalLayout_23")
        self.gen_lr_target_slider = QSlider(self.frame_41)
        self.gen_lr_target_slider.setObjectName(u"gen_lr_target_slider")
        self.gen_lr_target_slider.setMaximum(100)
        self.gen_lr_target_slider.setValue(2)
        self.gen_lr_target_slider.setOrientation(Qt.Horizontal)

        self.horizontalLayout_23.addWidget(self.gen_lr_target_slider)

        self.gen_lr_target = QDoubleSpinBox(self.frame_41)
        self.gen_lr_target.setObjectName(u"gen_lr_target")
        self.gen_lr_target.setDecimals(5)
        self.gen_lr_target.setMinimum(0.000000000000000)
        self.gen_lr_target.setMaximum(0.100000000000000)
        self.gen_lr_target.setSingleStep(0.000100000000000)
        self.gen_lr_target.setValue(0.000200000000000)

        self.horizontalLayout_23.addWidget(self.gen_lr_target)


        self.formLayout_15.setLayout(1, QFormLayout.ItemRole.FieldRole, self.horizontalLayout_23)


        self.verticalLayout_15.addWidget(self.frame_41)

        self.frame_39 = QFrame(self.scrollAreaWidgetContents_4)
        self.frame_39.setObjectName(u"frame_39")
        self.frame_39.setFrameShape(QFrame.StyledPanel)
        self.frame_39.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_40 = QHBoxLayout(self.frame_39)
        self.horizontalLayout_40.setObjectName(u"horizontalLayout_40")
        self.horizontalLayout_40.setContentsMargins(0, 0, 0, 0)
        self.label_59 = QLabel(self.frame_39)
        self.label_59.setObjectName(u"label_59")
        sizePolicy.setHeightForWidth(self.label_59.sizePolicy().hasHeightForWidth())
        self.label_59.setSizePolicy(sizePolicy)
        self.label_59.setFont(font11)
        self.label_59.setFrameShape(QFrame.NoFrame)
        self.label_59.setFrameShadow(QFrame.Raised)

        self.horizontalLayout_40.addWidget(self.label_59)

        self.line_13 = QFrame(self.frame_39)
        self.line_13.setObjectName(u"line_13")
        self.line_13.setFrameShadow(QFrame.Raised)
        self.line_13.setFrameShape(QFrame.Shape.HLine)

        self.horizontalLayout_40.addWidget(self.line_13)


        self.verticalLayout_15.addWidget(self.frame_39)

        self.formLayout_12 = QFormLayout()
        self.formLayout_12.setObjectName(u"formLayout_12")
        self.label_14 = QLabel(self.scrollAreaWidgetContents_4)
        self.label_14.setObjectName(u"label_14")

        self.formLayout_12.setWidget(0, QFormLayout.ItemRole.LabelRole, self.label_14)

        self.horizontalLayout_14 = QHBoxLayout()
        self.horizontalLayout_14.setObjectName(u"horizontalLayout_14")
        self.gen_optim_beta1_slider = QSlider(self.scrollAreaWidgetContents_4)
        self.gen_optim_beta1_slider.setObjectName(u"gen_optim_beta1_slider")
        self.gen_optim_beta1_slider.setMaximum(250)
        self.gen_optim_beta1_slider.setOrientation(Qt.Horizontal)

        self.horizontalLayout_14.addWidget(self.gen_optim_beta1_slider)

        self.gen_optim_beta1 = QDoubleSpinBox(self.scrollAreaWidgetContents_4)
        self.gen_optim_beta1.setObjectName(u"gen_optim_beta1")
        self.gen_optim_beta1.setMaximum(2.500000000000000)
        self.gen_optim_beta1.setSingleStep(0.100000000000000)
        self.gen_optim_beta1.setValue(0.500000000000000)

        self.horizontalLayout_14.addWidget(self.gen_optim_beta1)


        self.formLayout_12.setLayout(0, QFormLayout.ItemRole.FieldRole, self.horizontalLayout_14)


        self.verticalLayout_15.addLayout(self.formLayout_12)

        self.frame_38 = QFrame(self.scrollAreaWidgetContents_4)
        self.frame_38.setObjectName(u"frame_38")
        self.frame_38.setFrameShape(QFrame.StyledPanel)
        self.frame_38.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_39 = QHBoxLayout(self.frame_38)
        self.horizontalLayout_39.setObjectName(u"horizontalLayout_39")
        self.horizontalLayout_39.setContentsMargins(0, 0, 0, 0)
        self.label_58 = QLabel(self.frame_38)
        self.label_58.setObjectName(u"label_58")
        sizePolicy.setHeightForWidth(self.label_58.sizePolicy().hasHeightForWidth())
        self.label_58.setSizePolicy(sizePolicy)
        self.label_58.setFont(font11)
        self.label_58.setFrameShape(QFrame.NoFrame)
        self.label_58.setFrameShadow(QFrame.Raised)

        self.horizontalLayout_39.addWidget(self.label_58)

        self.line_12 = QFrame(self.frame_38)
        self.line_12.setObjectName(u"line_12")
        self.line_12.setFrameShadow(QFrame.Raised)
        self.line_12.setFrameShape(QFrame.Shape.HLine)

        self.horizontalLayout_39.addWidget(self.line_12)


        self.verticalLayout_15.addWidget(self.frame_38)

        self.verticalSpacer_9 = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.verticalLayout_15.addItem(self.verticalSpacer_9)

        self.scrollArea_4.setWidget(self.scrollAreaWidgetContents_4)

        self.verticalLayout_12.addWidget(self.scrollArea_4)

        self.tabWidget_2.addTab(self.tab, "")
        self.tab_7 = QWidget()
        self.tab_7.setObjectName(u"tab_7")
        self.verticalLayout_35 = QVBoxLayout(self.tab_7)
        self.verticalLayout_35.setObjectName(u"verticalLayout_35")
        self.verticalLayout_35.setContentsMargins(0, 0, 0, 0)
        self.scrollArea_6 = QScrollArea(self.tab_7)
        self.scrollArea_6.setObjectName(u"scrollArea_6")
        self.scrollArea_6.setWidgetResizable(True)
        self.scrollAreaWidgetContents_6 = QWidget()
        self.scrollAreaWidgetContents_6.setObjectName(u"scrollAreaWidgetContents_6")
        self.scrollAreaWidgetContents_6.setGeometry(QRect(0, 0, 323, 457))
        self.verticalLayout_36 = QVBoxLayout(self.scrollAreaWidgetContents_6)
        self.verticalLayout_36.setObjectName(u"verticalLayout_36")
        self.verticalLayout_36.setContentsMargins(6, 6, 6, 6)
        self.separate_lr_schedules = QCheckBox(self.scrollAreaWidgetContents_6)
        self.separate_lr_schedules.setObjectName(u"separate_lr_schedules")

        self.verticalLayout_36.addWidget(self.separate_lr_schedules)

        self.disc_schedule_box = QFrame(self.scrollAreaWidgetContents_6)
        self.disc_schedule_box.setObjectName(u"disc_schedule_box")
        self.disc_schedule_box.setFrameShape(QFrame.StyledPanel)
        self.disc_schedule_box.setFrameShadow(QFrame.Raised)
        self.verticalLayout_38 = QVBoxLayout(self.disc_schedule_box)
        self.verticalLayout_38.setObjectName(u"verticalLayout_38")
        self.verticalLayout_38.setContentsMargins(0, 0, 0, 0)
        self.frame_42 = QFrame(self.disc_schedule_box)
        self.frame_42.setObjectName(u"frame_42")
        self.frame_42.setFrameShape(QFrame.StyledPanel)
        self.frame_42.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_42 = QHBoxLayout(self.frame_42)
        self.horizontalLayout_42.setObjectName(u"horizontalLayout_42")
        self.horizontalLayout_42.setContentsMargins(0, 0, 0, 0)
        self.label_61 = QLabel(self.frame_42)
        self.label_61.setObjectName(u"label_61")
        sizePolicy.setHeightForWidth(self.label_61.sizePolicy().hasHeightForWidth())
        self.label_61.setSizePolicy(sizePolicy)
        self.label_61.setFont(font11)
        self.label_61.setFrameShape(QFrame.NoFrame)
        self.label_61.setFrameShadow(QFrame.Raised)

        self.horizontalLayout_42.addWidget(self.label_61)

        self.line_15 = QFrame(self.frame_42)
        self.line_15.setObjectName(u"line_15")
        self.line_15.setFrameShadow(QFrame.Raised)
        self.line_15.setFrameShape(QFrame.Shape.HLine)

        self.horizontalLayout_42.addWidget(self.line_15)


        self.verticalLayout_38.addWidget(self.frame_42)

        self.frame_23 = QFrame(self.disc_schedule_box)
        self.frame_23.setObjectName(u"frame_23")
        sizePolicy2.setHeightForWidth(self.frame_23.sizePolicy().hasHeightForWidth())
        self.frame_23.setSizePolicy(sizePolicy2)
        self.frame_23.setFrameShape(QFrame.StyledPanel)
        self.frame_23.setFrameShadow(QFrame.Raised)
        self.formLayout_11 = QFormLayout(self.frame_23)
        self.formLayout_11.setObjectName(u"formLayout_11")
        self.formLayout_11.setContentsMargins(0, 0, 0, 0)
        self.label_34 = QLabel(self.frame_23)
        self.label_34.setObjectName(u"label_34")

        self.formLayout_11.setWidget(0, QFormLayout.ItemRole.LabelRole, self.label_34)

        self.disc_epochs = QSpinBox(self.frame_23)
        self.disc_epochs.setObjectName(u"disc_epochs")
        self.disc_epochs.setMinimum(1)
        self.disc_epochs.setMaximum(999999)
        self.disc_epochs.setValue(100)

        self.formLayout_11.setWidget(0, QFormLayout.ItemRole.FieldRole, self.disc_epochs)

        self.label_35 = QLabel(self.frame_23)
        self.label_35.setObjectName(u"label_35")

        self.formLayout_11.setWidget(1, QFormLayout.ItemRole.LabelRole, self.label_35)

        self.disc_epochs_decay = QSpinBox(self.frame_23)
        self.disc_epochs_decay.setObjectName(u"disc_epochs_decay")
        self.disc_epochs_decay.setMinimum(1)
        self.disc_epochs_decay.setMaximum(999999)
        self.disc_epochs_decay.setValue(100)

        self.formLayout_11.setWidget(1, QFormLayout.ItemRole.FieldRole, self.disc_epochs_decay)


        self.verticalLayout_38.addWidget(self.frame_23)

        self.frame_43 = QFrame(self.disc_schedule_box)
        self.frame_43.setObjectName(u"frame_43")
        self.frame_43.setFrameShape(QFrame.StyledPanel)
        self.frame_43.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_43 = QHBoxLayout(self.frame_43)
        self.horizontalLayout_43.setObjectName(u"horizontalLayout_43")
        self.horizontalLayout_43.setContentsMargins(0, 0, 0, 0)
        self.label_62 = QLabel(self.frame_43)
        self.label_62.setObjectName(u"label_62")
        sizePolicy.setHeightForWidth(self.label_62.sizePolicy().hasHeightForWidth())
        self.label_62.setSizePolicy(sizePolicy)
        self.label_62.setFont(font11)
        self.label_62.setFrameShape(QFrame.NoFrame)
        self.label_62.setFrameShadow(QFrame.Raised)

        self.horizontalLayout_43.addWidget(self.label_62)

        self.line_16 = QFrame(self.frame_43)
        self.line_16.setObjectName(u"line_16")
        self.line_16.setFrameShadow(QFrame.Raised)
        self.line_16.setFrameShape(QFrame.Shape.HLine)

        self.horizontalLayout_43.addWidget(self.line_16)


        self.verticalLayout_38.addWidget(self.frame_43)

        self.frame_46 = QFrame(self.disc_schedule_box)
        self.frame_46.setObjectName(u"frame_46")
        self.frame_46.setFrameShape(QFrame.StyledPanel)
        self.frame_46.setFrameShadow(QFrame.Raised)
        self.formLayout_10 = QFormLayout(self.frame_46)
        self.formLayout_10.setObjectName(u"formLayout_10")
        self.formLayout_10.setContentsMargins(0, 0, 0, 0)
        self.label_12 = QLabel(self.frame_46)
        self.label_12.setObjectName(u"label_12")

        self.formLayout_10.setWidget(0, QFormLayout.ItemRole.LabelRole, self.label_12)

        self.label_36 = QLabel(self.frame_46)
        self.label_36.setObjectName(u"label_36")

        self.formLayout_10.setWidget(1, QFormLayout.ItemRole.LabelRole, self.label_36)

        self.horizontalLayout_24 = QHBoxLayout()
        self.horizontalLayout_24.setObjectName(u"horizontalLayout_24")
        self.disc_lr_initial_slider = QSlider(self.frame_46)
        self.disc_lr_initial_slider.setObjectName(u"disc_lr_initial_slider")
        self.disc_lr_initial_slider.setMaximum(100)
        self.disc_lr_initial_slider.setValue(2)
        self.disc_lr_initial_slider.setOrientation(Qt.Horizontal)

        self.horizontalLayout_24.addWidget(self.disc_lr_initial_slider)

        self.disc_lr_initial = QDoubleSpinBox(self.frame_46)
        self.disc_lr_initial.setObjectName(u"disc_lr_initial")
        self.disc_lr_initial.setDecimals(5)
        self.disc_lr_initial.setMinimum(0.000000000000000)
        self.disc_lr_initial.setMaximum(0.100000000000000)
        self.disc_lr_initial.setSingleStep(0.000100000000000)
        self.disc_lr_initial.setValue(0.000200000000000)

        self.horizontalLayout_24.addWidget(self.disc_lr_initial)


        self.formLayout_10.setLayout(0, QFormLayout.ItemRole.FieldRole, self.horizontalLayout_24)

        self.horizontalLayout_25 = QHBoxLayout()
        self.horizontalLayout_25.setObjectName(u"horizontalLayout_25")
        self.disc_lr_target_slider = QSlider(self.frame_46)
        self.disc_lr_target_slider.setObjectName(u"disc_lr_target_slider")
        self.disc_lr_target_slider.setMaximum(100)
        self.disc_lr_target_slider.setValue(2)
        self.disc_lr_target_slider.setOrientation(Qt.Horizontal)

        self.horizontalLayout_25.addWidget(self.disc_lr_target_slider)

        self.disc_lr_target = QDoubleSpinBox(self.frame_46)
        self.disc_lr_target.setObjectName(u"disc_lr_target")
        self.disc_lr_target.setDecimals(5)
        self.disc_lr_target.setMinimum(0.000000000000000)
        self.disc_lr_target.setMaximum(0.100000000000000)
        self.disc_lr_target.setSingleStep(0.000100000000000)
        self.disc_lr_target.setValue(0.000200000000000)

        self.horizontalLayout_25.addWidget(self.disc_lr_target)


        self.formLayout_10.setLayout(1, QFormLayout.ItemRole.FieldRole, self.horizontalLayout_25)


        self.verticalLayout_38.addWidget(self.frame_46)


        self.verticalLayout_36.addWidget(self.disc_schedule_box)

        self.frame_44 = QFrame(self.scrollAreaWidgetContents_6)
        self.frame_44.setObjectName(u"frame_44")
        self.frame_44.setFrameShape(QFrame.StyledPanel)
        self.frame_44.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_44 = QHBoxLayout(self.frame_44)
        self.horizontalLayout_44.setObjectName(u"horizontalLayout_44")
        self.horizontalLayout_44.setContentsMargins(0, 0, 0, 0)
        self.label_63 = QLabel(self.frame_44)
        self.label_63.setObjectName(u"label_63")
        sizePolicy.setHeightForWidth(self.label_63.sizePolicy().hasHeightForWidth())
        self.label_63.setSizePolicy(sizePolicy)
        self.label_63.setFont(font11)
        self.label_63.setFrameShape(QFrame.NoFrame)
        self.label_63.setFrameShadow(QFrame.Raised)

        self.horizontalLayout_44.addWidget(self.label_63)

        self.line_17 = QFrame(self.frame_44)
        self.line_17.setObjectName(u"line_17")
        self.line_17.setFrameShadow(QFrame.Raised)
        self.line_17.setFrameShape(QFrame.Shape.HLine)

        self.horizontalLayout_44.addWidget(self.line_17)


        self.verticalLayout_36.addWidget(self.frame_44)

        self.formLayout_14 = QFormLayout()
        self.formLayout_14.setObjectName(u"formLayout_14")
        self.label_15 = QLabel(self.scrollAreaWidgetContents_6)
        self.label_15.setObjectName(u"label_15")

        self.formLayout_14.setWidget(0, QFormLayout.ItemRole.LabelRole, self.label_15)

        self.horizontalLayout_26 = QHBoxLayout()
        self.horizontalLayout_26.setObjectName(u"horizontalLayout_26")
        self.disc_optim_beta1_slider = QSlider(self.scrollAreaWidgetContents_6)
        self.disc_optim_beta1_slider.setObjectName(u"disc_optim_beta1_slider")
        self.disc_optim_beta1_slider.setMaximum(250)
        self.disc_optim_beta1_slider.setOrientation(Qt.Horizontal)

        self.horizontalLayout_26.addWidget(self.disc_optim_beta1_slider)

        self.disc_optim_beta1 = QDoubleSpinBox(self.scrollAreaWidgetContents_6)
        self.disc_optim_beta1.setObjectName(u"disc_optim_beta1")
        self.disc_optim_beta1.setMaximum(2.500000000000000)
        self.disc_optim_beta1.setSingleStep(0.100000000000000)
        self.disc_optim_beta1.setValue(0.500000000000000)

        self.horizontalLayout_26.addWidget(self.disc_optim_beta1)


        self.formLayout_14.setLayout(0, QFormLayout.ItemRole.FieldRole, self.horizontalLayout_26)


        self.verticalLayout_36.addLayout(self.formLayout_14)

        self.frame_45 = QFrame(self.scrollAreaWidgetContents_6)
        self.frame_45.setObjectName(u"frame_45")
        self.frame_45.setFrameShape(QFrame.StyledPanel)
        self.frame_45.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_45 = QHBoxLayout(self.frame_45)
        self.horizontalLayout_45.setObjectName(u"horizontalLayout_45")
        self.horizontalLayout_45.setContentsMargins(0, 0, 0, 0)
        self.label_64 = QLabel(self.frame_45)
        self.label_64.setObjectName(u"label_64")
        sizePolicy.setHeightForWidth(self.label_64.sizePolicy().hasHeightForWidth())
        self.label_64.setSizePolicy(sizePolicy)
        self.label_64.setFont(font11)
        self.label_64.setFrameShape(QFrame.NoFrame)
        self.label_64.setFrameShadow(QFrame.Raised)

        self.horizontalLayout_45.addWidget(self.label_64)

        self.line_18 = QFrame(self.frame_45)
        self.line_18.setObjectName(u"line_18")
        self.line_18.setFrameShadow(QFrame.Raised)
        self.line_18.setFrameShape(QFrame.Shape.HLine)

        self.horizontalLayout_45.addWidget(self.line_18)


        self.verticalLayout_36.addWidget(self.frame_45)

        self.verticalSpacer_4 = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.verticalLayout_36.addItem(self.verticalSpacer_4)

        self.scrollArea_6.setWidget(self.scrollAreaWidgetContents_6)

        self.verticalLayout_35.addWidget(self.scrollArea_6)

        self.tabWidget_2.addTab(self.tab_7, "")
        self.tab_3 = QWidget()
        self.tab_3.setObjectName(u"tab_3")
        self.verticalLayout_14 = QVBoxLayout(self.tab_3)
        self.verticalLayout_14.setObjectName(u"verticalLayout_14")
        self.verticalLayout_14.setContentsMargins(0, 0, 0, 0)
        self.scrollArea_5 = QScrollArea(self.tab_3)
        self.scrollArea_5.setObjectName(u"scrollArea_5")
        self.scrollArea_5.setWidgetResizable(True)
        self.scrollAreaWidgetContents_5 = QWidget()
        self.scrollAreaWidgetContents_5.setObjectName(u"scrollAreaWidgetContents_5")
        self.scrollAreaWidgetContents_5.setGeometry(QRect(0, 0, 323, 457))
        self.verticalLayout_16 = QVBoxLayout(self.scrollAreaWidgetContents_5)
        self.verticalLayout_16.setObjectName(u"verticalLayout_16")
        self.verticalLayout_16.setContentsMargins(6, 6, 6, 6)
        self.frame_25 = QFrame(self.scrollAreaWidgetContents_5)
        self.frame_25.setObjectName(u"frame_25")
        self.frame_25.setFrameShape(QFrame.StyledPanel)
        self.frame_25.setFrameShadow(QFrame.Raised)
        self.formLayout_2 = QFormLayout(self.frame_25)
        self.formLayout_2.setObjectName(u"formLayout_2")
        self.formLayout_2.setContentsMargins(0, 0, 0, 0)
        self.label_13 = QLabel(self.frame_25)
        self.label_13.setObjectName(u"label_13")

        self.formLayout_2.setWidget(0, QFormLayout.ItemRole.LabelRole, self.label_13)

        self.horizontalLayout_10 = QHBoxLayout()
        self.horizontalLayout_10.setObjectName(u"horizontalLayout_10")
        self.lambda_l1_slider = QSlider(self.frame_25)
        self.lambda_l1_slider.setObjectName(u"lambda_l1_slider")
        self.lambda_l1_slider.setMaximum(500)
        self.lambda_l1_slider.setValue(100)
        self.lambda_l1_slider.setOrientation(Qt.Horizontal)

        self.horizontalLayout_10.addWidget(self.lambda_l1_slider)

        self.lambda_l1 = QDoubleSpinBox(self.frame_25)
        self.lambda_l1.setObjectName(u"lambda_l1")
        self.lambda_l1.setMaximum(500.000000000000000)
        self.lambda_l1.setSingleStep(5.000000000000000)
        self.lambda_l1.setValue(100.000000000000000)

        self.horizontalLayout_10.addWidget(self.lambda_l1)


        self.formLayout_2.setLayout(0, QFormLayout.ItemRole.FieldRole, self.horizontalLayout_10)

        self.label_17 = QLabel(self.frame_25)
        self.label_17.setObjectName(u"label_17")

        self.formLayout_2.setWidget(1, QFormLayout.ItemRole.LabelRole, self.label_17)

        self.horizontalLayout_11 = QHBoxLayout()
        self.horizontalLayout_11.setObjectName(u"horizontalLayout_11")
        self.lambda_sobel_slider = QSlider(self.frame_25)
        self.lambda_sobel_slider.setObjectName(u"lambda_sobel_slider")
        self.lambda_sobel_slider.setMaximum(100)
        self.lambda_sobel_slider.setOrientation(Qt.Horizontal)

        self.horizontalLayout_11.addWidget(self.lambda_sobel_slider)

        self.lambda_sobel = QDoubleSpinBox(self.frame_25)
        self.lambda_sobel.setObjectName(u"lambda_sobel")
        self.lambda_sobel.setMaximum(500.000000000000000)
        self.lambda_sobel.setValue(0.000000000000000)

        self.horizontalLayout_11.addWidget(self.lambda_sobel)


        self.formLayout_2.setLayout(1, QFormLayout.ItemRole.FieldRole, self.horizontalLayout_11)

        self.label_37 = QLabel(self.frame_25)
        self.label_37.setObjectName(u"label_37")

        self.formLayout_2.setWidget(2, QFormLayout.ItemRole.LabelRole, self.label_37)

        self.label_38 = QLabel(self.frame_25)
        self.label_38.setObjectName(u"label_38")

        self.formLayout_2.setWidget(3, QFormLayout.ItemRole.LabelRole, self.label_38)

        self.horizontalLayout_28 = QHBoxLayout()
        self.horizontalLayout_28.setObjectName(u"horizontalLayout_28")
        self.lambda_laplacian_slider = QSlider(self.frame_25)
        self.lambda_laplacian_slider.setObjectName(u"lambda_laplacian_slider")
        self.lambda_laplacian_slider.setMaximum(100)
        self.lambda_laplacian_slider.setOrientation(Qt.Horizontal)

        self.horizontalLayout_28.addWidget(self.lambda_laplacian_slider)

        self.lambda_laplacian = QDoubleSpinBox(self.frame_25)
        self.lambda_laplacian.setObjectName(u"lambda_laplacian")
        self.lambda_laplacian.setMaximum(500.000000000000000)
        self.lambda_laplacian.setValue(0.000000000000000)

        self.horizontalLayout_28.addWidget(self.lambda_laplacian)


        self.formLayout_2.setLayout(2, QFormLayout.ItemRole.FieldRole, self.horizontalLayout_28)

        self.horizontalLayout_27 = QHBoxLayout()
        self.horizontalLayout_27.setObjectName(u"horizontalLayout_27")
        self.lambda_vgg_slider = QSlider(self.frame_25)
        self.lambda_vgg_slider.setObjectName(u"lambda_vgg_slider")
        self.lambda_vgg_slider.setMaximum(100)
        self.lambda_vgg_slider.setValue(10)
        self.lambda_vgg_slider.setOrientation(Qt.Horizontal)

        self.horizontalLayout_27.addWidget(self.lambda_vgg_slider)

        self.lambda_vgg = QDoubleSpinBox(self.frame_25)
        self.lambda_vgg.setObjectName(u"lambda_vgg")
        self.lambda_vgg.setMaximum(500.000000000000000)
        self.lambda_vgg.setValue(10.000000000000000)

        self.horizontalLayout_27.addWidget(self.lambda_vgg)


        self.formLayout_2.setLayout(3, QFormLayout.ItemRole.FieldRole, self.horizontalLayout_27)


        self.verticalLayout_16.addWidget(self.frame_25)

        self.verticalSpacer_5 = QSpacerItem(20, 356, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.verticalLayout_16.addItem(self.verticalSpacer_5)

        self.scrollArea_5.setWidget(self.scrollAreaWidgetContents_5)

        self.verticalLayout_14.addWidget(self.scrollArea_5)

        self.tabWidget_2.addTab(self.tab_3, "")
        self.tab_4 = QWidget()
        self.tab_4.setObjectName(u"tab_4")
        self.verticalLayout_17 = QVBoxLayout(self.tab_4)
        self.verticalLayout_17.setObjectName(u"verticalLayout_17")
        self.verticalLayout_17.setContentsMargins(0, 0, 0, 0)
        self.scrollArea_7 = QScrollArea(self.tab_4)
        self.scrollArea_7.setObjectName(u"scrollArea_7")
        self.scrollArea_7.setWidgetResizable(True)
        self.scrollAreaWidgetContents_7 = QWidget()
        self.scrollAreaWidgetContents_7.setObjectName(u"scrollAreaWidgetContents_7")
        self.scrollAreaWidgetContents_7.setGeometry(QRect(0, 0, 323, 457))
        self.verticalLayout_18 = QVBoxLayout(self.scrollAreaWidgetContents_7)
        self.verticalLayout_18.setObjectName(u"verticalLayout_18")
        self.verticalLayout_18.setContentsMargins(2, 2, 2, 2)
        self.groupBox_6 = QGroupBox(self.scrollAreaWidgetContents_7)
        self.groupBox_6.setObjectName(u"groupBox_6")
        self.verticalLayout_24 = QVBoxLayout(self.groupBox_6)
        self.verticalLayout_24.setObjectName(u"verticalLayout_24")
        self.verticalLayout_24.setContentsMargins(4, 4, 4, 4)
        self.frame_48 = QFrame(self.groupBox_6)
        self.frame_48.setObjectName(u"frame_48")
        self.frame_48.setFrameShape(QFrame.StyledPanel)
        self.frame_48.setFrameShadow(QFrame.Raised)
        self.formLayout_16 = QFormLayout(self.frame_48)
        self.formLayout_16.setObjectName(u"formLayout_16")
        self.formLayout_16.setContentsMargins(0, 0, 0, 0)
        self.label_18 = QLabel(self.frame_48)
        self.label_18.setObjectName(u"label_18")
        sizePolicy.setHeightForWidth(self.label_18.sizePolicy().hasHeightForWidth())
        self.label_18.setSizePolicy(sizePolicy)

        self.formLayout_16.setWidget(0, QFormLayout.ItemRole.LabelRole, self.label_18)

        self.save_model = QCheckBox(self.frame_48)
        self.save_model.setObjectName(u"save_model")
        sizePolicy7.setHeightForWidth(self.save_model.sizePolicy().hasHeightForWidth())
        self.save_model.setSizePolicy(sizePolicy7)
        self.save_model.setChecked(True)

        self.formLayout_16.setWidget(0, QFormLayout.ItemRole.FieldRole, self.save_model)

        self.label_24 = QLabel(self.frame_48)
        self.label_24.setObjectName(u"label_24")
        sizePolicy.setHeightForWidth(self.label_24.sizePolicy().hasHeightForWidth())
        self.label_24.setSizePolicy(sizePolicy)

        self.formLayout_16.setWidget(1, QFormLayout.ItemRole.LabelRole, self.label_24)

        self.horizontalLayout_17 = QHBoxLayout()
        self.horizontalLayout_17.setObjectName(u"horizontalLayout_17")
        self.model_save_rate = QSpinBox(self.frame_48)
        self.model_save_rate.setObjectName(u"model_save_rate")
        sizePolicy7.setHeightForWidth(self.model_save_rate.sizePolicy().hasHeightForWidth())
        self.model_save_rate.setSizePolicy(sizePolicy7)
        self.model_save_rate.setMinimum(1)
        self.model_save_rate.setValue(5)

        self.horizontalLayout_17.addWidget(self.model_save_rate)

        self.label_42 = QLabel(self.frame_48)
        self.label_42.setObjectName(u"label_42")

        self.horizontalLayout_17.addWidget(self.label_42)

        self.horizontalSpacer_2 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_17.addItem(self.horizontalSpacer_2)


        self.formLayout_16.setLayout(1, QFormLayout.ItemRole.FieldRole, self.horizontalLayout_17)


        self.verticalLayout_24.addWidget(self.frame_48)


        self.verticalLayout_18.addWidget(self.groupBox_6)

        self.groupBox = QGroupBox(self.scrollAreaWidgetContents_7)
        self.groupBox.setObjectName(u"groupBox")
        self.formLayout_6 = QFormLayout(self.groupBox)
        self.formLayout_6.setObjectName(u"formLayout_6")
        self.formLayout_6.setContentsMargins(4, 4, 4, 4)
        self.label_25 = QLabel(self.groupBox)
        self.label_25.setObjectName(u"label_25")

        self.formLayout_6.setWidget(0, QFormLayout.ItemRole.LabelRole, self.label_25)

        self.save_examples = QCheckBox(self.groupBox)
        self.save_examples.setObjectName(u"save_examples")
        self.save_examples.setChecked(True)

        self.formLayout_6.setWidget(0, QFormLayout.ItemRole.FieldRole, self.save_examples)

        self.label_26 = QLabel(self.groupBox)
        self.label_26.setObjectName(u"label_26")

        self.formLayout_6.setWidget(1, QFormLayout.ItemRole.LabelRole, self.label_26)

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

        self.label_27 = QLabel(self.groupBox)
        self.label_27.setObjectName(u"label_27")

        self.formLayout_6.setWidget(2, QFormLayout.ItemRole.LabelRole, self.label_27)

        self.horizontalLayout_19 = QHBoxLayout()
        self.horizontalLayout_19.setObjectName(u"horizontalLayout_19")
        self.num_examples = QSpinBox(self.groupBox)
        self.num_examples.setObjectName(u"num_examples")

        self.horizontalLayout_19.addWidget(self.num_examples)

        self.horizontalSpacer_4 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_19.addItem(self.horizontalSpacer_4)


        self.formLayout_6.setLayout(2, QFormLayout.ItemRole.FieldRole, self.horizontalLayout_19)


        self.verticalLayout_18.addWidget(self.groupBox)

        self.verticalSpacer_10 = QSpacerItem(20, 270, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.verticalLayout_18.addItem(self.verticalSpacer_10)

        self.scrollArea_7.setWidget(self.scrollAreaWidgetContents_7)

        self.verticalLayout_17.addWidget(self.scrollArea_7)

        self.tabWidget_2.addTab(self.tab_4, "")
        self.tab_2 = QWidget()
        self.tab_2.setObjectName(u"tab_2")
        self.verticalLayout_19 = QVBoxLayout(self.tab_2)
        self.verticalLayout_19.setObjectName(u"verticalLayout_19")
        self.verticalLayout_19.setContentsMargins(3, 0, 0, 0)
        self.scrollArea_8 = QScrollArea(self.tab_2)
        self.scrollArea_8.setObjectName(u"scrollArea_8")
        self.scrollArea_8.setWidgetResizable(True)
        self.scrollAreaWidgetContents_8 = QWidget()
        self.scrollAreaWidgetContents_8.setObjectName(u"scrollAreaWidgetContents_8")
        self.scrollAreaWidgetContents_8.setGeometry(QRect(0, 0, 320, 457))
        self.verticalLayout_21 = QVBoxLayout(self.scrollAreaWidgetContents_8)
        self.verticalLayout_21.setObjectName(u"verticalLayout_21")
        self.verticalLayout_21.setContentsMargins(2, 2, 2, 2)
        self.visdom_enable = QGroupBox(self.scrollAreaWidgetContents_8)
        self.visdom_enable.setObjectName(u"visdom_enable")
        sizePolicy2.setHeightForWidth(self.visdom_enable.sizePolicy().hasHeightForWidth())
        self.visdom_enable.setSizePolicy(sizePolicy2)
        self.visdom_enable.setCheckable(True)
        self.visdom_enable.setChecked(False)
        self.formLayout_9 = QFormLayout(self.visdom_enable)
        self.formLayout_9.setObjectName(u"formLayout_9")
        self.formLayout_9.setContentsMargins(4, 4, 4, 4)
        self.label_28 = QLabel(self.visdom_enable)
        self.label_28.setObjectName(u"label_28")

        self.formLayout_9.setWidget(0, QFormLayout.ItemRole.LabelRole, self.label_28)

        self.label_29 = QLabel(self.visdom_enable)
        self.label_29.setObjectName(u"label_29")

        self.formLayout_9.setWidget(1, QFormLayout.ItemRole.LabelRole, self.label_29)

        self.label_30 = QLabel(self.visdom_enable)
        self.label_30.setObjectName(u"label_30")

        self.formLayout_9.setWidget(2, QFormLayout.ItemRole.LabelRole, self.label_30)

        self.label_65 = QLabel(self.visdom_enable)
        self.label_65.setObjectName(u"label_65")

        self.formLayout_9.setWidget(3, QFormLayout.ItemRole.LabelRole, self.label_65)

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


        self.verticalLayout_21.addWidget(self.visdom_enable)

        self.verticalSpacer_11 = QSpacerItem(20, 344, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.verticalLayout_21.addItem(self.verticalSpacer_11)

        self.scrollArea_8.setWidget(self.scrollAreaWidgetContents_8)

        self.verticalLayout_19.addWidget(self.scrollArea_8)

        self.tabWidget_2.addTab(self.tab_2, "")

        self.verticalLayout_20.addWidget(self.tabWidget_2)

        self.scrollArea_3.setWidget(self.train_settings_container)

        self.verticalLayout_22.addWidget(self.scrollArea_3)

        self.settings_pages.addWidget(self.train_page)
        self.testing_page = QWidget()
        self.testing_page.setObjectName(u"testing_page")
        self.verticalLayout_43 = QVBoxLayout(self.testing_page)
        self.verticalLayout_43.setObjectName(u"verticalLayout_43")
        self.verticalLayout_43.setContentsMargins(0, 0, 0, 0)
        self.scrollArea_10 = QScrollArea(self.testing_page)
        self.scrollArea_10.setObjectName(u"scrollArea_10")
        self.scrollArea_10.setWidgetResizable(True)
        self.scrollAreaWidgetContents_2 = QWidget()
        self.scrollAreaWidgetContents_2.setObjectName(u"scrollAreaWidgetContents_2")
        self.scrollAreaWidgetContents_2.setGeometry(QRect(0, 0, 331, 544))
        self.verticalLayout_44 = QVBoxLayout(self.scrollAreaWidgetContents_2)
        self.verticalLayout_44.setObjectName(u"verticalLayout_44")
        self.label_23 = QLabel(self.scrollAreaWidgetContents_2)
        self.label_23.setObjectName(u"label_23")
        self.label_23.setFont(font7)
        self.label_23.setAlignment(Qt.AlignCenter)

        self.verticalLayout_44.addWidget(self.label_23)

        self.scrollArea_10.setWidget(self.scrollAreaWidgetContents_2)

        self.verticalLayout_43.addWidget(self.scrollArea_10)

        self.settings_pages.addWidget(self.testing_page)
        self.review_page = QWidget()
        self.review_page.setObjectName(u"review_page")
        self.verticalLayout_45 = QVBoxLayout(self.review_page)
        self.verticalLayout_45.setObjectName(u"verticalLayout_45")
        self.verticalLayout_45.setContentsMargins(0, 0, 0, 0)
        self.scrollArea_11 = QScrollArea(self.review_page)
        self.scrollArea_11.setObjectName(u"scrollArea_11")
        self.scrollArea_11.setWidgetResizable(True)
        self.scrollAreaWidgetContents_3 = QWidget()
        self.scrollAreaWidgetContents_3.setObjectName(u"scrollAreaWidgetContents_3")
        self.scrollAreaWidgetContents_3.setGeometry(QRect(0, 0, 331, 544))
        self.verticalLayout_46 = QVBoxLayout(self.scrollAreaWidgetContents_3)
        self.verticalLayout_46.setObjectName(u"verticalLayout_46")
        self.label_40 = QLabel(self.scrollAreaWidgetContents_3)
        self.label_40.setObjectName(u"label_40")
        self.label_40.setFont(font7)
        self.label_40.setAlignment(Qt.AlignCenter)

        self.verticalLayout_46.addWidget(self.label_40)

        self.scrollArea_11.setWidget(self.scrollAreaWidgetContents_3)

        self.verticalLayout_45.addWidget(self.scrollArea_11)

        self.settings_pages.addWidget(self.review_page)
        self.utilities_page = QWidget()
        self.utilities_page.setObjectName(u"utilities_page")
        self.verticalLayout_47 = QVBoxLayout(self.utilities_page)
        self.verticalLayout_47.setObjectName(u"verticalLayout_47")
        self.verticalLayout_47.setContentsMargins(0, 0, 0, 0)
        self.scrollArea_12 = QScrollArea(self.utilities_page)
        self.scrollArea_12.setObjectName(u"scrollArea_12")
        self.scrollArea_12.setWidgetResizable(True)
        self.scrollAreaWidgetContents_9 = QWidget()
        self.scrollAreaWidgetContents_9.setObjectName(u"scrollAreaWidgetContents_9")
        self.scrollAreaWidgetContents_9.setGeometry(QRect(0, 0, 331, 544))
        self.verticalLayout_48 = QVBoxLayout(self.scrollAreaWidgetContents_9)
        self.verticalLayout_48.setObjectName(u"verticalLayout_48")
        self.label_66 = QLabel(self.scrollAreaWidgetContents_9)
        self.label_66.setObjectName(u"label_66")
        self.label_66.setFont(font7)
        self.label_66.setAlignment(Qt.AlignCenter)

        self.verticalLayout_48.addWidget(self.label_66)

        self.scrollArea_12.setWidget(self.scrollAreaWidgetContents_9)

        self.verticalLayout_47.addWidget(self.scrollArea_12)

        self.settings_pages.addWidget(self.utilities_page)
        self.settings_page = QWidget()
        self.settings_page.setObjectName(u"settings_page")
        self.verticalLayout_49 = QVBoxLayout(self.settings_page)
        self.verticalLayout_49.setObjectName(u"verticalLayout_49")
        self.verticalLayout_49.setContentsMargins(0, 0, 0, 0)
        self.scrollArea_13 = QScrollArea(self.settings_page)
        self.scrollArea_13.setObjectName(u"scrollArea_13")
        self.scrollArea_13.setWidgetResizable(True)
        self.scrollAreaWidgetContents_10 = QWidget()
        self.scrollAreaWidgetContents_10.setObjectName(u"scrollAreaWidgetContents_10")
        self.scrollAreaWidgetContents_10.setGeometry(QRect(0, 0, 331, 544))
        self.verticalLayout_50 = QVBoxLayout(self.scrollAreaWidgetContents_10)
        self.verticalLayout_50.setObjectName(u"verticalLayout_50")
        self.verticalLayout_50.setContentsMargins(0, 0, 0, 0)
        self.frame_53 = QFrame(self.scrollAreaWidgetContents_10)
        self.frame_53.setObjectName(u"frame_53")
        self.frame_53.setFrameShape(QFrame.StyledPanel)
        self.frame_53.setFrameShadow(QFrame.Raised)
        self.verticalLayout_42 = QVBoxLayout(self.frame_53)
        self.verticalLayout_42.setObjectName(u"verticalLayout_42")
        self.reload_stylesheet = QPushButton(self.frame_53)
        self.reload_stylesheet.setObjectName(u"reload_stylesheet")

        self.verticalLayout_42.addWidget(self.reload_stylesheet)

        self.always_on_top = QCheckBox(self.frame_53)
        self.always_on_top.setObjectName(u"always_on_top")
        self.always_on_top.setChecked(True)

        self.verticalLayout_42.addWidget(self.always_on_top)

        self.horizontalLayout_31 = QHBoxLayout()
        self.horizontalLayout_31.setObjectName(u"horizontalLayout_31")
        self.label_44 = QLabel(self.frame_53)
        self.label_44.setObjectName(u"label_44")

        self.horizontalLayout_31.addWidget(self.label_44)

        self.update_frequency_slider = QSlider(self.frame_53)
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

        self.update_frequency = QSpinBox(self.frame_53)
        self.update_frequency.setObjectName(u"update_frequency")
        self.update_frequency.setMinimum(1)
        self.update_frequency.setMaximum(99999)
        self.update_frequency.setSingleStep(10)
        self.update_frequency.setValue(50)

        self.horizontalLayout_31.addWidget(self.update_frequency)


        self.verticalLayout_42.addLayout(self.horizontalLayout_31)

        self.frame_20 = QFrame(self.frame_53)
        self.frame_20.setObjectName(u"frame_20")
        self.frame_20.setFrameShape(QFrame.StyledPanel)
        self.frame_20.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_21 = QHBoxLayout(self.frame_20)
        self.horizontalLayout_21.setObjectName(u"horizontalLayout_21")
        self.horizontalLayout_21.setContentsMargins(0, 0, 0, 0)
        self.label_67 = QLabel(self.frame_20)
        self.label_67.setObjectName(u"label_67")

        self.horizontalLayout_21.addWidget(self.label_67)

        self.set_accent_color = QPushButton(self.frame_20)
        self.set_accent_color.setObjectName(u"set_accent_color")

        self.horizontalLayout_21.addWidget(self.set_accent_color)


        self.verticalLayout_42.addWidget(self.frame_20)

        self.verticalSpacer_13 = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.verticalLayout_42.addItem(self.verticalSpacer_13)


        self.verticalLayout_50.addWidget(self.frame_53)

        self.scrollArea_13.setWidget(self.scrollAreaWidgetContents_10)

        self.verticalLayout_49.addWidget(self.scrollArea_13)

        self.settings_pages.addWidget(self.settings_page)

        self.verticalLayout_13.addWidget(self.settings_pages)


        self.verticalLayout_2.addWidget(self.train_settings)

        self.train_functions_frame = QFrame(self.frame_2)
        self.train_functions_frame.setObjectName(u"train_functions_frame")
        sizePolicy2.setHeightForWidth(self.train_functions_frame.sizePolicy().hasHeightForWidth())
        self.train_functions_frame.setSizePolicy(sizePolicy2)
        self.train_functions_frame.setFrameShape(QFrame.StyledPanel)
        self.train_functions_frame.setFrameShadow(QFrame.Raised)
        self.train_functions_frame.setLineWidth(1)
        self.verticalLayout_33 = QVBoxLayout(self.train_functions_frame)
        self.verticalLayout_33.setObjectName(u"verticalLayout_33")
        self.verticalLayout_33.setContentsMargins(3, 3, 3, 3)
        self.train_start = QPushButton(self.train_functions_frame)
        self.train_start.setObjectName(u"train_start")

        self.verticalLayout_33.addWidget(self.train_start)

        self.train_pause = QPushButton(self.train_functions_frame)
        self.train_pause.setObjectName(u"train_pause")

        self.verticalLayout_33.addWidget(self.train_pause)

        self.train_stop = QPushButton(self.train_functions_frame)
        self.train_stop.setObjectName(u"train_stop")

        self.verticalLayout_33.addWidget(self.train_stop)


        self.verticalLayout_2.addWidget(self.train_functions_frame)


        self.horizontalLayout_5.addWidget(self.frame_2)

        self.settings_dock.setWidget(self.dockWidgetContents_8)
        MainWindow.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.settings_dock)

        self.retranslateUi(MainWindow)

        self.settings_pages.setCurrentIndex(6)
        self.direction.setCurrentIndex(-1)
        self.tabWidget_2.setCurrentIndex(0)


        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"MainWindow", None))
        self.actionLoad_From_Config.setText(QCoreApplication.translate("MainWindow", u"Load From Config", None))
#if QT_CONFIG(tooltip)
        self.close_experiment_settings.setToolTip(QCoreApplication.translate("MainWindow", u"Close settings panel", None))
#endif // QT_CONFIG(tooltip)
        self.close_experiment_settings.setText("")
        self.label_19.setText(QCoreApplication.translate("MainWindow", u"A_real", None))
        self.label_20.setText(QCoreApplication.translate("MainWindow", u"B_fake", None))
        self.label_21.setText(QCoreApplication.translate("MainWindow", u"B_real", None))
        self.x_label.setText("")
        self.y_fake_label.setText("")
        self.y_label.setText("")
        self.label_6.setText(QCoreApplication.translate("MainWindow", u"Generator Loss", None))
        self.label_22.setText(QCoreApplication.translate("MainWindow", u"Disciminator Loss", None))
        self.utility_dock.setWindowTitle(QCoreApplication.translate("MainWindow", u"Lift for performance monitor / output log", None))
        self.times_group.setTitle(QCoreApplication.translate("MainWindow", u"Times", None))
        self.label_55.setText(QCoreApplication.translate("MainWindow", u"Train", None))
        self.performance_time_total.setText(QCoreApplication.translate("MainWindow", u"00:00:00.000", None))
        self.label_56.setText(QCoreApplication.translate("MainWindow", u"Total:      ", None))
        self.label_48.setText(QCoreApplication.translate("MainWindow", u"Epoch", None))
        self.label_47.setText(QCoreApplication.translate("MainWindow", u"Fastest:", None))
        self.performance_epoch_fastest.setText(QCoreApplication.translate("MainWindow", u"00:00:00.000", None))
        self.label_50.setText(QCoreApplication.translate("MainWindow", u"Slowest:", None))
        self.performance_epoch_slowest.setText(QCoreApplication.translate("MainWindow", u"00:00:00.000", None))
        self.label_51.setText(QCoreApplication.translate("MainWindow", u"Average:", None))
        self.performance_epoch_average.setText(QCoreApplication.translate("MainWindow", u"00:00:00.000", None))
        self.label_49.setText(QCoreApplication.translate("MainWindow", u"Iteration", None))
        self.label_52.setText(QCoreApplication.translate("MainWindow", u"Fastest:", None))
        self.performance_iter_fastest.setText(QCoreApplication.translate("MainWindow", u"00:00:00.000", None))
        self.label_53.setText(QCoreApplication.translate("MainWindow", u"Slowest:", None))
        self.performance_iter_slowest.setText(QCoreApplication.translate("MainWindow", u"00:00:00.000", None))
        self.label_54.setText(QCoreApplication.translate("MainWindow", u"Average:", None))
        self.performance_iter_average.setText(QCoreApplication.translate("MainWindow", u"00:00:00.000", None))
        self.performance_monitor_group.setTitle(QCoreApplication.translate("MainWindow", u"Performance Monitor", None))
        self.perf_graph_time_label.setText("")
        self.perf_graph_epoch_label.setText("")
        self.performance_graph_reset_framing.setText(QCoreApplication.translate("MainWindow", u"Reset Framing", None))
        self.performance_graph_clear.setText(QCoreApplication.translate("MainWindow", u"Clear Graph", None))
        self.output_log_group.setTitle(QCoreApplication.translate("MainWindow", u"Log", None))
        self.log_output_frozen_header.setText(QCoreApplication.translate("MainWindow", u"* OUTPUT FROZEN *", None))
        self.output_log.setDocumentTitle("")
        self.label_39.setText(QCoreApplication.translate("MainWindow", u"Log Functions", None))
        self.export_output_log.setText(QCoreApplication.translate("MainWindow", u"Export to File", None))
        self.freeze_output_log.setText(QCoreApplication.translate("MainWindow", u"Freeze Ouput", None))
        self.clear_output_log.setText(QCoreApplication.translate("MainWindow", u"Clear", None))
        self.autoscroll_log.setText(QCoreApplication.translate("MainWindow", u"Autoscroll", None))
        self.label_45.setText(QCoreApplication.translate("MainWindow", u"Current Epoch Progress", None))
        self.label_46.setText(QCoreApplication.translate("MainWindow", u"Current Train Progress", None))
        self.label_41.setText(QCoreApplication.translate("MainWindow", u"Current Epoch", None))
        self.settings_dock_main_icon.setText(QCoreApplication.translate("MainWindow", u"icon", None))
#if QT_CONFIG(tooltip)
        self.experiment_settings_btn.setToolTip(QCoreApplication.translate("MainWindow", u"Experiment settings", None))
#endif // QT_CONFIG(tooltip)
        self.experiment_settings_btn.setText("")
#if QT_CONFIG(tooltip)
        self.experiment_settings_label.setToolTip(QCoreApplication.translate("MainWindow", u"Experiment settings", None))
#endif // QT_CONFIG(tooltip)
        self.experiment_settings_label.setText(QCoreApplication.translate("MainWindow", u"Experiment", None))
#if QT_CONFIG(tooltip)
        self.dataset_settings_btn.setToolTip(QCoreApplication.translate("MainWindow", u"Dataset settings", None))
#endif // QT_CONFIG(tooltip)
        self.dataset_settings_btn.setText("")
#if QT_CONFIG(tooltip)
        self.dataset_settings_label.setToolTip(QCoreApplication.translate("MainWindow", u"Dataset settings", None))
#endif // QT_CONFIG(tooltip)
        self.dataset_settings_label.setText(QCoreApplication.translate("MainWindow", u"Dataset     ", None))
#if QT_CONFIG(tooltip)
        self.training_settings_btn.setToolTip(QCoreApplication.translate("MainWindow", u"Training settings", None))
#endif // QT_CONFIG(tooltip)
        self.training_settings_btn.setText("")
#if QT_CONFIG(tooltip)
        self.training_settings_label.setToolTip(QCoreApplication.translate("MainWindow", u"Training settings", None))
#endif // QT_CONFIG(tooltip)
        self.training_settings_label.setText(QCoreApplication.translate("MainWindow", u"Training     ", None))
#if QT_CONFIG(tooltip)
        self.testing_settings_btn.setToolTip(QCoreApplication.translate("MainWindow", u"Testing settings", None))
#endif // QT_CONFIG(tooltip)
        self.testing_settings_btn.setText("")
#if QT_CONFIG(tooltip)
        self.testing_settings_label.setToolTip(QCoreApplication.translate("MainWindow", u"Testing settings", None))
#endif // QT_CONFIG(tooltip)
        self.testing_settings_label.setText(QCoreApplication.translate("MainWindow", u"Testing      ", None))
#if QT_CONFIG(tooltip)
        self.review_settings_btn.setToolTip(QCoreApplication.translate("MainWindow", u"Utilities", None))
#endif // QT_CONFIG(tooltip)
        self.review_settings_btn.setText("")
#if QT_CONFIG(tooltip)
        self.review_settings_label.setToolTip(QCoreApplication.translate("MainWindow", u"Utilities", None))
#endif // QT_CONFIG(tooltip)
        self.review_settings_label.setText(QCoreApplication.translate("MainWindow", u"Review      ", None))
#if QT_CONFIG(tooltip)
        self.utils_settings_btn.setToolTip(QCoreApplication.translate("MainWindow", u"Utilities", None))
#endif // QT_CONFIG(tooltip)
        self.utils_settings_btn.setText("")
#if QT_CONFIG(tooltip)
        self.utils_settings_label.setToolTip(QCoreApplication.translate("MainWindow", u"Utilities", None))
#endif // QT_CONFIG(tooltip)
        self.utils_settings_label.setText(QCoreApplication.translate("MainWindow", u"Utilities      ", None))
#if QT_CONFIG(tooltip)
        self.settings_settings_btn.setToolTip(QCoreApplication.translate("MainWindow", u"Utilities", None))
#endif // QT_CONFIG(tooltip)
        self.settings_settings_btn.setText("")
#if QT_CONFIG(tooltip)
        self.settings_settings_label.setToolTip(QCoreApplication.translate("MainWindow", u"Utilities", None))
#endif // QT_CONFIG(tooltip)
        self.settings_settings_label.setText(QCoreApplication.translate("MainWindow", u"Settings     ", None))
        self.title_tag.setText(QCoreApplication.translate("MainWindow", u"NectarGAN Toolbox", None))
        self.creator_tag.setText(QCoreApplication.translate("MainWindow", u"Created by Zachary Bork", None))
        self.settings_section_header_icon.setText(QCoreApplication.translate("MainWindow", u"HEADER ICON", None))
        self.settings_section_header.setText(QCoreApplication.translate("MainWindow", u"Experiment", None))
        self.label_2.setText(QCoreApplication.translate("MainWindow", u"Output Root", None))
        self.browse_output.setText(QCoreApplication.translate("MainWindow", u"Browse", None))
        self.label_3.setText(QCoreApplication.translate("MainWindow", u"Experiment Name", None))
        self.label_4.setText(QCoreApplication.translate("MainWindow", u"Version", None))
        self.init_from_last_train.setText(QCoreApplication.translate("MainWindow", u"Initialize settings from most recent train", None))
        self.label.setText(QCoreApplication.translate("MainWindow", u"Dataset Root", None))
        self.browse_dataset.setText(QCoreApplication.translate("MainWindow", u"Browse", None))
        self.label_10.setText(QCoreApplication.translate("MainWindow", u"Load Size", None))
        self.label_11.setText(QCoreApplication.translate("MainWindow", u"Crop Size", None))
        self.label_9.setText(QCoreApplication.translate("MainWindow", u"Input Channels", None))
        self.label_5.setText(QCoreApplication.translate("MainWindow", u"Direction", None))
        self.direction.setCurrentText("")
        self.label_8.setText(QCoreApplication.translate("MainWindow", u"Batch Size", None))
        self.continue_train.setTitle(QCoreApplication.translate("MainWindow", u"Continue Train", None))
        self.label_16.setText(QCoreApplication.translate("MainWindow", u"Load Epoch", None))
        self.label_57.setText(QCoreApplication.translate("MainWindow", u"Duration", None))
        self.label_32.setText(QCoreApplication.translate("MainWindow", u"Epochs", None))
        self.label_31.setText(QCoreApplication.translate("MainWindow", u"Epochs Decay", None))
        self.label_60.setText(QCoreApplication.translate("MainWindow", u"Learning Rate", None))
        self.label_7.setText(QCoreApplication.translate("MainWindow", u"Initial", None))
        self.label_33.setText(QCoreApplication.translate("MainWindow", u"Target", None))
        self.label_59.setText(QCoreApplication.translate("MainWindow", u"Optimizer", None))
        self.label_14.setText(QCoreApplication.translate("MainWindow", u"Beta1", None))
        self.label_58.setText(QCoreApplication.translate("MainWindow", u"Architecture", None))
        self.tabWidget_2.setTabText(self.tabWidget_2.indexOf(self.tab), QCoreApplication.translate("MainWindow", u"Generator", None))
        self.separate_lr_schedules.setText(QCoreApplication.translate("MainWindow", u"Seperate Learning Schedules", None))
        self.label_61.setText(QCoreApplication.translate("MainWindow", u"Duration", None))
        self.label_34.setText(QCoreApplication.translate("MainWindow", u"Epochs", None))
        self.label_35.setText(QCoreApplication.translate("MainWindow", u"Epochs Decay", None))
        self.label_62.setText(QCoreApplication.translate("MainWindow", u"Learning Rate", None))
        self.label_12.setText(QCoreApplication.translate("MainWindow", u"Initial", None))
        self.label_36.setText(QCoreApplication.translate("MainWindow", u"Target", None))
        self.label_63.setText(QCoreApplication.translate("MainWindow", u"Optimizer", None))
        self.label_15.setText(QCoreApplication.translate("MainWindow", u"Beta1", None))
        self.label_64.setText(QCoreApplication.translate("MainWindow", u"Architecture", None))
        self.tabWidget_2.setTabText(self.tabWidget_2.indexOf(self.tab_7), QCoreApplication.translate("MainWindow", u"Discriminator", None))
        self.label_13.setText(QCoreApplication.translate("MainWindow", u"L1", None))
        self.label_17.setText(QCoreApplication.translate("MainWindow", u"Sobel", None))
        self.label_37.setText(QCoreApplication.translate("MainWindow", u"Laplacian", None))
        self.label_38.setText(QCoreApplication.translate("MainWindow", u"VGG", None))
        self.tabWidget_2.setTabText(self.tabWidget_2.indexOf(self.tab_3), QCoreApplication.translate("MainWindow", u"Loss", None))
        self.groupBox_6.setTitle(QCoreApplication.translate("MainWindow", u"Save Checkpoints", None))
        self.label_18.setText(QCoreApplication.translate("MainWindow", u"Enable", None))
        self.save_model.setText("")
        self.label_24.setText(QCoreApplication.translate("MainWindow", u"Frequency", None))
        self.label_42.setText(QCoreApplication.translate("MainWindow", u"Epochs", None))
        self.groupBox.setTitle(QCoreApplication.translate("MainWindow", u"Save Example Images", None))
        self.label_25.setText(QCoreApplication.translate("MainWindow", u"Enable", None))
        self.save_examples.setText("")
        self.label_26.setText(QCoreApplication.translate("MainWindow", u"Frequency", None))
        self.label_43.setText(QCoreApplication.translate("MainWindow", u"Epochs", None))
        self.label_27.setText(QCoreApplication.translate("MainWindow", u"Count", None))
        self.tabWidget_2.setTabText(self.tabWidget_2.indexOf(self.tab_4), QCoreApplication.translate("MainWindow", u"Saving", None))
        self.visdom_enable.setTitle(QCoreApplication.translate("MainWindow", u"Enable Visdom", None))
        self.label_28.setText(QCoreApplication.translate("MainWindow", u"Environment Name", None))
        self.label_29.setText(QCoreApplication.translate("MainWindow", u"Port", None))
        self.label_30.setText(QCoreApplication.translate("MainWindow", u"Image Size", None))
        self.label_65.setText(QCoreApplication.translate("MainWindow", u"Update Frequency", None))
        self.visdom_env_name.setText(QCoreApplication.translate("MainWindow", u"main", None))
        self.tabWidget_2.setTabText(self.tabWidget_2.indexOf(self.tab_2), QCoreApplication.translate("MainWindow", u"Visdom", None))
        self.label_23.setText(QCoreApplication.translate("MainWindow", u"Testing", None))
        self.label_40.setText(QCoreApplication.translate("MainWindow", u"Review", None))
        self.label_66.setText(QCoreApplication.translate("MainWindow", u"Utilities", None))
        self.reload_stylesheet.setText(QCoreApplication.translate("MainWindow", u"Reload Stylesheet", None))
        self.always_on_top.setText(QCoreApplication.translate("MainWindow", u"Always on Top", None))
        self.label_44.setText(QCoreApplication.translate("MainWindow", u"Update Rate", None))
        self.label_67.setText(QCoreApplication.translate("MainWindow", u"Set Accent Color", None))
        self.set_accent_color.setText(QCoreApplication.translate("MainWindow", u"Set Accent Color", None))
        self.train_start.setText(QCoreApplication.translate("MainWindow", u"Begin Training", None))
        self.train_pause.setText(QCoreApplication.translate("MainWindow", u"Pause Training", None))
        self.train_stop.setText(QCoreApplication.translate("MainWindow", u"Stop Training", None))
    # retranslateUi

