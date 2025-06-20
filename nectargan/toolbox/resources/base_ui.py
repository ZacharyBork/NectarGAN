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
    QLineEdit, QMainWindow, QProgressBar, QPushButton,
    QScrollArea, QSizePolicy, QSlider, QSpacerItem,
    QSpinBox, QSplitter, QStackedWidget, QStatusBar,
    QTabWidget, QTextEdit, QVBoxLayout, QWidget)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(1920, 1080)
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(MainWindow.sizePolicy().hasHeightForWidth())
        MainWindow.setSizePolicy(sizePolicy)
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
        sizePolicy1 = QSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Preferred)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.close_experiment_settings.sizePolicy().hasHeightForWidth())
        self.close_experiment_settings.setSizePolicy(sizePolicy1)
        self.close_experiment_settings.setMaximumSize(QSize(6, 16777215))
        font = QFont()
        font.setPointSize(12)
        font.setBold(False)
        font.setItalic(False)
        font.setUnderline(False)
        self.close_experiment_settings.setFont(font)
        self.close_experiment_settings.setIconSize(QSize(8, 8))

        self.horizontalLayout.addWidget(self.close_experiment_settings)

        self.centralwidget_pages = QStackedWidget(self.frame)
        self.centralwidget_pages.setObjectName(u"centralwidget_pages")
        self.central_train_page = QWidget()
        self.central_train_page.setObjectName(u"central_train_page")
        self.verticalLayout_25 = QVBoxLayout(self.central_train_page)
        self.verticalLayout_25.setSpacing(0)
        self.verticalLayout_25.setObjectName(u"verticalLayout_25")
        self.verticalLayout_25.setContentsMargins(0, 0, 0, 0)
        self.frame_7 = QFrame(self.central_train_page)
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
        sizePolicy.setHeightForWidth(self.frame_4.sizePolicy().hasHeightForWidth())
        self.frame_4.setSizePolicy(sizePolicy)
        self.frame_4.setFrameShape(QFrame.StyledPanel)
        self.frame_4.setFrameShadow(QFrame.Raised)
        self.verticalLayout_32 = QVBoxLayout(self.frame_4)
        self.verticalLayout_32.setSpacing(3)
        self.verticalLayout_32.setObjectName(u"verticalLayout_32")
        self.verticalLayout_32.setContentsMargins(3, 3, 3, 3)
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
        sizePolicy.setHeightForWidth(self.frame_11.sizePolicy().hasHeightForWidth())
        self.frame_11.setSizePolicy(sizePolicy)
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


        self.verticalLayout_25.addWidget(self.frame_7)

        self.centralwidget_pages.addWidget(self.central_train_page)
        self.central_test_page = QWidget()
        self.central_test_page.setObjectName(u"central_test_page")
        self.verticalLayout_52 = QVBoxLayout(self.central_test_page)
        self.verticalLayout_52.setSpacing(0)
        self.verticalLayout_52.setObjectName(u"verticalLayout_52")
        self.verticalLayout_52.setContentsMargins(0, 0, 0, 0)
        self.scrollArea_9 = QScrollArea(self.central_test_page)
        self.scrollArea_9.setObjectName(u"scrollArea_9")
        self.scrollArea_9.setFrameShape(QFrame.Box)
        self.scrollArea_9.setFrameShadow(QFrame.Sunken)
        self.scrollArea_9.setLineWidth(3)
        self.scrollArea_9.setWidgetResizable(True)
        self.scrollAreaWidgetContents = QWidget()
        self.scrollAreaWidgetContents.setObjectName(u"scrollAreaWidgetContents")
        self.scrollAreaWidgetContents.setGeometry(QRect(0, 0, 1273, 937))
        self.verticalLayout_58 = QVBoxLayout(self.scrollAreaWidgetContents)
        self.verticalLayout_58.setSpacing(0)
        self.verticalLayout_58.setObjectName(u"verticalLayout_58")
        self.verticalLayout_58.setContentsMargins(3, 3, 3, 3)
        self.test_image_labels_frame = QFrame(self.scrollAreaWidgetContents)
        self.test_image_labels_frame.setObjectName(u"test_image_labels_frame")
        sizePolicy2.setHeightForWidth(self.test_image_labels_frame.sizePolicy().hasHeightForWidth())
        self.test_image_labels_frame.setSizePolicy(sizePolicy2)
        self.test_image_labels_frame.setFrameShape(QFrame.Box)
        self.test_image_labels_frame.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_67 = QHBoxLayout(self.test_image_labels_frame)
        self.horizontalLayout_67.setObjectName(u"horizontalLayout_67")
        self.horizontalLayout_67.setContentsMargins(3, 3, 3, 3)
        self.test_a_real_label = QLabel(self.test_image_labels_frame)
        self.test_a_real_label.setObjectName(u"test_a_real_label")
        font3 = QFont()
        font3.setBold(True)
        self.test_a_real_label.setFont(font3)
        self.test_a_real_label.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_67.addWidget(self.test_a_real_label)

        self.test_b_fake_label = QLabel(self.test_image_labels_frame)
        self.test_b_fake_label.setObjectName(u"test_b_fake_label")
        self.test_b_fake_label.setFont(font3)
        self.test_b_fake_label.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_67.addWidget(self.test_b_fake_label)

        self.test_b_real_label = QLabel(self.test_image_labels_frame)
        self.test_b_real_label.setObjectName(u"test_b_real_label")
        self.test_b_real_label.setFont(font3)
        self.test_b_real_label.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_67.addWidget(self.test_b_real_label)

        self.test_info_label = QLabel(self.test_image_labels_frame)
        self.test_info_label.setObjectName(u"test_info_label")
        sizePolicy.setHeightForWidth(self.test_info_label.sizePolicy().hasHeightForWidth())
        self.test_info_label.setSizePolicy(sizePolicy)
        self.test_info_label.setFont(font3)
        self.test_info_label.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_67.addWidget(self.test_info_label)


        self.verticalLayout_58.addWidget(self.test_image_labels_frame)

        self.test_image_frame = QFrame(self.scrollAreaWidgetContents)
        self.test_image_frame.setObjectName(u"test_image_frame")
        self.test_image_frame.setFrameShape(QFrame.StyledPanel)
        self.test_image_frame.setFrameShadow(QFrame.Raised)
        self.test_image_layout = QVBoxLayout(self.test_image_frame)
        self.test_image_layout.setSpacing(0)
        self.test_image_layout.setObjectName(u"test_image_layout")
        self.test_image_layout.setContentsMargins(0, 0, 0, 0)

        self.verticalLayout_58.addWidget(self.test_image_frame)

        self.scrollArea_9.setWidget(self.scrollAreaWidgetContents)

        self.verticalLayout_52.addWidget(self.scrollArea_9)

        self.frame_77 = QFrame(self.central_test_page)
        self.frame_77.setObjectName(u"frame_77")
        sizePolicy2.setHeightForWidth(self.frame_77.sizePolicy().hasHeightForWidth())
        self.frame_77.setSizePolicy(sizePolicy2)
        self.frame_77.setFrameShape(QFrame.StyledPanel)
        self.frame_77.setFrameShadow(QFrame.Raised)
        self.frame_77.setLineWidth(1)
        self.horizontalLayout_66 = QHBoxLayout(self.frame_77)
        self.horizontalLayout_66.setObjectName(u"horizontalLayout_66")
        self.horizontalLayout_66.setContentsMargins(4, 4, 4, 4)
        self.label_93 = QLabel(self.frame_77)
        self.label_93.setObjectName(u"label_93")

        self.horizontalLayout_66.addWidget(self.label_93)

        self.previous_tests = QComboBox(self.frame_77)
        self.previous_tests.setObjectName(u"previous_tests")

        self.horizontalLayout_66.addWidget(self.previous_tests)

        self.label_80 = QLabel(self.frame_77)
        self.label_80.setObjectName(u"label_80")

        self.horizontalLayout_66.addWidget(self.label_80)

        self.test_image_scale = QSlider(self.frame_77)
        self.test_image_scale.setObjectName(u"test_image_scale")
        sizePolicy5 = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        sizePolicy5.setHorizontalStretch(0)
        sizePolicy5.setVerticalStretch(0)
        sizePolicy5.setHeightForWidth(self.test_image_scale.sizePolicy().hasHeightForWidth())
        self.test_image_scale.setSizePolicy(sizePolicy5)
        self.test_image_scale.setMaximum(300)
        self.test_image_scale.setSingleStep(10)
        self.test_image_scale.setValue(100)
        self.test_image_scale.setOrientation(Qt.Horizontal)

        self.horizontalLayout_66.addWidget(self.test_image_scale)

        self.test_image_scale_reset = QPushButton(self.frame_77)
        self.test_image_scale_reset.setObjectName(u"test_image_scale_reset")

        self.horizontalLayout_66.addWidget(self.test_image_scale_reset)

        self.label_94 = QLabel(self.frame_77)
        self.label_94.setObjectName(u"label_94")

        self.horizontalLayout_66.addWidget(self.label_94)

        self.test_sort_type = QComboBox(self.frame_77)
        self.test_sort_type.setObjectName(u"test_sort_type")

        self.horizontalLayout_66.addWidget(self.test_sort_type)

        self.test_sort_direction = QComboBox(self.frame_77)
        self.test_sort_direction.setObjectName(u"test_sort_direction")

        self.horizontalLayout_66.addWidget(self.test_sort_direction)


        self.verticalLayout_52.addWidget(self.frame_77)

        self.centralwidget_pages.addWidget(self.central_test_page)
        self.central_review_page = QWidget()
        self.central_review_page.setObjectName(u"central_review_page")
        self.verticalLayout_53 = QVBoxLayout(self.central_review_page)
        self.verticalLayout_53.setObjectName(u"verticalLayout_53")
        self.splitter = QSplitter(self.central_review_page)
        self.splitter.setObjectName(u"splitter")
        self.splitter.setOrientation(Qt.Horizontal)
        self.frame_20 = QFrame(self.splitter)
        self.frame_20.setObjectName(u"frame_20")
        self.frame_20.setFrameShape(QFrame.StyledPanel)
        self.frame_20.setFrameShadow(QFrame.Raised)
        self.verticalLayout_63 = QVBoxLayout(self.frame_20)
        self.verticalLayout_63.setObjectName(u"verticalLayout_63")
        self.frame_91 = QFrame(self.frame_20)
        self.frame_91.setObjectName(u"frame_91")
        self.frame_91.setFrameShape(QFrame.Panel)
        self.frame_91.setFrameShadow(QFrame.Raised)
        self.frame_91.setLineWidth(2)
        self.horizontalLayout_21 = QHBoxLayout(self.frame_91)
        self.horizontalLayout_21.setObjectName(u"horizontalLayout_21")
        self.horizontalLayout_21.setContentsMargins(0, 0, 0, 0)
        self.label_23 = QLabel(self.frame_91)
        self.label_23.setObjectName(u"label_23")
        font4 = QFont()
        font4.setPointSize(14)
        self.label_23.setFont(font4)
        self.label_23.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_21.addWidget(self.label_23)


        self.verticalLayout_63.addWidget(self.frame_91)

        self.scrollArea_14 = QScrollArea(self.frame_20)
        self.scrollArea_14.setObjectName(u"scrollArea_14")
        self.scrollArea_14.setFrameShape(QFrame.Panel)
        self.scrollArea_14.setWidgetResizable(True)
        self.review_images_widget = QWidget()
        self.review_images_widget.setObjectName(u"review_images_widget")
        self.review_images_widget.setGeometry(QRect(0, 0, 686, 907))
        self.review_images_layout = QVBoxLayout(self.review_images_widget)
        self.review_images_layout.setObjectName(u"review_images_layout")
        self.scrollArea_14.setWidget(self.review_images_widget)

        self.verticalLayout_63.addWidget(self.scrollArea_14)

        self.splitter.addWidget(self.frame_20)
        self.frame_45 = QFrame(self.splitter)
        self.frame_45.setObjectName(u"frame_45")
        self.frame_45.setFrameShape(QFrame.StyledPanel)
        self.frame_45.setFrameShadow(QFrame.Raised)
        self.verticalLayout_64 = QVBoxLayout(self.frame_45)
        self.verticalLayout_64.setObjectName(u"verticalLayout_64")
        self.frame_92 = QFrame(self.frame_45)
        self.frame_92.setObjectName(u"frame_92")
        self.frame_92.setFrameShape(QFrame.Panel)
        self.frame_92.setFrameShadow(QFrame.Raised)
        self.frame_92.setLineWidth(2)
        self.horizontalLayout_72 = QHBoxLayout(self.frame_92)
        self.horizontalLayout_72.setObjectName(u"horizontalLayout_72")
        self.horizontalLayout_72.setContentsMargins(0, 0, 0, 0)
        self.label_25 = QLabel(self.frame_92)
        self.label_25.setObjectName(u"label_25")
        self.label_25.setFont(font4)
        self.label_25.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_72.addWidget(self.label_25)


        self.verticalLayout_64.addWidget(self.frame_92)

        self.scrollArea_15 = QScrollArea(self.frame_45)
        self.scrollArea_15.setObjectName(u"scrollArea_15")
        self.scrollArea_15.setFrameShape(QFrame.Panel)
        self.scrollArea_15.setWidgetResizable(True)
        self.review_graphs_widget = QWidget()
        self.review_graphs_widget.setObjectName(u"review_graphs_widget")
        self.review_graphs_widget.setGeometry(QRect(0, 0, 532, 907))
        self.review_graphs_layout = QVBoxLayout(self.review_graphs_widget)
        self.review_graphs_layout.setObjectName(u"review_graphs_layout")
        self.scrollArea_15.setWidget(self.review_graphs_widget)

        self.verticalLayout_64.addWidget(self.scrollArea_15)

        self.splitter.addWidget(self.frame_45)

        self.verticalLayout_53.addWidget(self.splitter)

        self.centralwidget_pages.addWidget(self.central_review_page)
        self.central_utils_page = QWidget()
        self.central_utils_page.setObjectName(u"central_utils_page")
        self.verticalLayout_51 = QVBoxLayout(self.central_utils_page)
        self.verticalLayout_51.setSpacing(0)
        self.verticalLayout_51.setObjectName(u"verticalLayout_51")
        self.verticalLayout_51.setContentsMargins(0, 0, 0, 0)
        self.scrollArea_16 = QScrollArea(self.central_utils_page)
        self.scrollArea_16.setObjectName(u"scrollArea_16")
        self.scrollArea_16.setWidgetResizable(True)
        self.scrollAreaWidgetContents_2 = QWidget()
        self.scrollAreaWidgetContents_2.setObjectName(u"scrollAreaWidgetContents_2")
        self.scrollAreaWidgetContents_2.setGeometry(QRect(0, 0, 303, 131))
        self.verticalLayout_67 = QVBoxLayout(self.scrollAreaWidgetContents_2)
        self.verticalLayout_67.setObjectName(u"verticalLayout_67")
        self.splitter_3 = QSplitter(self.scrollAreaWidgetContents_2)
        self.splitter_3.setObjectName(u"splitter_3")
        self.splitter_3.setOrientation(Qt.Horizontal)
        self.frame_111 = QFrame(self.splitter_3)
        self.frame_111.setObjectName(u"frame_111")
        self.frame_111.setFrameShape(QFrame.StyledPanel)
        self.frame_111.setFrameShadow(QFrame.Raised)
        self.verticalLayout_65 = QVBoxLayout(self.frame_111)
        self.verticalLayout_65.setObjectName(u"verticalLayout_65")
        self.verticalLayout_65.setContentsMargins(2, 2, 2, 2)
        self.label_121 = QLabel(self.frame_111)
        self.label_121.setObjectName(u"label_121")
        sizePolicy2.setHeightForWidth(self.label_121.sizePolicy().hasHeightForWidth())
        self.label_121.setSizePolicy(sizePolicy2)
        self.label_121.setFont(font4)
        self.label_121.setFrameShape(QFrame.WinPanel)
        self.label_121.setFrameShadow(QFrame.Raised)
        self.label_121.setAlignment(Qt.AlignCenter)

        self.verticalLayout_65.addWidget(self.label_121)

        self.file_change_log = QTextEdit(self.frame_111)
        self.file_change_log.setObjectName(u"file_change_log")
        self.file_change_log.setReadOnly(True)
        self.file_change_log.setTextInteractionFlags(Qt.NoTextInteraction)

        self.verticalLayout_65.addWidget(self.file_change_log)

        self.splitter_3.addWidget(self.frame_111)
        self.frame_112 = QFrame(self.splitter_3)
        self.frame_112.setObjectName(u"frame_112")
        self.frame_112.setFrameShape(QFrame.StyledPanel)
        self.frame_112.setFrameShadow(QFrame.Raised)
        self.verticalLayout_66 = QVBoxLayout(self.frame_112)
        self.verticalLayout_66.setObjectName(u"verticalLayout_66")
        self.verticalLayout_66.setContentsMargins(2, 2, 2, 2)
        self.label_122 = QLabel(self.frame_112)
        self.label_122.setObjectName(u"label_122")
        sizePolicy2.setHeightForWidth(self.label_122.sizePolicy().hasHeightForWidth())
        self.label_122.setSizePolicy(sizePolicy2)
        self.label_122.setFont(font4)
        self.label_122.setFrameShape(QFrame.WinPanel)
        self.label_122.setFrameShadow(QFrame.Raised)
        self.label_122.setAlignment(Qt.AlignCenter)

        self.verticalLayout_66.addWidget(self.label_122)

        self.test_onnx_vis_frame = QFrame(self.frame_112)
        self.test_onnx_vis_frame.setObjectName(u"test_onnx_vis_frame")
        self.test_onnx_vis_frame.setFrameShape(QFrame.StyledPanel)
        self.test_onnx_vis_frame.setFrameShadow(QFrame.Sunken)
        self.test_onnx_vis_layout = QVBoxLayout(self.test_onnx_vis_frame)
        self.test_onnx_vis_layout.setSpacing(0)
        self.test_onnx_vis_layout.setObjectName(u"test_onnx_vis_layout")
        self.test_onnx_vis_layout.setContentsMargins(0, 0, 0, 0)

        self.verticalLayout_66.addWidget(self.test_onnx_vis_frame)

        self.splitter_3.addWidget(self.frame_112)

        self.verticalLayout_67.addWidget(self.splitter_3)

        self.scrollArea_16.setWidget(self.scrollAreaWidgetContents_2)

        self.verticalLayout_51.addWidget(self.scrollArea_16)

        self.centralwidget_pages.addWidget(self.central_utils_page)

        self.horizontalLayout.addWidget(self.centralwidget_pages)


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
        sizePolicy.setHeightForWidth(self.dockWidgetContents_2.sizePolicy().hasHeightForWidth())
        self.dockWidgetContents_2.setSizePolicy(sizePolicy)
        self.verticalLayout_7 = QVBoxLayout(self.dockWidgetContents_2)
        self.verticalLayout_7.setSpacing(0)
        self.verticalLayout_7.setObjectName(u"verticalLayout_7")
        self.verticalLayout_7.setContentsMargins(0, 0, 0, 0)
        self.frame_24 = QFrame(self.dockWidgetContents_2)
        self.frame_24.setObjectName(u"frame_24")
        sizePolicy6 = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Ignored)
        sizePolicy6.setHorizontalStretch(0)
        sizePolicy6.setVerticalStretch(0)
        sizePolicy6.setHeightForWidth(self.frame_24.sizePolicy().hasHeightForWidth())
        self.frame_24.setSizePolicy(sizePolicy6)
        self.frame_24.setFrameShape(QFrame.StyledPanel)
        self.frame_24.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_32 = QHBoxLayout(self.frame_24)
        self.horizontalLayout_32.setSpacing(0)
        self.horizontalLayout_32.setObjectName(u"horizontalLayout_32")
        self.horizontalLayout_32.setContentsMargins(0, 0, 0, 0)
        self.times_group = QGroupBox(self.frame_24)
        self.times_group.setObjectName(u"times_group")
        sizePolicy7 = QSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Ignored)
        sizePolicy7.setHorizontalStretch(0)
        sizePolicy7.setVerticalStretch(0)
        sizePolicy7.setHeightForWidth(self.times_group.sizePolicy().hasHeightForWidth())
        self.times_group.setSizePolicy(sizePolicy7)
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
        sizePolicy8 = QSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
        sizePolicy8.setHorizontalStretch(0)
        sizePolicy8.setVerticalStretch(0)
        sizePolicy8.setHeightForWidth(self.label_55.sizePolicy().hasHeightForWidth())
        self.label_55.setSizePolicy(sizePolicy8)
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
        sizePolicy9 = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        sizePolicy9.setHorizontalStretch(0)
        sizePolicy9.setVerticalStretch(0)
        sizePolicy9.setHeightForWidth(self.label_56.sizePolicy().hasHeightForWidth())
        self.label_56.setSizePolicy(sizePolicy9)

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
        sizePolicy8.setHeightForWidth(self.label_48.sizePolicy().hasHeightForWidth())
        self.label_48.setSizePolicy(sizePolicy8)
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
        sizePolicy9.setHeightForWidth(self.label_47.sizePolicy().hasHeightForWidth())
        self.label_47.setSizePolicy(sizePolicy9)

        self.formLayout_17.setWidget(0, QFormLayout.ItemRole.LabelRole, self.label_47)

        self.performance_epoch_fastest = QLabel(self.frame_30)
        self.performance_epoch_fastest.setObjectName(u"performance_epoch_fastest")

        self.formLayout_17.setWidget(0, QFormLayout.ItemRole.FieldRole, self.performance_epoch_fastest)

        self.label_50 = QLabel(self.frame_30)
        self.label_50.setObjectName(u"label_50")
        sizePolicy9.setHeightForWidth(self.label_50.sizePolicy().hasHeightForWidth())
        self.label_50.setSizePolicy(sizePolicy9)

        self.formLayout_17.setWidget(1, QFormLayout.ItemRole.LabelRole, self.label_50)

        self.performance_epoch_slowest = QLabel(self.frame_30)
        self.performance_epoch_slowest.setObjectName(u"performance_epoch_slowest")

        self.formLayout_17.setWidget(1, QFormLayout.ItemRole.FieldRole, self.performance_epoch_slowest)

        self.label_51 = QLabel(self.frame_30)
        self.label_51.setObjectName(u"label_51")
        sizePolicy9.setHeightForWidth(self.label_51.sizePolicy().hasHeightForWidth())
        self.label_51.setSizePolicy(sizePolicy9)

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
        sizePolicy8.setHeightForWidth(self.label_49.sizePolicy().hasHeightForWidth())
        self.label_49.setSizePolicy(sizePolicy8)
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
        sizePolicy9.setHeightForWidth(self.label_52.sizePolicy().hasHeightForWidth())
        self.label_52.setSizePolicy(sizePolicy9)

        self.formLayout_18.setWidget(0, QFormLayout.ItemRole.LabelRole, self.label_52)

        self.performance_iter_fastest = QLabel(self.frame_32)
        self.performance_iter_fastest.setObjectName(u"performance_iter_fastest")

        self.formLayout_18.setWidget(0, QFormLayout.ItemRole.FieldRole, self.performance_iter_fastest)

        self.label_53 = QLabel(self.frame_32)
        self.label_53.setObjectName(u"label_53")
        sizePolicy9.setHeightForWidth(self.label_53.sizePolicy().hasHeightForWidth())
        self.label_53.setSizePolicy(sizePolicy9)

        self.formLayout_18.setWidget(1, QFormLayout.ItemRole.LabelRole, self.label_53)

        self.performance_iter_slowest = QLabel(self.frame_32)
        self.performance_iter_slowest.setObjectName(u"performance_iter_slowest")

        self.formLayout_18.setWidget(1, QFormLayout.ItemRole.FieldRole, self.performance_iter_slowest)

        self.label_54 = QLabel(self.frame_32)
        self.label_54.setObjectName(u"label_54")
        sizePolicy9.setHeightForWidth(self.label_54.sizePolicy().hasHeightForWidth())
        self.label_54.setSizePolicy(sizePolicy9)

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
        sizePolicy1.setHeightForWidth(self.perf_graph_time_label.sizePolicy().hasHeightForWidth())
        self.perf_graph_time_label.setSizePolicy(sizePolicy1)
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
        sizePolicy6.setHeightForWidth(self.output_log_group.sizePolicy().hasHeightForWidth())
        self.output_log_group.setSizePolicy(sizePolicy6)
        self.verticalLayout_9 = QVBoxLayout(self.output_log_group)
        self.verticalLayout_9.setSpacing(2)
        self.verticalLayout_9.setObjectName(u"verticalLayout_9")
        self.verticalLayout_9.setContentsMargins(3, 3, 3, 3)
        self.log_output_frozen_header = QLabel(self.output_log_group)
        self.log_output_frozen_header.setObjectName(u"log_output_frozen_header")
        font5 = QFont()
        font5.setBold(False)
        font5.setItalic(True)
        self.log_output_frozen_header.setFont(font5)
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

        self.line_7 = QFrame(self.frame_15)
        self.line_7.setObjectName(u"line_7")
        self.line_7.setFrameShadow(QFrame.Raised)
        self.line_7.setFrameShape(QFrame.Shape.HLine)

        self.verticalLayout_23.addWidget(self.line_7)

        self.autoscroll_log = QCheckBox(self.frame_15)
        self.autoscroll_log.setObjectName(u"autoscroll_log")
        sizePolicy8.setHeightForWidth(self.autoscroll_log.sizePolicy().hasHeightForWidth())
        self.autoscroll_log.setSizePolicy(sizePolicy8)
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
        sizePolicy5.setHeightForWidth(self.epoch_progress.sizePolicy().hasHeightForWidth())
        self.epoch_progress.setSizePolicy(sizePolicy5)
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
        sizePolicy5.setHeightForWidth(self.train_progress.sizePolicy().hasHeightForWidth())
        self.train_progress.setSizePolicy(sizePolicy5)
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
        sizePolicy1.setHeightForWidth(self.settings_btns_frame.sizePolicy().hasHeightForWidth())
        self.settings_btns_frame.setSizePolicy(sizePolicy1)
        self.settings_btns_frame.setFrameShape(QFrame.Panel)
        self.settings_btns_frame.setFrameShadow(QFrame.Raised)
        self.settings_btns_frame.setLineWidth(1)
        self.verticalLayout_4 = QVBoxLayout(self.settings_btns_frame)
        self.verticalLayout_4.setSpacing(12)
        self.verticalLayout_4.setObjectName(u"verticalLayout_4")
        self.verticalLayout_4.setContentsMargins(4, 4, 4, 4)
        self.settings_dock_main_icon = QLabel(self.settings_btns_frame)
        self.settings_dock_main_icon.setObjectName(u"settings_dock_main_icon")
        sizePolicy.setHeightForWidth(self.settings_dock_main_icon.sizePolicy().hasHeightForWidth())
        self.settings_dock_main_icon.setSizePolicy(sizePolicy)
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
        font8 = QFont()
        font8.setPointSize(16)
        self.experiment_settings_label.setFont(font8)
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
        self.dataset_settings_label.setFont(font8)
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
        self.training_settings_label.setFont(font8)
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
        self.testing_settings_label.setFont(font8)
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
        self.review_settings_label.setFont(font8)
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
        self.utils_settings_label.setFont(font8)
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
        self.settings_settings_label.setFont(font8)
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
        font9 = QFont()
        font9.setPointSize(8)
        self.title_tag.setFont(font9)
        self.title_tag.setAlignment(Qt.AlignCenter)

        self.verticalLayout_40.addWidget(self.title_tag)

        self.creator_tag = QLabel(self.frame_49)
        self.creator_tag.setObjectName(u"creator_tag")
        sizePolicy2.setHeightForWidth(self.creator_tag.sizePolicy().hasHeightForWidth())
        self.creator_tag.setSizePolicy(sizePolicy2)
        font10 = QFont()
        font10.setPointSize(6)
        self.creator_tag.setFont(font10)
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
        sizePolicy1.setHeightForWidth(self.settings_section_header_icon.sizePolicy().hasHeightForWidth())
        self.settings_section_header_icon.setSizePolicy(sizePolicy1)

        self.horizontalLayout_20.addWidget(self.settings_section_header_icon)

        self.settings_section_header = QLabel(self.settings_section_header_frame)
        self.settings_section_header.setObjectName(u"settings_section_header")
        self.settings_section_header.setMinimumSize(QSize(0, 80))
        font11 = QFont()
        font11.setPointSize(22)
        font11.setBold(False)
        font11.setItalic(False)
        self.settings_section_header.setFont(font11)

        self.horizontalLayout_20.addWidget(self.settings_section_header)


        self.verticalLayout_2.addWidget(self.settings_section_header_frame)

        self.train_settings = QFrame(self.frame_2)
        self.train_settings.setObjectName(u"train_settings")
        self.train_settings.setFont(font2)
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
        self.experiment_settings_container.setGeometry(QRect(0, 0, 310, 417))
        self.verticalLayout_11 = QVBoxLayout(self.experiment_settings_container)
        self.verticalLayout_11.setSpacing(6)
        self.verticalLayout_11.setObjectName(u"verticalLayout_11")
        self.verticalLayout_11.setContentsMargins(0, 0, 0, 0)
        self.frame_87 = QFrame(self.experiment_settings_container)
        self.frame_87.setObjectName(u"frame_87")
        self.frame_87.setFrameShape(QFrame.StyledPanel)
        self.frame_87.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_76 = QHBoxLayout(self.frame_87)
        self.horizontalLayout_76.setObjectName(u"horizontalLayout_76")
        self.horizontalLayout_76.setContentsMargins(0, 0, 0, 0)
        self.line_34 = QFrame(self.frame_87)
        self.line_34.setObjectName(u"line_34")
        self.line_34.setFrameShadow(QFrame.Raised)
        self.line_34.setFrameShape(QFrame.Shape.HLine)

        self.horizontalLayout_76.addWidget(self.line_34)

        self.label_104 = QLabel(self.frame_87)
        self.label_104.setObjectName(u"label_104")
        sizePolicy1.setHeightForWidth(self.label_104.sizePolicy().hasHeightForWidth())
        self.label_104.setSizePolicy(sizePolicy1)
        self.label_104.setFont(font4)
        self.label_104.setFrameShape(QFrame.NoFrame)
        self.label_104.setFrameShadow(QFrame.Raised)

        self.horizontalLayout_76.addWidget(self.label_104)

        self.line_35 = QFrame(self.frame_87)
        self.line_35.setObjectName(u"line_35")
        self.line_35.setFrameShadow(QFrame.Raised)
        self.line_35.setFrameShape(QFrame.Shape.HLine)

        self.horizontalLayout_76.addWidget(self.line_35)


        self.verticalLayout_11.addWidget(self.frame_87)

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

        self.frame_88 = QFrame(self.experiment_settings_container)
        self.frame_88.setObjectName(u"frame_88")
        self.frame_88.setFrameShape(QFrame.StyledPanel)
        self.frame_88.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_77 = QHBoxLayout(self.frame_88)
        self.horizontalLayout_77.setObjectName(u"horizontalLayout_77")
        self.horizontalLayout_77.setContentsMargins(0, 0, 0, 0)
        self.line_36 = QFrame(self.frame_88)
        self.line_36.setObjectName(u"line_36")
        self.line_36.setFrameShadow(QFrame.Raised)
        self.line_36.setFrameShape(QFrame.Shape.HLine)

        self.horizontalLayout_77.addWidget(self.line_36)

        self.label_105 = QLabel(self.frame_88)
        self.label_105.setObjectName(u"label_105")
        sizePolicy1.setHeightForWidth(self.label_105.sizePolicy().hasHeightForWidth())
        self.label_105.setSizePolicy(sizePolicy1)
        self.label_105.setFont(font4)
        self.label_105.setFrameShape(QFrame.NoFrame)
        self.label_105.setFrameShadow(QFrame.Raised)

        self.horizontalLayout_77.addWidget(self.label_105)

        self.line_37 = QFrame(self.frame_88)
        self.line_37.setObjectName(u"line_37")
        self.line_37.setFrameShadow(QFrame.Raised)
        self.line_37.setFrameShape(QFrame.Shape.HLine)

        self.horizontalLayout_77.addWidget(self.line_37)


        self.verticalLayout_11.addWidget(self.frame_88)

        self.frame_75 = QFrame(self.experiment_settings_container)
        self.frame_75.setObjectName(u"frame_75")
        self.frame_75.setFrameShape(QFrame.StyledPanel)
        self.frame_75.setFrameShadow(QFrame.Raised)
        self.formLayout_22 = QFormLayout(self.frame_75)
        self.formLayout_22.setObjectName(u"formLayout_22")
        self.formLayout_22.setContentsMargins(0, 0, 0, 0)
        self.label_87 = QLabel(self.frame_75)
        self.label_87.setObjectName(u"label_87")

        self.formLayout_22.setWidget(3, QFormLayout.ItemRole.LabelRole, self.label_87)

        self.gen_upsample_type = QComboBox(self.frame_75)
        self.gen_upsample_type.setObjectName(u"gen_upsample_type")

        self.formLayout_22.setWidget(3, QFormLayout.ItemRole.FieldRole, self.gen_upsample_type)

        self.label_124 = QLabel(self.frame_75)
        self.label_124.setObjectName(u"label_124")

        self.formLayout_22.setWidget(0, QFormLayout.ItemRole.LabelRole, self.label_124)

        self.gen_features = QSpinBox(self.frame_75)
        self.gen_features.setObjectName(u"gen_features")
        self.gen_features.setMinimum(16)
        self.gen_features.setMaximum(512)
        self.gen_features.setSingleStep(16)
        self.gen_features.setValue(64)

        self.formLayout_22.setWidget(0, QFormLayout.ItemRole.FieldRole, self.gen_features)

        self.label_125 = QLabel(self.frame_75)
        self.label_125.setObjectName(u"label_125")

        self.formLayout_22.setWidget(1, QFormLayout.ItemRole.LabelRole, self.label_125)

        self.gen_n_downs = QSpinBox(self.frame_75)
        self.gen_n_downs.setObjectName(u"gen_n_downs")
        self.gen_n_downs.setMinimum(1)
        self.gen_n_downs.setMaximum(7)
        self.gen_n_downs.setValue(6)

        self.formLayout_22.setWidget(1, QFormLayout.ItemRole.FieldRole, self.gen_n_downs)

        self.label_126 = QLabel(self.frame_75)
        self.label_126.setObjectName(u"label_126")

        self.formLayout_22.setWidget(2, QFormLayout.ItemRole.LabelRole, self.label_126)

        self.gen_block_type = QComboBox(self.frame_75)
        self.gen_block_type.setObjectName(u"gen_block_type")

        self.formLayout_22.setWidget(2, QFormLayout.ItemRole.FieldRole, self.gen_block_type)


        self.verticalLayout_11.addWidget(self.frame_75)

        self.frame_86 = QFrame(self.experiment_settings_container)
        self.frame_86.setObjectName(u"frame_86")
        self.frame_86.setFrameShape(QFrame.StyledPanel)
        self.frame_86.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_75 = QHBoxLayout(self.frame_86)
        self.horizontalLayout_75.setObjectName(u"horizontalLayout_75")
        self.horizontalLayout_75.setContentsMargins(0, 0, 0, 0)
        self.line_33 = QFrame(self.frame_86)
        self.line_33.setObjectName(u"line_33")
        self.line_33.setFrameShadow(QFrame.Raised)
        self.line_33.setFrameShape(QFrame.Shape.HLine)

        self.horizontalLayout_75.addWidget(self.line_33)

        self.label_103 = QLabel(self.frame_86)
        self.label_103.setObjectName(u"label_103")
        sizePolicy1.setHeightForWidth(self.label_103.sizePolicy().hasHeightForWidth())
        self.label_103.setSizePolicy(sizePolicy1)
        self.label_103.setFont(font4)
        self.label_103.setFrameShape(QFrame.NoFrame)
        self.label_103.setFrameShadow(QFrame.Raised)

        self.horizontalLayout_75.addWidget(self.label_103)

        self.line_29 = QFrame(self.frame_86)
        self.line_29.setObjectName(u"line_29")
        self.line_29.setFrameShadow(QFrame.Raised)
        self.line_29.setFrameShape(QFrame.Shape.HLine)

        self.horizontalLayout_75.addWidget(self.line_29)


        self.verticalLayout_11.addWidget(self.frame_86)

        self.frame_71 = QFrame(self.experiment_settings_container)
        self.frame_71.setObjectName(u"frame_71")
        self.frame_71.setFrameShape(QFrame.StyledPanel)
        self.frame_71.setFrameShadow(QFrame.Raised)
        self.formLayout_21 = QFormLayout(self.frame_71)
        self.formLayout_21.setObjectName(u"formLayout_21")
        self.formLayout_21.setHorizontalSpacing(6)
        self.formLayout_21.setVerticalSpacing(6)
        self.formLayout_21.setContentsMargins(0, 0, 0, 0)
        self.label_84 = QLabel(self.frame_71)
        self.label_84.setObjectName(u"label_84")

        self.formLayout_21.setWidget(0, QFormLayout.ItemRole.LabelRole, self.label_84)

        self.frame_72 = QFrame(self.frame_71)
        self.frame_72.setObjectName(u"frame_72")
        self.frame_72.setFrameShape(QFrame.StyledPanel)
        self.frame_72.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_60 = QHBoxLayout(self.frame_72)
        self.horizontalLayout_60.setObjectName(u"horizontalLayout_60")
        self.horizontalLayout_60.setContentsMargins(0, 0, 0, 0)
        self.n_layers_d_slider = QSlider(self.frame_72)
        self.n_layers_d_slider.setObjectName(u"n_layers_d_slider")
        self.n_layers_d_slider.setMinimum(1)
        self.n_layers_d_slider.setMaximum(7)
        self.n_layers_d_slider.setValue(3)
        self.n_layers_d_slider.setOrientation(Qt.Horizontal)
        self.n_layers_d_slider.setTickPosition(QSlider.NoTicks)
        self.n_layers_d_slider.setTickInterval(10)

        self.horizontalLayout_60.addWidget(self.n_layers_d_slider)

        self.n_layers_d = QSpinBox(self.frame_72)
        self.n_layers_d.setObjectName(u"n_layers_d")
        self.n_layers_d.setMinimum(1)
        self.n_layers_d.setMaximum(7)
        self.n_layers_d.setValue(3)

        self.horizontalLayout_60.addWidget(self.n_layers_d)


        self.formLayout_21.setWidget(0, QFormLayout.ItemRole.FieldRole, self.frame_72)

        self.label_85 = QLabel(self.frame_71)
        self.label_85.setObjectName(u"label_85")

        self.formLayout_21.setWidget(1, QFormLayout.ItemRole.LabelRole, self.label_85)

        self.label_86 = QLabel(self.frame_71)
        self.label_86.setObjectName(u"label_86")

        self.formLayout_21.setWidget(2, QFormLayout.ItemRole.LabelRole, self.label_86)

        self.frame_73 = QFrame(self.frame_71)
        self.frame_73.setObjectName(u"frame_73")
        self.frame_73.setFrameShape(QFrame.StyledPanel)
        self.frame_73.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_61 = QHBoxLayout(self.frame_73)
        self.horizontalLayout_61.setObjectName(u"horizontalLayout_61")
        self.horizontalLayout_61.setContentsMargins(0, 0, 0, 0)
        self.base_channels_d_slider = QSlider(self.frame_73)
        self.base_channels_d_slider.setObjectName(u"base_channels_d_slider")
        self.base_channels_d_slider.setMinimum(16)
        self.base_channels_d_slider.setMaximum(256)
        self.base_channels_d_slider.setSingleStep(16)
        self.base_channels_d_slider.setValue(64)
        self.base_channels_d_slider.setOrientation(Qt.Horizontal)
        self.base_channels_d_slider.setTickPosition(QSlider.NoTicks)
        self.base_channels_d_slider.setTickInterval(10)

        self.horizontalLayout_61.addWidget(self.base_channels_d_slider)

        self.base_channels_d = QSpinBox(self.frame_73)
        self.base_channels_d.setObjectName(u"base_channels_d")
        self.base_channels_d.setMinimum(16)
        self.base_channels_d.setMaximum(256)
        self.base_channels_d.setSingleStep(16)
        self.base_channels_d.setValue(64)

        self.horizontalLayout_61.addWidget(self.base_channels_d)


        self.formLayout_21.setWidget(1, QFormLayout.ItemRole.FieldRole, self.frame_73)

        self.frame_74 = QFrame(self.frame_71)
        self.frame_74.setObjectName(u"frame_74")
        self.frame_74.setFrameShape(QFrame.StyledPanel)
        self.frame_74.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_62 = QHBoxLayout(self.frame_74)
        self.horizontalLayout_62.setObjectName(u"horizontalLayout_62")
        self.horizontalLayout_62.setContentsMargins(0, 0, 0, 0)
        self.max_channels_d_slider = QSlider(self.frame_74)
        self.max_channels_d_slider.setObjectName(u"max_channels_d_slider")
        self.max_channels_d_slider.setMinimum(32)
        self.max_channels_d_slider.setMaximum(2048)
        self.max_channels_d_slider.setSingleStep(32)
        self.max_channels_d_slider.setValue(512)
        self.max_channels_d_slider.setOrientation(Qt.Horizontal)
        self.max_channels_d_slider.setTickPosition(QSlider.NoTicks)
        self.max_channels_d_slider.setTickInterval(10)

        self.horizontalLayout_62.addWidget(self.max_channels_d_slider)

        self.max_channels_d = QSpinBox(self.frame_74)
        self.max_channels_d.setObjectName(u"max_channels_d")
        self.max_channels_d.setMinimum(32)
        self.max_channels_d.setMaximum(2048)
        self.max_channels_d.setSingleStep(32)
        self.max_channels_d.setValue(512)

        self.horizontalLayout_62.addWidget(self.max_channels_d)


        self.formLayout_21.setWidget(2, QFormLayout.ItemRole.FieldRole, self.frame_74)


        self.verticalLayout_11.addWidget(self.frame_71)

        self.verticalSpacer_8 = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.verticalLayout_11.addItem(self.verticalSpacer_8)

        self.scrollArea.setWidget(self.experiment_settings_container)

        self.verticalLayout_3.addWidget(self.scrollArea)

        self.frame_50 = QFrame(self.experiment_page)
        self.frame_50.setObjectName(u"frame_50")
        self.frame_50.setFrameShape(QFrame.Panel)
        self.frame_50.setFrameShadow(QFrame.Sunken)
        self.verticalLayout_59 = QVBoxLayout(self.frame_50)
        self.verticalLayout_59.setObjectName(u"verticalLayout_59")
        self.verticalLayout_59.setContentsMargins(0, 0, 0, 0)
        self.load_from_config = QPushButton(self.frame_50)
        self.load_from_config.setObjectName(u"load_from_config")

        self.verticalLayout_59.addWidget(self.load_from_config)


        self.verticalLayout_3.addWidget(self.frame_50)

        self.settings_pages.addWidget(self.experiment_page)
        self.dataset_page = QWidget()
        self.dataset_page.setObjectName(u"dataset_page")
        self.verticalLayout_5 = QVBoxLayout(self.dataset_page)
        self.verticalLayout_5.setObjectName(u"verticalLayout_5")
        self.verticalLayout_5.setContentsMargins(0, 0, 0, 0)
        self.scrollArea_2 = QScrollArea(self.dataset_page)
        self.scrollArea_2.setObjectName(u"scrollArea_2")
        self.scrollArea_2.setFrameShadow(QFrame.Raised)
        self.scrollArea_2.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scrollArea_2.setWidgetResizable(True)
        self.dataloader_settings_container = QWidget()
        self.dataloader_settings_container.setObjectName(u"dataloader_settings_container")
        self.dataloader_settings_container.setGeometry(QRect(0, 0, 437, 1245))
        self.verticalLayout_10 = QVBoxLayout(self.dataloader_settings_container)
        self.verticalLayout_10.setObjectName(u"verticalLayout_10")
        self.verticalLayout_10.setContentsMargins(0, 0, 0, 0)
        self.frame_84 = QFrame(self.dataloader_settings_container)
        self.frame_84.setObjectName(u"frame_84")
        self.frame_84.setFrameShape(QFrame.StyledPanel)
        self.frame_84.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_73 = QHBoxLayout(self.frame_84)
        self.horizontalLayout_73.setObjectName(u"horizontalLayout_73")
        self.horizontalLayout_73.setContentsMargins(0, 0, 0, 0)
        self.line_31 = QFrame(self.frame_84)
        self.line_31.setObjectName(u"line_31")
        self.line_31.setFrameShadow(QFrame.Raised)
        self.line_31.setFrameShape(QFrame.Shape.HLine)

        self.horizontalLayout_73.addWidget(self.line_31)

        self.label_101 = QLabel(self.frame_84)
        self.label_101.setObjectName(u"label_101")
        sizePolicy1.setHeightForWidth(self.label_101.sizePolicy().hasHeightForWidth())
        self.label_101.setSizePolicy(sizePolicy1)
        self.label_101.setFont(font4)
        self.label_101.setFrameShape(QFrame.NoFrame)
        self.label_101.setFrameShadow(QFrame.Raised)

        self.horizontalLayout_73.addWidget(self.label_101)

        self.line_27 = QFrame(self.frame_84)
        self.line_27.setObjectName(u"line_27")
        self.line_27.setFrameShadow(QFrame.Raised)
        self.line_27.setFrameShape(QFrame.Shape.HLine)

        self.horizontalLayout_73.addWidget(self.line_27)


        self.verticalLayout_10.addWidget(self.frame_84)

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

        self.formLayout_3.setWidget(1, QFormLayout.ItemRole.LabelRole, self.label)

        self.horizontalLayout_2 = QHBoxLayout()
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.dataroot = QLineEdit(self.frame_5)
        self.dataroot.setObjectName(u"dataroot")

        self.horizontalLayout_2.addWidget(self.dataroot)

        self.browse_dataset = QPushButton(self.frame_5)
        self.browse_dataset.setObjectName(u"browse_dataset")

        self.horizontalLayout_2.addWidget(self.browse_dataset)


        self.formLayout_3.setLayout(1, QFormLayout.ItemRole.FieldRole, self.horizontalLayout_2)


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
        sizePolicy5.setHeightForWidth(self.load_size.sizePolicy().hasHeightForWidth())
        self.load_size.setSizePolicy(sizePolicy5)
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
        self.crop_size = QComboBox(self.frame_18)
        self.crop_size.setObjectName(u"crop_size")

        self.formLayout_5.setWidget(0, QFormLayout.ItemRole.FieldRole, self.crop_size)

        self.label_11 = QLabel(self.frame_18)
        self.label_11.setObjectName(u"label_11")

        self.formLayout_5.setWidget(0, QFormLayout.ItemRole.LabelRole, self.label_11)


        self.horizontalLayout_9.addWidget(self.frame_18)

        self.label_9 = QLabel(self.frame_6)
        self.label_9.setObjectName(u"label_9")
        sizePolicy.setHeightForWidth(self.label_9.sizePolicy().hasHeightForWidth())
        self.label_9.setSizePolicy(sizePolicy)

        self.horizontalLayout_9.addWidget(self.label_9)

        self.input_nc = QSpinBox(self.frame_6)
        self.input_nc.setObjectName(u"input_nc")
        sizePolicy9.setHeightForWidth(self.input_nc.sizePolicy().hasHeightForWidth())
        self.input_nc.setSizePolicy(sizePolicy9)
        self.input_nc.setReadOnly(False)
        self.input_nc.setMinimum(1)
        self.input_nc.setMaximum(3)
        self.input_nc.setValue(3)

        self.horizontalLayout_9.addWidget(self.input_nc)


        self.verticalLayout_10.addWidget(self.frame_6)

        self.frame_78 = QFrame(self.dataloader_settings_container)
        self.frame_78.setObjectName(u"frame_78")
        self.frame_78.setFrameShape(QFrame.StyledPanel)
        self.frame_78.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_68 = QHBoxLayout(self.frame_78)
        self.horizontalLayout_68.setObjectName(u"horizontalLayout_68")
        self.horizontalLayout_68.setContentsMargins(0, 0, 0, 0)
        self.line_30 = QFrame(self.frame_78)
        self.line_30.setObjectName(u"line_30")
        self.line_30.setFrameShadow(QFrame.Raised)
        self.line_30.setFrameShape(QFrame.Shape.HLine)

        self.horizontalLayout_68.addWidget(self.line_30)

        self.label_95 = QLabel(self.frame_78)
        self.label_95.setObjectName(u"label_95")
        sizePolicy1.setHeightForWidth(self.label_95.sizePolicy().hasHeightForWidth())
        self.label_95.setSizePolicy(sizePolicy1)
        self.label_95.setFont(font4)
        self.label_95.setFrameShape(QFrame.NoFrame)
        self.label_95.setFrameShadow(QFrame.Raised)

        self.horizontalLayout_68.addWidget(self.label_95)

        self.line_22 = QFrame(self.frame_78)
        self.line_22.setObjectName(u"line_22")
        self.line_22.setFrameShadow(QFrame.Raised)
        self.line_22.setFrameShape(QFrame.Shape.HLine)

        self.horizontalLayout_68.addWidget(self.line_22)


        self.verticalLayout_10.addWidget(self.frame_78)

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

        self.formLayout_7.setWidget(1, QFormLayout.ItemRole.LabelRole, self.label_5)

        self.direction = QComboBox(self.frame_19)
        self.direction.setObjectName(u"direction")
        self.direction.setEditable(False)
        self.direction.setFrame(True)

        self.formLayout_7.setWidget(1, QFormLayout.ItemRole.FieldRole, self.direction)

        self.label_8 = QLabel(self.frame_19)
        self.label_8.setObjectName(u"label_8")

        self.formLayout_7.setWidget(2, QFormLayout.ItemRole.LabelRole, self.label_8)

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


        self.formLayout_7.setLayout(2, QFormLayout.ItemRole.FieldRole, self.horizontalLayout_7)


        self.verticalLayout_10.addWidget(self.frame_19)

        self.groupBox_2 = QGroupBox(self.dataloader_settings_container)
        self.groupBox_2.setObjectName(u"groupBox_2")
        self.verticalLayout_60 = QVBoxLayout(self.groupBox_2)
        self.verticalLayout_60.setSpacing(3)
        self.verticalLayout_60.setObjectName(u"verticalLayout_60")
        self.frame_52 = QFrame(self.groupBox_2)
        self.frame_52.setObjectName(u"frame_52")
        self.frame_52.setFrameShape(QFrame.StyledPanel)
        self.frame_52.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_51 = QHBoxLayout(self.frame_52)
        self.horizontalLayout_51.setObjectName(u"horizontalLayout_51")
        self.horizontalLayout_51.setContentsMargins(0, 0, 0, 0)
        self.show_input_transforms = QCheckBox(self.frame_52)
        self.show_input_transforms.setObjectName(u"show_input_transforms")
        sizePolicy8.setHeightForWidth(self.show_input_transforms.sizePolicy().hasHeightForWidth())
        self.show_input_transforms.setSizePolicy(sizePolicy8)
        font12 = QFont()
        font12.setPointSize(13)
        self.show_input_transforms.setFont(font12)

        self.horizontalLayout_51.addWidget(self.show_input_transforms)

        self.line_4 = QFrame(self.frame_52)
        self.line_4.setObjectName(u"line_4")
        self.line_4.setFrameShadow(QFrame.Raised)
        self.line_4.setFrameShape(QFrame.Shape.HLine)

        self.horizontalLayout_51.addWidget(self.line_4)


        self.verticalLayout_60.addWidget(self.frame_52)

        self.input_transforms_frame = QFrame(self.groupBox_2)
        self.input_transforms_frame.setObjectName(u"input_transforms_frame")
        self.input_transforms_frame.setFrameShape(QFrame.StyledPanel)
        self.input_transforms_frame.setFrameShadow(QFrame.Raised)
        self.formLayout_20 = QFormLayout(self.input_transforms_frame)
        self.formLayout_20.setObjectName(u"formLayout_20")
        self.formLayout_20.setHorizontalSpacing(6)
        self.formLayout_20.setVerticalSpacing(6)
        self.formLayout_20.setContentsMargins(0, 0, 0, 0)
        self.label_74 = QLabel(self.input_transforms_frame)
        self.label_74.setObjectName(u"label_74")

        self.formLayout_20.setWidget(0, QFormLayout.ItemRole.LabelRole, self.label_74)

        self.frame_66 = QFrame(self.input_transforms_frame)
        self.frame_66.setObjectName(u"frame_66")
        self.frame_66.setFrameShape(QFrame.StyledPanel)
        self.frame_66.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_56 = QHBoxLayout(self.frame_66)
        self.horizontalLayout_56.setObjectName(u"horizontalLayout_56")
        self.horizontalLayout_56.setContentsMargins(0, 0, 0, 0)
        self.colorjitter_chance_slider = QSlider(self.frame_66)
        self.colorjitter_chance_slider.setObjectName(u"colorjitter_chance_slider")
        self.colorjitter_chance_slider.setMaximum(100)
        self.colorjitter_chance_slider.setOrientation(Qt.Horizontal)
        self.colorjitter_chance_slider.setTickPosition(QSlider.NoTicks)
        self.colorjitter_chance_slider.setTickInterval(10)

        self.horizontalLayout_56.addWidget(self.colorjitter_chance_slider)

        self.colorjitter_chance = QDoubleSpinBox(self.frame_66)
        self.colorjitter_chance.setObjectName(u"colorjitter_chance")
        self.colorjitter_chance.setDecimals(1)

        self.horizontalLayout_56.addWidget(self.colorjitter_chance)


        self.formLayout_20.setWidget(0, QFormLayout.ItemRole.FieldRole, self.frame_66)

        self.label_75 = QLabel(self.input_transforms_frame)
        self.label_75.setObjectName(u"label_75")

        self.formLayout_20.setWidget(1, QFormLayout.ItemRole.LabelRole, self.label_75)

        self.frame_67 = QFrame(self.input_transforms_frame)
        self.frame_67.setObjectName(u"frame_67")
        self.frame_67.setFrameShape(QFrame.StyledPanel)
        self.frame_67.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_57 = QHBoxLayout(self.frame_67)
        self.horizontalLayout_57.setObjectName(u"horizontalLayout_57")
        self.horizontalLayout_57.setContentsMargins(0, 0, 0, 0)
        self.colorjitter_min_brightness = QDoubleSpinBox(self.frame_67)
        self.colorjitter_min_brightness.setObjectName(u"colorjitter_min_brightness")
        self.colorjitter_min_brightness.setMaximum(2.500000000000000)
        self.colorjitter_min_brightness.setSingleStep(0.050000000000000)
        self.colorjitter_min_brightness.setValue(0.800000000000000)

        self.horizontalLayout_57.addWidget(self.colorjitter_min_brightness)

        self.colorjitter_max_brightness = QDoubleSpinBox(self.frame_67)
        self.colorjitter_max_brightness.setObjectName(u"colorjitter_max_brightness")
        self.colorjitter_max_brightness.setMaximum(2.500000000000000)
        self.colorjitter_max_brightness.setSingleStep(0.050000000000000)
        self.colorjitter_max_brightness.setValue(1.200000000000000)

        self.horizontalLayout_57.addWidget(self.colorjitter_max_brightness)


        self.formLayout_20.setWidget(1, QFormLayout.ItemRole.FieldRole, self.frame_67)

        self.label_127 = QLabel(self.input_transforms_frame)
        self.label_127.setObjectName(u"label_127")

        self.formLayout_20.setWidget(2, QFormLayout.ItemRole.LabelRole, self.label_127)

        self.frame_115 = QFrame(self.input_transforms_frame)
        self.frame_115.setObjectName(u"frame_115")
        self.frame_115.setFrameShape(QFrame.StyledPanel)
        self.frame_115.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_109 = QHBoxLayout(self.frame_115)
        self.horizontalLayout_109.setObjectName(u"horizontalLayout_109")
        self.horizontalLayout_109.setContentsMargins(0, 0, 0, 0)
        self.gaussnoise_chance_slider = QSlider(self.frame_115)
        self.gaussnoise_chance_slider.setObjectName(u"gaussnoise_chance_slider")
        self.gaussnoise_chance_slider.setMaximum(100)
        self.gaussnoise_chance_slider.setOrientation(Qt.Horizontal)
        self.gaussnoise_chance_slider.setTickPosition(QSlider.NoTicks)
        self.gaussnoise_chance_slider.setTickInterval(10)

        self.horizontalLayout_109.addWidget(self.gaussnoise_chance_slider)

        self.gaussnoise_chance = QDoubleSpinBox(self.frame_115)
        self.gaussnoise_chance.setObjectName(u"gaussnoise_chance")
        self.gaussnoise_chance.setDecimals(1)

        self.horizontalLayout_109.addWidget(self.gaussnoise_chance)


        self.formLayout_20.setWidget(2, QFormLayout.ItemRole.FieldRole, self.frame_115)

        self.label_128 = QLabel(self.input_transforms_frame)
        self.label_128.setObjectName(u"label_128")

        self.formLayout_20.setWidget(3, QFormLayout.ItemRole.LabelRole, self.label_128)

        self.frame_116 = QFrame(self.input_transforms_frame)
        self.frame_116.setObjectName(u"frame_116")
        self.frame_116.setFrameShape(QFrame.StyledPanel)
        self.frame_116.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_110 = QHBoxLayout(self.frame_116)
        self.horizontalLayout_110.setObjectName(u"horizontalLayout_110")
        self.horizontalLayout_110.setContentsMargins(0, 0, 0, 0)
        self.gaussnoise_min = QDoubleSpinBox(self.frame_116)
        self.gaussnoise_min.setObjectName(u"gaussnoise_min")
        self.gaussnoise_min.setMaximum(2.500000000000000)
        self.gaussnoise_min.setSingleStep(0.050000000000000)
        self.gaussnoise_min.setValue(0.800000000000000)

        self.horizontalLayout_110.addWidget(self.gaussnoise_min)

        self.gaussnoise_max = QDoubleSpinBox(self.frame_116)
        self.gaussnoise_max.setObjectName(u"gaussnoise_max")
        self.gaussnoise_max.setMaximum(2.500000000000000)
        self.gaussnoise_max.setSingleStep(0.050000000000000)
        self.gaussnoise_max.setValue(1.200000000000000)

        self.horizontalLayout_110.addWidget(self.gaussnoise_max)


        self.formLayout_20.setWidget(3, QFormLayout.ItemRole.FieldRole, self.frame_116)

        self.label_129 = QLabel(self.input_transforms_frame)
        self.label_129.setObjectName(u"label_129")

        self.formLayout_20.setWidget(4, QFormLayout.ItemRole.LabelRole, self.label_129)

        self.frame_117 = QFrame(self.input_transforms_frame)
        self.frame_117.setObjectName(u"frame_117")
        self.frame_117.setFrameShape(QFrame.StyledPanel)
        self.frame_117.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_111 = QHBoxLayout(self.frame_117)
        self.horizontalLayout_111.setObjectName(u"horizontalLayout_111")
        self.horizontalLayout_111.setContentsMargins(0, 0, 0, 0)
        self.motionblur_chance_slider = QSlider(self.frame_117)
        self.motionblur_chance_slider.setObjectName(u"motionblur_chance_slider")
        self.motionblur_chance_slider.setMaximum(100)
        self.motionblur_chance_slider.setOrientation(Qt.Horizontal)
        self.motionblur_chance_slider.setTickPosition(QSlider.NoTicks)
        self.motionblur_chance_slider.setTickInterval(10)

        self.horizontalLayout_111.addWidget(self.motionblur_chance_slider)

        self.motionblur_chance = QDoubleSpinBox(self.frame_117)
        self.motionblur_chance.setObjectName(u"motionblur_chance")
        self.motionblur_chance.setDecimals(1)

        self.horizontalLayout_111.addWidget(self.motionblur_chance)


        self.formLayout_20.setWidget(4, QFormLayout.ItemRole.FieldRole, self.frame_117)

        self.label_130 = QLabel(self.input_transforms_frame)
        self.label_130.setObjectName(u"label_130")

        self.formLayout_20.setWidget(5, QFormLayout.ItemRole.LabelRole, self.label_130)

        self.motionblur_limit = QSpinBox(self.input_transforms_frame)
        self.motionblur_limit.setObjectName(u"motionblur_limit")
        self.motionblur_limit.setMinimum(3)
        self.motionblur_limit.setValue(7)

        self.formLayout_20.setWidget(5, QFormLayout.ItemRole.FieldRole, self.motionblur_limit)

        self.label_131 = QLabel(self.input_transforms_frame)
        self.label_131.setObjectName(u"label_131")

        self.formLayout_20.setWidget(6, QFormLayout.ItemRole.LabelRole, self.label_131)

        self.frame_118 = QFrame(self.input_transforms_frame)
        self.frame_118.setObjectName(u"frame_118")
        self.frame_118.setFrameShape(QFrame.StyledPanel)
        self.frame_118.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_112 = QHBoxLayout(self.frame_118)
        self.horizontalLayout_112.setObjectName(u"horizontalLayout_112")
        self.horizontalLayout_112.setContentsMargins(0, 0, 0, 0)
        self.randgamma_chance_slider = QSlider(self.frame_118)
        self.randgamma_chance_slider.setObjectName(u"randgamma_chance_slider")
        self.randgamma_chance_slider.setMaximum(100)
        self.randgamma_chance_slider.setOrientation(Qt.Horizontal)
        self.randgamma_chance_slider.setTickPosition(QSlider.NoTicks)
        self.randgamma_chance_slider.setTickInterval(10)

        self.horizontalLayout_112.addWidget(self.randgamma_chance_slider)

        self.randgamma_chance = QDoubleSpinBox(self.frame_118)
        self.randgamma_chance.setObjectName(u"randgamma_chance")
        self.randgamma_chance.setDecimals(1)

        self.horizontalLayout_112.addWidget(self.randgamma_chance)


        self.formLayout_20.setWidget(6, QFormLayout.ItemRole.FieldRole, self.frame_118)

        self.label_133 = QLabel(self.input_transforms_frame)
        self.label_133.setObjectName(u"label_133")

        self.formLayout_20.setWidget(7, QFormLayout.ItemRole.LabelRole, self.label_133)

        self.frame_119 = QFrame(self.input_transforms_frame)
        self.frame_119.setObjectName(u"frame_119")
        self.frame_119.setFrameShape(QFrame.StyledPanel)
        self.frame_119.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_113 = QHBoxLayout(self.frame_119)
        self.horizontalLayout_113.setObjectName(u"horizontalLayout_113")
        self.horizontalLayout_113.setContentsMargins(0, 0, 0, 0)
        self.randgamma_min = QDoubleSpinBox(self.frame_119)
        self.randgamma_min.setObjectName(u"randgamma_min")
        self.randgamma_min.setMaximum(250.000000000000000)
        self.randgamma_min.setSingleStep(5.000000000000000)
        self.randgamma_min.setValue(80.000000000000000)

        self.horizontalLayout_113.addWidget(self.randgamma_min)

        self.randgamma_max = QDoubleSpinBox(self.frame_119)
        self.randgamma_max.setObjectName(u"randgamma_max")
        self.randgamma_max.setMaximum(250.000000000000000)
        self.randgamma_max.setSingleStep(5.000000000000000)
        self.randgamma_max.setValue(120.000000000000000)

        self.horizontalLayout_113.addWidget(self.randgamma_max)


        self.formLayout_20.setWidget(7, QFormLayout.ItemRole.FieldRole, self.frame_119)

        self.label_132 = QLabel(self.input_transforms_frame)
        self.label_132.setObjectName(u"label_132")

        self.formLayout_20.setWidget(8, QFormLayout.ItemRole.LabelRole, self.label_132)

        self.frame_120 = QFrame(self.input_transforms_frame)
        self.frame_120.setObjectName(u"frame_120")
        self.frame_120.setFrameShape(QFrame.StyledPanel)
        self.frame_120.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_114 = QHBoxLayout(self.frame_120)
        self.horizontalLayout_114.setObjectName(u"horizontalLayout_114")
        self.horizontalLayout_114.setContentsMargins(0, 0, 0, 0)
        self.grayscale_chance_slider = QSlider(self.frame_120)
        self.grayscale_chance_slider.setObjectName(u"grayscale_chance_slider")
        self.grayscale_chance_slider.setMaximum(100)
        self.grayscale_chance_slider.setOrientation(Qt.Horizontal)
        self.grayscale_chance_slider.setTickPosition(QSlider.NoTicks)
        self.grayscale_chance_slider.setTickInterval(10)

        self.horizontalLayout_114.addWidget(self.grayscale_chance_slider)

        self.grayscale_chance = QDoubleSpinBox(self.frame_120)
        self.grayscale_chance.setObjectName(u"grayscale_chance")
        self.grayscale_chance.setDecimals(1)

        self.horizontalLayout_114.addWidget(self.grayscale_chance)


        self.formLayout_20.setWidget(8, QFormLayout.ItemRole.FieldRole, self.frame_120)

        self.label_134 = QLabel(self.input_transforms_frame)
        self.label_134.setObjectName(u"label_134")

        self.formLayout_20.setWidget(9, QFormLayout.ItemRole.LabelRole, self.label_134)

        self.grayscale_method = QComboBox(self.input_transforms_frame)
        self.grayscale_method.setObjectName(u"grayscale_method")

        self.formLayout_20.setWidget(9, QFormLayout.ItemRole.FieldRole, self.grayscale_method)

        self.label_135 = QLabel(self.input_transforms_frame)
        self.label_135.setObjectName(u"label_135")

        self.formLayout_20.setWidget(10, QFormLayout.ItemRole.LabelRole, self.label_135)

        self.label_136 = QLabel(self.input_transforms_frame)
        self.label_136.setObjectName(u"label_136")

        self.formLayout_20.setWidget(11, QFormLayout.ItemRole.LabelRole, self.label_136)

        self.label_137 = QLabel(self.input_transforms_frame)
        self.label_137.setObjectName(u"label_137")

        self.formLayout_20.setWidget(12, QFormLayout.ItemRole.LabelRole, self.label_137)

        self.frame_121 = QFrame(self.input_transforms_frame)
        self.frame_121.setObjectName(u"frame_121")
        self.frame_121.setFrameShape(QFrame.StyledPanel)
        self.frame_121.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_115 = QHBoxLayout(self.frame_121)
        self.horizontalLayout_115.setObjectName(u"horizontalLayout_115")
        self.horizontalLayout_115.setContentsMargins(0, 0, 0, 0)
        self.compression_chance_slider = QSlider(self.frame_121)
        self.compression_chance_slider.setObjectName(u"compression_chance_slider")
        self.compression_chance_slider.setMaximum(100)
        self.compression_chance_slider.setOrientation(Qt.Horizontal)
        self.compression_chance_slider.setTickPosition(QSlider.NoTicks)
        self.compression_chance_slider.setTickInterval(10)

        self.horizontalLayout_115.addWidget(self.compression_chance_slider)

        self.compression_chance = QDoubleSpinBox(self.frame_121)
        self.compression_chance.setObjectName(u"compression_chance")
        self.compression_chance.setDecimals(1)

        self.horizontalLayout_115.addWidget(self.compression_chance)


        self.formLayout_20.setWidget(10, QFormLayout.ItemRole.FieldRole, self.frame_121)

        self.compression_type = QComboBox(self.input_transforms_frame)
        self.compression_type.setObjectName(u"compression_type")

        self.formLayout_20.setWidget(11, QFormLayout.ItemRole.FieldRole, self.compression_type)

        self.frame_122 = QFrame(self.input_transforms_frame)
        self.frame_122.setObjectName(u"frame_122")
        self.frame_122.setFrameShape(QFrame.StyledPanel)
        self.frame_122.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_116 = QHBoxLayout(self.frame_122)
        self.horizontalLayout_116.setObjectName(u"horizontalLayout_116")
        self.horizontalLayout_116.setContentsMargins(0, 0, 0, 0)
        self.compression_quality_min = QSpinBox(self.frame_122)
        self.compression_quality_min.setObjectName(u"compression_quality_min")
        self.compression_quality_min.setMaximum(100)
        self.compression_quality_min.setValue(99)

        self.horizontalLayout_116.addWidget(self.compression_quality_min)

        self.compression_quality_max = QSpinBox(self.frame_122)
        self.compression_quality_max.setObjectName(u"compression_quality_max")
        self.compression_quality_max.setMaximum(100)
        self.compression_quality_max.setValue(100)

        self.horizontalLayout_116.addWidget(self.compression_quality_max)


        self.formLayout_20.setWidget(12, QFormLayout.ItemRole.FieldRole, self.frame_122)


        self.verticalLayout_60.addWidget(self.input_transforms_frame)

        self.line_6 = QFrame(self.groupBox_2)
        self.line_6.setObjectName(u"line_6")
        self.line_6.setMinimumSize(QSize(0, 10))
        self.line_6.setFrameShape(QFrame.Shape.HLine)
        self.line_6.setFrameShadow(QFrame.Shadow.Sunken)

        self.verticalLayout_60.addWidget(self.line_6)

        self.frame_51 = QFrame(self.groupBox_2)
        self.frame_51.setObjectName(u"frame_51")
        self.frame_51.setFrameShape(QFrame.StyledPanel)
        self.frame_51.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_30 = QHBoxLayout(self.frame_51)
        self.horizontalLayout_30.setObjectName(u"horizontalLayout_30")
        self.horizontalLayout_30.setContentsMargins(0, 0, 0, 0)
        self.show_both_transforms = QCheckBox(self.frame_51)
        self.show_both_transforms.setObjectName(u"show_both_transforms")
        sizePolicy8.setHeightForWidth(self.show_both_transforms.sizePolicy().hasHeightForWidth())
        self.show_both_transforms.setSizePolicy(sizePolicy8)
        self.show_both_transforms.setFont(font12)

        self.horizontalLayout_30.addWidget(self.show_both_transforms)

        self.line_3 = QFrame(self.frame_51)
        self.line_3.setObjectName(u"line_3")
        self.line_3.setFrameShadow(QFrame.Raised)
        self.line_3.setFrameShape(QFrame.Shape.HLine)

        self.horizontalLayout_30.addWidget(self.line_3)


        self.verticalLayout_60.addWidget(self.frame_51)

        self.both_transforms_frame = QFrame(self.groupBox_2)
        self.both_transforms_frame.setObjectName(u"both_transforms_frame")
        self.both_transforms_frame.setFrameShape(QFrame.StyledPanel)
        self.both_transforms_frame.setFrameShadow(QFrame.Raised)
        self.formLayout = QFormLayout(self.both_transforms_frame)
        self.formLayout.setObjectName(u"formLayout")
        self.formLayout.setHorizontalSpacing(6)
        self.formLayout.setVerticalSpacing(6)
        self.formLayout.setContentsMargins(0, 0, 0, 0)
        self.label_71 = QLabel(self.both_transforms_frame)
        self.label_71.setObjectName(u"label_71")

        self.formLayout.setWidget(0, QFormLayout.ItemRole.LabelRole, self.label_71)

        self.frame_62 = QFrame(self.both_transforms_frame)
        self.frame_62.setObjectName(u"frame_62")
        self.frame_62.setFrameShape(QFrame.StyledPanel)
        self.frame_62.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_53 = QHBoxLayout(self.frame_62)
        self.horizontalLayout_53.setObjectName(u"horizontalLayout_53")
        self.horizontalLayout_53.setContentsMargins(0, 0, 0, 0)
        self.h_flip_chance_slider = QSlider(self.frame_62)
        self.h_flip_chance_slider.setObjectName(u"h_flip_chance_slider")
        self.h_flip_chance_slider.setMaximum(100)
        self.h_flip_chance_slider.setOrientation(Qt.Horizontal)
        self.h_flip_chance_slider.setTickPosition(QSlider.NoTicks)
        self.h_flip_chance_slider.setTickInterval(10)

        self.horizontalLayout_53.addWidget(self.h_flip_chance_slider)

        self.h_flip_chance = QDoubleSpinBox(self.frame_62)
        self.h_flip_chance.setObjectName(u"h_flip_chance")
        self.h_flip_chance.setDecimals(1)

        self.horizontalLayout_53.addWidget(self.h_flip_chance)


        self.formLayout.setWidget(0, QFormLayout.ItemRole.FieldRole, self.frame_62)

        self.label_72 = QLabel(self.both_transforms_frame)
        self.label_72.setObjectName(u"label_72")

        self.formLayout.setWidget(1, QFormLayout.ItemRole.LabelRole, self.label_72)

        self.label_73 = QLabel(self.both_transforms_frame)
        self.label_73.setObjectName(u"label_73")

        self.formLayout.setWidget(2, QFormLayout.ItemRole.LabelRole, self.label_73)

        self.frame_64 = QFrame(self.both_transforms_frame)
        self.frame_64.setObjectName(u"frame_64")
        self.frame_64.setFrameShape(QFrame.StyledPanel)
        self.frame_64.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_55 = QHBoxLayout(self.frame_64)
        self.horizontalLayout_55.setObjectName(u"horizontalLayout_55")
        self.horizontalLayout_55.setContentsMargins(0, 0, 0, 0)
        self.v_flip_chance_slider = QSlider(self.frame_64)
        self.v_flip_chance_slider.setObjectName(u"v_flip_chance_slider")
        self.v_flip_chance_slider.setMaximum(100)
        self.v_flip_chance_slider.setOrientation(Qt.Horizontal)
        self.v_flip_chance_slider.setTickPosition(QSlider.NoTicks)
        self.v_flip_chance_slider.setTickInterval(10)

        self.horizontalLayout_55.addWidget(self.v_flip_chance_slider)

        self.v_flip_chance = QDoubleSpinBox(self.frame_64)
        self.v_flip_chance.setObjectName(u"v_flip_chance")
        self.v_flip_chance.setDecimals(1)

        self.horizontalLayout_55.addWidget(self.v_flip_chance)


        self.formLayout.setWidget(1, QFormLayout.ItemRole.FieldRole, self.frame_64)

        self.frame_63 = QFrame(self.both_transforms_frame)
        self.frame_63.setObjectName(u"frame_63")
        self.frame_63.setFrameShape(QFrame.StyledPanel)
        self.frame_63.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_54 = QHBoxLayout(self.frame_63)
        self.horizontalLayout_54.setObjectName(u"horizontalLayout_54")
        self.horizontalLayout_54.setContentsMargins(0, 0, 0, 0)
        self.rot90_chance_slider = QSlider(self.frame_63)
        self.rot90_chance_slider.setObjectName(u"rot90_chance_slider")
        self.rot90_chance_slider.setMaximum(100)
        self.rot90_chance_slider.setOrientation(Qt.Horizontal)
        self.rot90_chance_slider.setTickPosition(QSlider.NoTicks)
        self.rot90_chance_slider.setTickInterval(10)

        self.horizontalLayout_54.addWidget(self.rot90_chance_slider)

        self.rot90_chance = QDoubleSpinBox(self.frame_63)
        self.rot90_chance.setObjectName(u"rot90_chance")
        self.rot90_chance.setDecimals(1)

        self.horizontalLayout_54.addWidget(self.rot90_chance)


        self.formLayout.setWidget(2, QFormLayout.ItemRole.FieldRole, self.frame_63)

        self.label_138 = QLabel(self.both_transforms_frame)
        self.label_138.setObjectName(u"label_138")

        self.formLayout.setWidget(3, QFormLayout.ItemRole.LabelRole, self.label_138)

        self.frame_123 = QFrame(self.both_transforms_frame)
        self.frame_123.setObjectName(u"frame_123")
        self.frame_123.setFrameShape(QFrame.StyledPanel)
        self.frame_123.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_117 = QHBoxLayout(self.frame_123)
        self.horizontalLayout_117.setObjectName(u"horizontalLayout_117")
        self.horizontalLayout_117.setContentsMargins(0, 0, 0, 0)
        self.elastic_transform_chance_slider = QSlider(self.frame_123)
        self.elastic_transform_chance_slider.setObjectName(u"elastic_transform_chance_slider")
        self.elastic_transform_chance_slider.setMaximum(100)
        self.elastic_transform_chance_slider.setOrientation(Qt.Horizontal)
        self.elastic_transform_chance_slider.setTickPosition(QSlider.NoTicks)
        self.elastic_transform_chance_slider.setTickInterval(10)

        self.horizontalLayout_117.addWidget(self.elastic_transform_chance_slider)

        self.elastic_transform_chance = QDoubleSpinBox(self.frame_123)
        self.elastic_transform_chance.setObjectName(u"elastic_transform_chance")
        self.elastic_transform_chance.setDecimals(1)

        self.horizontalLayout_117.addWidget(self.elastic_transform_chance)


        self.formLayout.setWidget(3, QFormLayout.ItemRole.FieldRole, self.frame_123)

        self.label_139 = QLabel(self.both_transforms_frame)
        self.label_139.setObjectName(u"label_139")

        self.formLayout.setWidget(4, QFormLayout.ItemRole.LabelRole, self.label_139)

        self.label_140 = QLabel(self.both_transforms_frame)
        self.label_140.setObjectName(u"label_140")

        self.formLayout.setWidget(5, QFormLayout.ItemRole.LabelRole, self.label_140)

        self.frame_125 = QFrame(self.both_transforms_frame)
        self.frame_125.setObjectName(u"frame_125")
        self.frame_125.setFrameShape(QFrame.StyledPanel)
        self.frame_125.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_119 = QHBoxLayout(self.frame_125)
        self.horizontalLayout_119.setObjectName(u"horizontalLayout_119")
        self.horizontalLayout_119.setContentsMargins(0, 0, 0, 0)
        self.elastic_transform_alpha_slider = QSlider(self.frame_125)
        self.elastic_transform_alpha_slider.setObjectName(u"elastic_transform_alpha_slider")
        self.elastic_transform_alpha_slider.setMaximum(100)
        self.elastic_transform_alpha_slider.setOrientation(Qt.Horizontal)
        self.elastic_transform_alpha_slider.setTickPosition(QSlider.NoTicks)
        self.elastic_transform_alpha_slider.setTickInterval(10)

        self.horizontalLayout_119.addWidget(self.elastic_transform_alpha_slider)

        self.elastic_transform_alpha = QDoubleSpinBox(self.frame_125)
        self.elastic_transform_alpha.setObjectName(u"elastic_transform_alpha")
        self.elastic_transform_alpha.setMinimumSize(QSize(74, 0))
        self.elastic_transform_alpha.setDecimals(1)
        self.elastic_transform_alpha.setMaximum(5.000000000000000)
        self.elastic_transform_alpha.setSingleStep(0.050000000000000)
        self.elastic_transform_alpha.setValue(1.000000000000000)

        self.horizontalLayout_119.addWidget(self.elastic_transform_alpha)


        self.formLayout.setWidget(4, QFormLayout.ItemRole.FieldRole, self.frame_125)

        self.frame_124 = QFrame(self.both_transforms_frame)
        self.frame_124.setObjectName(u"frame_124")
        self.frame_124.setFrameShape(QFrame.StyledPanel)
        self.frame_124.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_118 = QHBoxLayout(self.frame_124)
        self.horizontalLayout_118.setObjectName(u"horizontalLayout_118")
        self.horizontalLayout_118.setContentsMargins(0, 0, 0, 0)
        self.elastic_transform_sigma_slider = QSlider(self.frame_124)
        self.elastic_transform_sigma_slider.setObjectName(u"elastic_transform_sigma_slider")
        self.elastic_transform_sigma_slider.setMaximum(100)
        self.elastic_transform_sigma_slider.setOrientation(Qt.Horizontal)
        self.elastic_transform_sigma_slider.setTickPosition(QSlider.NoTicks)
        self.elastic_transform_sigma_slider.setTickInterval(10)

        self.horizontalLayout_118.addWidget(self.elastic_transform_sigma_slider)

        self.elastic_transform_sigma = QDoubleSpinBox(self.frame_124)
        self.elastic_transform_sigma.setObjectName(u"elastic_transform_sigma")
        sizePolicy9.setHeightForWidth(self.elastic_transform_sigma.sizePolicy().hasHeightForWidth())
        self.elastic_transform_sigma.setSizePolicy(sizePolicy9)
        self.elastic_transform_sigma.setMinimumSize(QSize(74, 0))
        self.elastic_transform_sigma.setDecimals(1)
        self.elastic_transform_sigma.setSingleStep(5.000000000000000)
        self.elastic_transform_sigma.setValue(50.000000000000000)

        self.horizontalLayout_118.addWidget(self.elastic_transform_sigma)


        self.formLayout.setWidget(5, QFormLayout.ItemRole.FieldRole, self.frame_124)

        self.label_141 = QLabel(self.both_transforms_frame)
        self.label_141.setObjectName(u"label_141")

        self.formLayout.setWidget(6, QFormLayout.ItemRole.LabelRole, self.label_141)

        self.frame_126 = QFrame(self.both_transforms_frame)
        self.frame_126.setObjectName(u"frame_126")
        self.frame_126.setFrameShape(QFrame.StyledPanel)
        self.frame_126.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_120 = QHBoxLayout(self.frame_126)
        self.horizontalLayout_120.setObjectName(u"horizontalLayout_120")
        self.horizontalLayout_120.setContentsMargins(0, 0, 0, 0)
        self.optical_distortion_chance_slider = QSlider(self.frame_126)
        self.optical_distortion_chance_slider.setObjectName(u"optical_distortion_chance_slider")
        self.optical_distortion_chance_slider.setMaximum(100)
        self.optical_distortion_chance_slider.setOrientation(Qt.Horizontal)
        self.optical_distortion_chance_slider.setTickPosition(QSlider.NoTicks)
        self.optical_distortion_chance_slider.setTickInterval(10)

        self.horizontalLayout_120.addWidget(self.optical_distortion_chance_slider)

        self.optical_distortion_chance = QDoubleSpinBox(self.frame_126)
        self.optical_distortion_chance.setObjectName(u"optical_distortion_chance")
        self.optical_distortion_chance.setDecimals(1)

        self.horizontalLayout_120.addWidget(self.optical_distortion_chance)


        self.formLayout.setWidget(6, QFormLayout.ItemRole.FieldRole, self.frame_126)

        self.label_142 = QLabel(self.both_transforms_frame)
        self.label_142.setObjectName(u"label_142")

        self.formLayout.setWidget(7, QFormLayout.ItemRole.LabelRole, self.label_142)

        self.frame_127 = QFrame(self.both_transforms_frame)
        self.frame_127.setObjectName(u"frame_127")
        self.frame_127.setFrameShape(QFrame.StyledPanel)
        self.frame_127.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_121 = QHBoxLayout(self.frame_127)
        self.horizontalLayout_121.setObjectName(u"horizontalLayout_121")
        self.horizontalLayout_121.setContentsMargins(0, 0, 0, 0)
        self.optical_distortion_min = QDoubleSpinBox(self.frame_127)
        self.optical_distortion_min.setObjectName(u"optical_distortion_min")
        self.optical_distortion_min.setMinimum(-2.500000000000000)
        self.optical_distortion_min.setMaximum(2.500000000000000)
        self.optical_distortion_min.setSingleStep(0.050000000000000)
        self.optical_distortion_min.setValue(-0.050000000000000)

        self.horizontalLayout_121.addWidget(self.optical_distortion_min)

        self.optical_distortion_max = QDoubleSpinBox(self.frame_127)
        self.optical_distortion_max.setObjectName(u"optical_distortion_max")
        self.optical_distortion_max.setMinimum(-2.500000000000000)
        self.optical_distortion_max.setMaximum(2.500000000000000)
        self.optical_distortion_max.setSingleStep(0.050000000000000)
        self.optical_distortion_max.setValue(0.050000000000000)

        self.horizontalLayout_121.addWidget(self.optical_distortion_max)


        self.formLayout.setWidget(7, QFormLayout.ItemRole.FieldRole, self.frame_127)

        self.label_143 = QLabel(self.both_transforms_frame)
        self.label_143.setObjectName(u"label_143")

        self.formLayout.setWidget(8, QFormLayout.ItemRole.LabelRole, self.label_143)

        self.optical_distortion_mode = QComboBox(self.both_transforms_frame)
        self.optical_distortion_mode.setObjectName(u"optical_distortion_mode")

        self.formLayout.setWidget(8, QFormLayout.ItemRole.FieldRole, self.optical_distortion_mode)

        self.label_144 = QLabel(self.both_transforms_frame)
        self.label_144.setObjectName(u"label_144")

        self.formLayout.setWidget(9, QFormLayout.ItemRole.LabelRole, self.label_144)

        self.label_145 = QLabel(self.both_transforms_frame)
        self.label_145.setObjectName(u"label_145")

        self.formLayout.setWidget(10, QFormLayout.ItemRole.LabelRole, self.label_145)

        self.label_146 = QLabel(self.both_transforms_frame)
        self.label_146.setObjectName(u"label_146")

        self.formLayout.setWidget(11, QFormLayout.ItemRole.LabelRole, self.label_146)

        self.label_147 = QLabel(self.both_transforms_frame)
        self.label_147.setObjectName(u"label_147")

        self.formLayout.setWidget(12, QFormLayout.ItemRole.LabelRole, self.label_147)

        self.frame_128 = QFrame(self.both_transforms_frame)
        self.frame_128.setObjectName(u"frame_128")
        self.frame_128.setFrameShape(QFrame.StyledPanel)
        self.frame_128.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_122 = QHBoxLayout(self.frame_128)
        self.horizontalLayout_122.setObjectName(u"horizontalLayout_122")
        self.horizontalLayout_122.setContentsMargins(0, 0, 0, 0)
        self.coarse_dropout_chance_slider = QSlider(self.frame_128)
        self.coarse_dropout_chance_slider.setObjectName(u"coarse_dropout_chance_slider")
        self.coarse_dropout_chance_slider.setMaximum(100)
        self.coarse_dropout_chance_slider.setOrientation(Qt.Horizontal)
        self.coarse_dropout_chance_slider.setTickPosition(QSlider.NoTicks)
        self.coarse_dropout_chance_slider.setTickInterval(10)

        self.horizontalLayout_122.addWidget(self.coarse_dropout_chance_slider)

        self.coarse_dropout_chance = QDoubleSpinBox(self.frame_128)
        self.coarse_dropout_chance.setObjectName(u"coarse_dropout_chance")
        self.coarse_dropout_chance.setDecimals(1)

        self.horizontalLayout_122.addWidget(self.coarse_dropout_chance)


        self.formLayout.setWidget(9, QFormLayout.ItemRole.FieldRole, self.frame_128)

        self.frame_129 = QFrame(self.both_transforms_frame)
        self.frame_129.setObjectName(u"frame_129")
        self.frame_129.setFrameShape(QFrame.StyledPanel)
        self.frame_129.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_123 = QHBoxLayout(self.frame_129)
        self.horizontalLayout_123.setObjectName(u"horizontalLayout_123")
        self.horizontalLayout_123.setContentsMargins(0, 0, 0, 0)
        self.coarse_dropout_holes_min = QSpinBox(self.frame_129)
        self.coarse_dropout_holes_min.setObjectName(u"coarse_dropout_holes_min")
        self.coarse_dropout_holes_min.setMinimum(1)

        self.horizontalLayout_123.addWidget(self.coarse_dropout_holes_min)

        self.coarse_dropout_holes_max = QSpinBox(self.frame_129)
        self.coarse_dropout_holes_max.setObjectName(u"coarse_dropout_holes_max")
        self.coarse_dropout_holes_max.setMinimum(1)
        self.coarse_dropout_holes_max.setValue(2)

        self.horizontalLayout_123.addWidget(self.coarse_dropout_holes_max)


        self.formLayout.setWidget(10, QFormLayout.ItemRole.FieldRole, self.frame_129)

        self.frame_130 = QFrame(self.both_transforms_frame)
        self.frame_130.setObjectName(u"frame_130")
        self.frame_130.setFrameShape(QFrame.StyledPanel)
        self.frame_130.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_124 = QHBoxLayout(self.frame_130)
        self.horizontalLayout_124.setObjectName(u"horizontalLayout_124")
        self.horizontalLayout_124.setContentsMargins(0, 0, 0, 0)
        self.coarse_dropout_height_min = QDoubleSpinBox(self.frame_130)
        self.coarse_dropout_height_min.setObjectName(u"coarse_dropout_height_min")
        self.coarse_dropout_height_min.setMaximum(1.000000000000000)
        self.coarse_dropout_height_min.setSingleStep(0.050000000000000)
        self.coarse_dropout_height_min.setValue(0.100000000000000)

        self.horizontalLayout_124.addWidget(self.coarse_dropout_height_min)

        self.coarse_dropout_height_max = QDoubleSpinBox(self.frame_130)
        self.coarse_dropout_height_max.setObjectName(u"coarse_dropout_height_max")
        self.coarse_dropout_height_max.setMaximum(1.000000000000000)
        self.coarse_dropout_height_max.setSingleStep(0.050000000000000)
        self.coarse_dropout_height_max.setValue(0.200000000000000)

        self.horizontalLayout_124.addWidget(self.coarse_dropout_height_max)


        self.formLayout.setWidget(11, QFormLayout.ItemRole.FieldRole, self.frame_130)

        self.frame_131 = QFrame(self.both_transforms_frame)
        self.frame_131.setObjectName(u"frame_131")
        self.frame_131.setFrameShape(QFrame.StyledPanel)
        self.frame_131.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_125 = QHBoxLayout(self.frame_131)
        self.horizontalLayout_125.setObjectName(u"horizontalLayout_125")
        self.horizontalLayout_125.setContentsMargins(0, 0, 0, 0)
        self.coarse_dropout_width_min = QDoubleSpinBox(self.frame_131)
        self.coarse_dropout_width_min.setObjectName(u"coarse_dropout_width_min")
        self.coarse_dropout_width_min.setMaximum(1.000000000000000)
        self.coarse_dropout_width_min.setSingleStep(0.050000000000000)
        self.coarse_dropout_width_min.setValue(0.100000000000000)

        self.horizontalLayout_125.addWidget(self.coarse_dropout_width_min)

        self.coarse_dropout_width_max = QDoubleSpinBox(self.frame_131)
        self.coarse_dropout_width_max.setObjectName(u"coarse_dropout_width_max")
        self.coarse_dropout_width_max.setMaximum(1.000000000000000)
        self.coarse_dropout_width_max.setSingleStep(0.050000000000000)
        self.coarse_dropout_width_max.setValue(0.200000000000000)

        self.horizontalLayout_125.addWidget(self.coarse_dropout_width_max)


        self.formLayout.setWidget(12, QFormLayout.ItemRole.FieldRole, self.frame_131)


        self.verticalLayout_60.addWidget(self.both_transforms_frame)


        self.verticalLayout_10.addWidget(self.groupBox_2)

        self.verticalSpacer_3 = QSpacerItem(20, 565, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.verticalLayout_10.addItem(self.verticalSpacer_3)

        self.scrollArea_2.setWidget(self.dataloader_settings_container)

        self.verticalLayout_5.addWidget(self.scrollArea_2)

        self.settings_pages.addWidget(self.dataset_page)
        self.training_page = QWidget()
        self.training_page.setObjectName(u"training_page")
        self.verticalLayout_22 = QVBoxLayout(self.training_page)
        self.verticalLayout_22.setSpacing(0)
        self.verticalLayout_22.setObjectName(u"verticalLayout_22")
        self.verticalLayout_22.setContentsMargins(0, 0, 0, 0)
        self.train_page_state_swap = QStackedWidget(self.training_page)
        self.train_page_state_swap.setObjectName(u"train_page_state_swap")
        self.train_page_state_unlocked = QWidget()
        self.train_page_state_unlocked.setObjectName(u"train_page_state_unlocked")
        self.verticalLayout_55 = QVBoxLayout(self.train_page_state_unlocked)
        self.verticalLayout_55.setSpacing(0)
        self.verticalLayout_55.setObjectName(u"verticalLayout_55")
        self.verticalLayout_55.setContentsMargins(0, 0, 0, 0)
        self.scrollArea_3 = QScrollArea(self.train_page_state_unlocked)
        self.scrollArea_3.setObjectName(u"scrollArea_3")
        self.scrollArea_3.setFrameShadow(QFrame.Raised)
        self.scrollArea_3.setWidgetResizable(True)
        self.training_settings_container_ = QWidget()
        self.training_settings_container_.setObjectName(u"training_settings_container_")
        self.training_settings_container_.setGeometry(QRect(0, 0, 437, 754))
        self.verticalLayout_20 = QVBoxLayout(self.training_settings_container_)
        self.verticalLayout_20.setSpacing(0)
        self.verticalLayout_20.setObjectName(u"verticalLayout_20")
        self.verticalLayout_20.setContentsMargins(0, 0, 0, 0)
        self.training_settings_container = QFrame(self.training_settings_container_)
        self.training_settings_container.setObjectName(u"training_settings_container")
        self.training_settings_container.setFrameShape(QFrame.StyledPanel)
        self.training_settings_container.setFrameShadow(QFrame.Raised)
        self.verticalLayout_54 = QVBoxLayout(self.training_settings_container)
        self.verticalLayout_54.setObjectName(u"verticalLayout_54")
        self.verticalLayout_54.setContentsMargins(0, 0, 0, 0)
        self.frame_85 = QFrame(self.training_settings_container)
        self.frame_85.setObjectName(u"frame_85")
        self.frame_85.setFrameShape(QFrame.StyledPanel)
        self.frame_85.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_74 = QHBoxLayout(self.frame_85)
        self.horizontalLayout_74.setSpacing(3)
        self.horizontalLayout_74.setObjectName(u"horizontalLayout_74")
        self.horizontalLayout_74.setContentsMargins(0, 0, 0, 0)
        self.continue_train = QCheckBox(self.frame_85)
        self.continue_train.setObjectName(u"continue_train")
        sizePolicy8.setHeightForWidth(self.continue_train.sizePolicy().hasHeightForWidth())
        self.continue_train.setSizePolicy(sizePolicy8)
        self.continue_train.setMinimumSize(QSize(0, 0))
        font13 = QFont()
        font13.setPointSize(11)
        font13.setBold(False)
        font13.setUnderline(False)
        font13.setKerning(True)
        self.continue_train.setFont(font13)
        self.continue_train.setLayoutDirection(Qt.RightToLeft)
        self.continue_train.setTristate(False)

        self.horizontalLayout_74.addWidget(self.continue_train)

        self.continue_train_settings = QFrame(self.frame_85)
        self.continue_train_settings.setObjectName(u"continue_train_settings")
        self.continue_train_settings.setFrameShape(QFrame.StyledPanel)
        self.continue_train_settings.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_58 = QHBoxLayout(self.continue_train_settings)
        self.horizontalLayout_58.setObjectName(u"horizontalLayout_58")
        self.horizontalLayout_58.setContentsMargins(0, 0, 0, 0)
        self.line_12 = QFrame(self.continue_train_settings)
        self.line_12.setObjectName(u"line_12")
        sizePolicy15 = QSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Minimum)
        sizePolicy15.setHorizontalStretch(0)
        sizePolicy15.setVerticalStretch(0)
        sizePolicy15.setHeightForWidth(self.line_12.sizePolicy().hasHeightForWidth())
        self.line_12.setSizePolicy(sizePolicy15)
        self.line_12.setMinimumSize(QSize(0, 0))
        self.line_12.setFrameShape(QFrame.Shape.VLine)
        self.line_12.setFrameShadow(QFrame.Shadow.Sunken)

        self.horizontalLayout_58.addWidget(self.line_12)

        self.label_16 = QLabel(self.continue_train_settings)
        self.label_16.setObjectName(u"label_16")
        sizePolicy1.setHeightForWidth(self.label_16.sizePolicy().hasHeightForWidth())
        self.label_16.setSizePolicy(sizePolicy1)
        font14 = QFont()
        font14.setPointSize(11)
        self.label_16.setFont(font14)

        self.horizontalLayout_58.addWidget(self.label_16)

        self.load_epoch = QSpinBox(self.continue_train_settings)
        self.load_epoch.setObjectName(u"load_epoch")
        self.load_epoch.setMinimum(1)
        self.load_epoch.setMaximum(999999)

        self.horizontalLayout_58.addWidget(self.load_epoch)


        self.horizontalLayout_74.addWidget(self.continue_train_settings)

        self.line_28 = QFrame(self.frame_85)
        self.line_28.setObjectName(u"line_28")
        sizePolicy5.setHeightForWidth(self.line_28.sizePolicy().hasHeightForWidth())
        self.line_28.setSizePolicy(sizePolicy5)
        self.line_28.setFrameShadow(QFrame.Raised)
        self.line_28.setFrameShape(QFrame.Shape.HLine)

        self.horizontalLayout_74.addWidget(self.line_28)

        self.verticalSpacer_14 = QSpacerItem(0, 27, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_74.addItem(self.verticalSpacer_14)


        self.verticalLayout_54.addWidget(self.frame_85)

        self.tabWidget_2 = QTabWidget(self.training_settings_container)
        self.tabWidget_2.setObjectName(u"tabWidget_2")
        sizePolicy16 = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        sizePolicy16.setHorizontalStretch(0)
        sizePolicy16.setVerticalStretch(0)
        sizePolicy16.setHeightForWidth(self.tabWidget_2.sizePolicy().hasHeightForWidth())
        self.tabWidget_2.setSizePolicy(sizePolicy16)
        self.tabWidget_2.setTabPosition(QTabWidget.North)
        self._generator_tab = QWidget()
        self._generator_tab.setObjectName(u"_generator_tab")
        sizePolicy.setHeightForWidth(self._generator_tab.sizePolicy().hasHeightForWidth())
        self._generator_tab.setSizePolicy(sizePolicy)
        self.verticalLayout_12 = QVBoxLayout(self._generator_tab)
        self.verticalLayout_12.setObjectName(u"verticalLayout_12")
        self.verticalLayout_12.setContentsMargins(0, 0, 0, 0)
        self.scrollArea_4 = QScrollArea(self._generator_tab)
        self.scrollArea_4.setObjectName(u"scrollArea_4")
        self.scrollArea_4.setFrameShadow(QFrame.Raised)
        self.scrollArea_4.setWidgetResizable(True)
        self.generator_tab = QWidget()
        self.generator_tab.setObjectName(u"generator_tab")
        self.generator_tab.setGeometry(QRect(0, 0, 427, 652))
        self.verticalLayout_15 = QVBoxLayout(self.generator_tab)
        self.verticalLayout_15.setObjectName(u"verticalLayout_15")
        self.verticalLayout_15.setContentsMargins(6, 6, 6, 6)
        self.frame_31 = QFrame(self.generator_tab)
        self.frame_31.setObjectName(u"frame_31")
        self.frame_31.setFrameShape(QFrame.StyledPanel)
        self.frame_31.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_38 = QHBoxLayout(self.frame_31)
        self.horizontalLayout_38.setObjectName(u"horizontalLayout_38")
        self.horizontalLayout_38.setContentsMargins(0, 0, 0, 0)
        self.label_57 = QLabel(self.frame_31)
        self.label_57.setObjectName(u"label_57")
        sizePolicy1.setHeightForWidth(self.label_57.sizePolicy().hasHeightForWidth())
        self.label_57.setSizePolicy(sizePolicy1)
        self.label_57.setFont(font14)
        self.label_57.setFrameShape(QFrame.NoFrame)
        self.label_57.setFrameShadow(QFrame.Raised)

        self.horizontalLayout_38.addWidget(self.label_57)

        self.line_11 = QFrame(self.frame_31)
        self.line_11.setObjectName(u"line_11")
        self.line_11.setFrameShadow(QFrame.Raised)
        self.line_11.setFrameShape(QFrame.Shape.HLine)

        self.horizontalLayout_38.addWidget(self.line_11)


        self.verticalLayout_15.addWidget(self.frame_31)

        self.frame_21 = QFrame(self.generator_tab)
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
        self.gen_epochs.setMinimum(0)
        self.gen_epochs.setMaximum(999999)
        self.gen_epochs.setValue(100)

        self.formLayout_8.setWidget(0, QFormLayout.ItemRole.FieldRole, self.gen_epochs)

        self.label_31 = QLabel(self.frame_21)
        self.label_31.setObjectName(u"label_31")

        self.formLayout_8.setWidget(1, QFormLayout.ItemRole.LabelRole, self.label_31)

        self.gen_epochs_decay = QSpinBox(self.frame_21)
        self.gen_epochs_decay.setObjectName(u"gen_epochs_decay")
        self.gen_epochs_decay.setMinimum(0)
        self.gen_epochs_decay.setMaximum(999999)
        self.gen_epochs_decay.setValue(100)

        self.formLayout_8.setWidget(1, QFormLayout.ItemRole.FieldRole, self.gen_epochs_decay)


        self.verticalLayout_15.addWidget(self.frame_21)

        self.frame_40 = QFrame(self.generator_tab)
        self.frame_40.setObjectName(u"frame_40")
        self.frame_40.setFrameShape(QFrame.StyledPanel)
        self.frame_40.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_41 = QHBoxLayout(self.frame_40)
        self.horizontalLayout_41.setObjectName(u"horizontalLayout_41")
        self.horizontalLayout_41.setContentsMargins(0, 0, 0, 0)
        self.label_60 = QLabel(self.frame_40)
        self.label_60.setObjectName(u"label_60")
        sizePolicy1.setHeightForWidth(self.label_60.sizePolicy().hasHeightForWidth())
        self.label_60.setSizePolicy(sizePolicy1)
        self.label_60.setFont(font14)
        self.label_60.setFrameShape(QFrame.NoFrame)
        self.label_60.setFrameShadow(QFrame.Raised)

        self.horizontalLayout_41.addWidget(self.label_60)

        self.line_14 = QFrame(self.frame_40)
        self.line_14.setObjectName(u"line_14")
        self.line_14.setFrameShadow(QFrame.Raised)
        self.line_14.setFrameShape(QFrame.Shape.HLine)

        self.horizontalLayout_41.addWidget(self.line_14)


        self.verticalLayout_15.addWidget(self.frame_40)

        self.frame_41 = QFrame(self.generator_tab)
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

        self.frame_39 = QFrame(self.generator_tab)
        self.frame_39.setObjectName(u"frame_39")
        self.frame_39.setFrameShape(QFrame.StyledPanel)
        self.frame_39.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_40 = QHBoxLayout(self.frame_39)
        self.horizontalLayout_40.setObjectName(u"horizontalLayout_40")
        self.horizontalLayout_40.setContentsMargins(0, 0, 0, 0)
        self.label_59 = QLabel(self.frame_39)
        self.label_59.setObjectName(u"label_59")
        sizePolicy1.setHeightForWidth(self.label_59.sizePolicy().hasHeightForWidth())
        self.label_59.setSizePolicy(sizePolicy1)
        self.label_59.setFont(font14)
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
        self.label_14 = QLabel(self.generator_tab)
        self.label_14.setObjectName(u"label_14")

        self.formLayout_12.setWidget(0, QFormLayout.ItemRole.LabelRole, self.label_14)

        self.horizontalLayout_14 = QHBoxLayout()
        self.horizontalLayout_14.setObjectName(u"horizontalLayout_14")
        self.gen_optim_beta1_slider = QSlider(self.generator_tab)
        self.gen_optim_beta1_slider.setObjectName(u"gen_optim_beta1_slider")
        self.gen_optim_beta1_slider.setMaximum(250)
        self.gen_optim_beta1_slider.setOrientation(Qt.Horizontal)

        self.horizontalLayout_14.addWidget(self.gen_optim_beta1_slider)

        self.gen_optim_beta1 = QDoubleSpinBox(self.generator_tab)
        self.gen_optim_beta1.setObjectName(u"gen_optim_beta1")
        self.gen_optim_beta1.setMaximum(2.500000000000000)
        self.gen_optim_beta1.setSingleStep(0.100000000000000)
        self.gen_optim_beta1.setValue(0.500000000000000)

        self.horizontalLayout_14.addWidget(self.gen_optim_beta1)


        self.formLayout_12.setLayout(0, QFormLayout.ItemRole.FieldRole, self.horizontalLayout_14)


        self.verticalLayout_15.addLayout(self.formLayout_12)

        self.verticalSpacer_9 = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.verticalLayout_15.addItem(self.verticalSpacer_9)

        self.scrollArea_4.setWidget(self.generator_tab)

        self.verticalLayout_12.addWidget(self.scrollArea_4)

        self.tabWidget_2.addTab(self._generator_tab, "")
        self._discriminator_tab = QWidget()
        self._discriminator_tab.setObjectName(u"_discriminator_tab")
        self.verticalLayout_35 = QVBoxLayout(self._discriminator_tab)
        self.verticalLayout_35.setObjectName(u"verticalLayout_35")
        self.verticalLayout_35.setContentsMargins(0, 0, 0, 0)
        self.scrollArea_6 = QScrollArea(self._discriminator_tab)
        self.scrollArea_6.setObjectName(u"scrollArea_6")
        self.scrollArea_6.setFrameShadow(QFrame.Raised)
        self.scrollArea_6.setWidgetResizable(True)
        self.discriminator_tab = QWidget()
        self.discriminator_tab.setObjectName(u"discriminator_tab")
        self.discriminator_tab.setGeometry(QRect(0, 0, 233, 307))
        self.verticalLayout_36 = QVBoxLayout(self.discriminator_tab)
        self.verticalLayout_36.setObjectName(u"verticalLayout_36")
        self.verticalLayout_36.setContentsMargins(6, 6, 6, 6)
        self.frame_76 = QFrame(self.discriminator_tab)
        self.frame_76.setObjectName(u"frame_76")
        self.frame_76.setFrameShape(QFrame.StyledPanel)
        self.frame_76.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_79 = QHBoxLayout(self.frame_76)
        self.horizontalLayout_79.setObjectName(u"horizontalLayout_79")
        self.horizontalLayout_79.setContentsMargins(0, 0, 0, 0)
        self.separate_lr_schedules = QCheckBox(self.frame_76)
        self.separate_lr_schedules.setObjectName(u"separate_lr_schedules")
        self.separate_lr_schedules.setFont(font14)

        self.horizontalLayout_79.addWidget(self.separate_lr_schedules)

        self.line_21 = QFrame(self.frame_76)
        self.line_21.setObjectName(u"line_21")
        self.line_21.setFrameShadow(QFrame.Raised)
        self.line_21.setFrameShape(QFrame.Shape.HLine)

        self.horizontalLayout_79.addWidget(self.line_21)


        self.verticalLayout_36.addWidget(self.frame_76)

        self.disc_schedule_box = QFrame(self.discriminator_tab)
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
        sizePolicy1.setHeightForWidth(self.label_61.sizePolicy().hasHeightForWidth())
        self.label_61.setSizePolicy(sizePolicy1)
        self.label_61.setFont(font14)
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
        self.disc_epochs.setMinimum(0)
        self.disc_epochs.setMaximum(999999)
        self.disc_epochs.setValue(100)

        self.formLayout_11.setWidget(0, QFormLayout.ItemRole.FieldRole, self.disc_epochs)

        self.label_35 = QLabel(self.frame_23)
        self.label_35.setObjectName(u"label_35")

        self.formLayout_11.setWidget(1, QFormLayout.ItemRole.LabelRole, self.label_35)

        self.disc_epochs_decay = QSpinBox(self.frame_23)
        self.disc_epochs_decay.setObjectName(u"disc_epochs_decay")
        self.disc_epochs_decay.setMinimum(0)
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
        sizePolicy1.setHeightForWidth(self.label_62.sizePolicy().hasHeightForWidth())
        self.label_62.setSizePolicy(sizePolicy1)
        self.label_62.setFont(font14)
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

        self.frame_44 = QFrame(self.discriminator_tab)
        self.frame_44.setObjectName(u"frame_44")
        self.frame_44.setFrameShape(QFrame.StyledPanel)
        self.frame_44.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_44 = QHBoxLayout(self.frame_44)
        self.horizontalLayout_44.setObjectName(u"horizontalLayout_44")
        self.horizontalLayout_44.setContentsMargins(0, 0, 0, 0)
        self.label_63 = QLabel(self.frame_44)
        self.label_63.setObjectName(u"label_63")
        sizePolicy1.setHeightForWidth(self.label_63.sizePolicy().hasHeightForWidth())
        self.label_63.setSizePolicy(sizePolicy1)
        self.label_63.setFont(font14)
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
        self.label_15 = QLabel(self.discriminator_tab)
        self.label_15.setObjectName(u"label_15")

        self.formLayout_14.setWidget(0, QFormLayout.ItemRole.LabelRole, self.label_15)

        self.horizontalLayout_26 = QHBoxLayout()
        self.horizontalLayout_26.setObjectName(u"horizontalLayout_26")
        self.disc_optim_beta1_slider = QSlider(self.discriminator_tab)
        self.disc_optim_beta1_slider.setObjectName(u"disc_optim_beta1_slider")
        self.disc_optim_beta1_slider.setMaximum(250)
        self.disc_optim_beta1_slider.setOrientation(Qt.Horizontal)

        self.horizontalLayout_26.addWidget(self.disc_optim_beta1_slider)

        self.disc_optim_beta1 = QDoubleSpinBox(self.discriminator_tab)
        self.disc_optim_beta1.setObjectName(u"disc_optim_beta1")
        self.disc_optim_beta1.setMaximum(2.500000000000000)
        self.disc_optim_beta1.setSingleStep(0.100000000000000)
        self.disc_optim_beta1.setValue(0.500000000000000)

        self.horizontalLayout_26.addWidget(self.disc_optim_beta1)


        self.formLayout_14.setLayout(0, QFormLayout.ItemRole.FieldRole, self.horizontalLayout_26)


        self.verticalLayout_36.addLayout(self.formLayout_14)

        self.verticalSpacer_4 = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.verticalLayout_36.addItem(self.verticalSpacer_4)

        self.scrollArea_6.setWidget(self.discriminator_tab)

        self.verticalLayout_35.addWidget(self.scrollArea_6)

        self.tabWidget_2.addTab(self._discriminator_tab, "")
        self._loss_tab = QWidget()
        self._loss_tab.setObjectName(u"_loss_tab")
        self.verticalLayout_14 = QVBoxLayout(self._loss_tab)
        self.verticalLayout_14.setObjectName(u"verticalLayout_14")
        self.verticalLayout_14.setContentsMargins(0, 0, 0, 0)
        self.scrollArea_5 = QScrollArea(self._loss_tab)
        self.scrollArea_5.setObjectName(u"scrollArea_5")
        self.scrollArea_5.setFrameShadow(QFrame.Raised)
        self.scrollArea_5.setWidgetResizable(True)
        self.loss_tab = QWidget()
        self.loss_tab.setObjectName(u"loss_tab")
        self.loss_tab.setGeometry(QRect(0, 0, 281, 299))
        self.verticalLayout_16 = QVBoxLayout(self.loss_tab)
        self.verticalLayout_16.setObjectName(u"verticalLayout_16")
        self.verticalLayout_16.setContentsMargins(6, 6, 6, 6)
        self.frame_25 = QFrame(self.loss_tab)
        self.frame_25.setObjectName(u"frame_25")
        self.frame_25.setFrameShape(QFrame.StyledPanel)
        self.frame_25.setFrameShadow(QFrame.Raised)
        self.formLayout_2 = QFormLayout(self.frame_25)
        self.formLayout_2.setObjectName(u"formLayout_2")
        self.formLayout_2.setContentsMargins(0, 0, 0, 0)
        self.label_13 = QLabel(self.frame_25)
        self.label_13.setObjectName(u"label_13")

        self.formLayout_2.setWidget(1, QFormLayout.ItemRole.LabelRole, self.label_13)

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


        self.formLayout_2.setLayout(1, QFormLayout.ItemRole.FieldRole, self.horizontalLayout_10)

        self.label_17 = QLabel(self.frame_25)
        self.label_17.setObjectName(u"label_17")

        self.formLayout_2.setWidget(3, QFormLayout.ItemRole.LabelRole, self.label_17)

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


        self.formLayout_2.setLayout(3, QFormLayout.ItemRole.FieldRole, self.horizontalLayout_11)

        self.label_37 = QLabel(self.frame_25)
        self.label_37.setObjectName(u"label_37")

        self.formLayout_2.setWidget(4, QFormLayout.ItemRole.LabelRole, self.label_37)

        self.label_38 = QLabel(self.frame_25)
        self.label_38.setObjectName(u"label_38")

        self.formLayout_2.setWidget(5, QFormLayout.ItemRole.LabelRole, self.label_38)

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


        self.formLayout_2.setLayout(4, QFormLayout.ItemRole.FieldRole, self.horizontalLayout_28)

        self.horizontalLayout_27 = QHBoxLayout()
        self.horizontalLayout_27.setObjectName(u"horizontalLayout_27")
        self.lambda_vgg_slider = QSlider(self.frame_25)
        self.lambda_vgg_slider.setObjectName(u"lambda_vgg_slider")
        self.lambda_vgg_slider.setMaximum(100)
        self.lambda_vgg_slider.setValue(0)
        self.lambda_vgg_slider.setOrientation(Qt.Horizontal)

        self.horizontalLayout_27.addWidget(self.lambda_vgg_slider)

        self.lambda_vgg = QDoubleSpinBox(self.frame_25)
        self.lambda_vgg.setObjectName(u"lambda_vgg")
        self.lambda_vgg.setMaximum(500.000000000000000)
        self.lambda_vgg.setValue(0.000000000000000)

        self.horizontalLayout_27.addWidget(self.lambda_vgg)


        self.formLayout_2.setLayout(5, QFormLayout.ItemRole.FieldRole, self.horizontalLayout_27)

        self.label_28 = QLabel(self.frame_25)
        self.label_28.setObjectName(u"label_28")

        self.formLayout_2.setWidget(0, QFormLayout.ItemRole.LabelRole, self.label_28)

        self.horizontalLayout_52 = QHBoxLayout()
        self.horizontalLayout_52.setObjectName(u"horizontalLayout_52")
        self.lambda_gan_slider = QSlider(self.frame_25)
        self.lambda_gan_slider.setObjectName(u"lambda_gan_slider")
        self.lambda_gan_slider.setMaximum(100)
        self.lambda_gan_slider.setValue(1)
        self.lambda_gan_slider.setOrientation(Qt.Horizontal)

        self.horizontalLayout_52.addWidget(self.lambda_gan_slider)

        self.lambda_gan = QDoubleSpinBox(self.frame_25)
        self.lambda_gan.setObjectName(u"lambda_gan")
        self.lambda_gan.setMaximum(100.000000000000000)
        self.lambda_gan.setValue(1.000000000000000)

        self.horizontalLayout_52.addWidget(self.lambda_gan)


        self.formLayout_2.setLayout(0, QFormLayout.ItemRole.FieldRole, self.horizontalLayout_52)

        self.label_29 = QLabel(self.frame_25)
        self.label_29.setObjectName(u"label_29")

        self.formLayout_2.setWidget(2, QFormLayout.ItemRole.LabelRole, self.label_29)

        self.horizontalLayout_78 = QHBoxLayout()
        self.horizontalLayout_78.setObjectName(u"horizontalLayout_78")
        self.lambda_l2_slider = QSlider(self.frame_25)
        self.lambda_l2_slider.setObjectName(u"lambda_l2_slider")
        self.lambda_l2_slider.setOrientation(Qt.Horizontal)

        self.horizontalLayout_78.addWidget(self.lambda_l2_slider)

        self.lambda_l2 = QDoubleSpinBox(self.frame_25)
        self.lambda_l2.setObjectName(u"lambda_l2")
        self.lambda_l2.setMaximum(500.000000000000000)
        self.lambda_l2.setSingleStep(5.000000000000000)

        self.horizontalLayout_78.addWidget(self.lambda_l2)


        self.formLayout_2.setLayout(2, QFormLayout.ItemRole.FieldRole, self.horizontalLayout_78)


        self.verticalLayout_16.addWidget(self.frame_25)

        self.line_44 = QFrame(self.loss_tab)
        self.line_44.setObjectName(u"line_44")
        self.line_44.setFrameShape(QFrame.Shape.HLine)
        self.line_44.setFrameShadow(QFrame.Shadow.Sunken)

        self.verticalLayout_16.addWidget(self.line_44)

        self.log_losses = QCheckBox(self.loss_tab)
        self.log_losses.setObjectName(u"log_losses")
        self.log_losses.setChecked(True)

        self.verticalLayout_16.addWidget(self.log_losses)

        self.log_loss_settings = QFrame(self.loss_tab)
        self.log_loss_settings.setObjectName(u"log_loss_settings")
        self.log_loss_settings.setFrameShape(QFrame.StyledPanel)
        self.log_loss_settings.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_108 = QHBoxLayout(self.log_loss_settings)
        self.horizontalLayout_108.setObjectName(u"horizontalLayout_108")
        self.horizontalLayout_108.setContentsMargins(0, 0, 0, 0)
        self.label_18 = QLabel(self.log_loss_settings)
        self.label_18.setObjectName(u"label_18")
        sizePolicy1.setHeightForWidth(self.label_18.sizePolicy().hasHeightForWidth())
        self.label_18.setSizePolicy(sizePolicy1)

        self.horizontalLayout_108.addWidget(self.label_18)

        self.loss_dump_frequency = QSpinBox(self.log_loss_settings)
        self.loss_dump_frequency.setObjectName(u"loss_dump_frequency")
        self.loss_dump_frequency.setMinimum(1)
        self.loss_dump_frequency.setMaximum(999)
        self.loss_dump_frequency.setSingleStep(1)
        self.loss_dump_frequency.setValue(5)

        self.horizontalLayout_108.addWidget(self.loss_dump_frequency)


        self.verticalLayout_16.addWidget(self.log_loss_settings)

        self.verticalSpacer_5 = QSpacerItem(20, 356, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.verticalLayout_16.addItem(self.verticalSpacer_5)

        self.scrollArea_5.setWidget(self.loss_tab)

        self.verticalLayout_14.addWidget(self.scrollArea_5)

        self.tabWidget_2.addTab(self._loss_tab, "")
        self._saving_tab = QWidget()
        self._saving_tab.setObjectName(u"_saving_tab")
        self.verticalLayout_17 = QVBoxLayout(self._saving_tab)
        self.verticalLayout_17.setObjectName(u"verticalLayout_17")
        self.verticalLayout_17.setContentsMargins(0, 0, 0, 0)
        self.scrollArea_7 = QScrollArea(self._saving_tab)
        self.scrollArea_7.setObjectName(u"scrollArea_7")
        self.scrollArea_7.setFrameShadow(QFrame.Raised)
        self.scrollArea_7.setWidgetResizable(True)
        self.saving_tab = QWidget()
        self.saving_tab.setObjectName(u"saving_tab")
        self.saving_tab.setGeometry(QRect(0, 0, 187, 177))
        self.verticalLayout_18 = QVBoxLayout(self.saving_tab)
        self.verticalLayout_18.setObjectName(u"verticalLayout_18")
        self.verticalLayout_18.setContentsMargins(2, 2, 2, 2)
        self.frame_70 = QFrame(self.saving_tab)
        self.frame_70.setObjectName(u"frame_70")
        self.frame_70.setFrameShape(QFrame.StyledPanel)
        self.frame_70.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_63 = QHBoxLayout(self.frame_70)
        self.horizontalLayout_63.setObjectName(u"horizontalLayout_63")
        self.horizontalLayout_63.setContentsMargins(0, 0, 0, 0)
        self.save_model = QCheckBox(self.frame_70)
        self.save_model.setObjectName(u"save_model")
        sizePolicy8.setHeightForWidth(self.save_model.sizePolicy().hasHeightForWidth())
        self.save_model.setSizePolicy(sizePolicy8)
        self.save_model.setFont(font14)
        self.save_model.setChecked(True)

        self.horizontalLayout_63.addWidget(self.save_model)

        self.line_19 = QFrame(self.frame_70)
        self.line_19.setObjectName(u"line_19")
        self.line_19.setFrameShadow(QFrame.Raised)
        self.line_19.setFrameShape(QFrame.Shape.HLine)

        self.horizontalLayout_63.addWidget(self.line_19)


        self.verticalLayout_18.addWidget(self.frame_70)

        self.save_model_settings = QFrame(self.saving_tab)
        self.save_model_settings.setObjectName(u"save_model_settings")
        self.save_model_settings.setFrameShape(QFrame.StyledPanel)
        self.save_model_settings.setFrameShadow(QFrame.Raised)
        self.formLayout_16 = QFormLayout(self.save_model_settings)
        self.formLayout_16.setObjectName(u"formLayout_16")
        self.formLayout_16.setContentsMargins(0, 0, 0, 0)
        self.label_24 = QLabel(self.save_model_settings)
        self.label_24.setObjectName(u"label_24")
        sizePolicy1.setHeightForWidth(self.label_24.sizePolicy().hasHeightForWidth())
        self.label_24.setSizePolicy(sizePolicy1)

        self.formLayout_16.setWidget(0, QFormLayout.ItemRole.LabelRole, self.label_24)

        self.horizontalLayout_17 = QHBoxLayout()
        self.horizontalLayout_17.setObjectName(u"horizontalLayout_17")
        self.model_save_rate = QSpinBox(self.save_model_settings)
        self.model_save_rate.setObjectName(u"model_save_rate")
        sizePolicy8.setHeightForWidth(self.model_save_rate.sizePolicy().hasHeightForWidth())
        self.model_save_rate.setSizePolicy(sizePolicy8)
        self.model_save_rate.setMinimum(1)
        self.model_save_rate.setValue(5)

        self.horizontalLayout_17.addWidget(self.model_save_rate)

        self.label_42 = QLabel(self.save_model_settings)
        self.label_42.setObjectName(u"label_42")

        self.horizontalLayout_17.addWidget(self.label_42)

        self.horizontalSpacer_2 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_17.addItem(self.horizontalSpacer_2)


        self.formLayout_16.setLayout(0, QFormLayout.ItemRole.FieldRole, self.horizontalLayout_17)


        self.verticalLayout_18.addWidget(self.save_model_settings)

        self.frame_69 = QFrame(self.saving_tab)
        self.frame_69.setObjectName(u"frame_69")
        self.frame_69.setFrameShape(QFrame.StyledPanel)
        self.frame_69.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_59 = QHBoxLayout(self.frame_69)
        self.horizontalLayout_59.setObjectName(u"horizontalLayout_59")
        self.horizontalLayout_59.setContentsMargins(0, 0, 0, 0)
        self.save_examples = QCheckBox(self.frame_69)
        self.save_examples.setObjectName(u"save_examples")
        sizePolicy8.setHeightForWidth(self.save_examples.sizePolicy().hasHeightForWidth())
        self.save_examples.setSizePolicy(sizePolicy8)
        self.save_examples.setFont(font14)
        self.save_examples.setChecked(True)

        self.horizontalLayout_59.addWidget(self.save_examples)

        self.line_18 = QFrame(self.frame_69)
        self.line_18.setObjectName(u"line_18")
        self.line_18.setFrameShadow(QFrame.Raised)
        self.line_18.setFrameShape(QFrame.Shape.HLine)

        self.horizontalLayout_59.addWidget(self.line_18)


        self.verticalLayout_18.addWidget(self.frame_69)

        self.save_examples_settings = QFrame(self.saving_tab)
        self.save_examples_settings.setObjectName(u"save_examples_settings")
        self.save_examples_settings.setFrameShape(QFrame.StyledPanel)
        self.save_examples_settings.setFrameShadow(QFrame.Raised)
        self.formLayout_27 = QFormLayout(self.save_examples_settings)
        self.formLayout_27.setObjectName(u"formLayout_27")
        self.formLayout_27.setContentsMargins(0, 0, 0, 0)
        self.label_26 = QLabel(self.save_examples_settings)
        self.label_26.setObjectName(u"label_26")

        self.formLayout_27.setWidget(0, QFormLayout.ItemRole.LabelRole, self.label_26)

        self.horizontalLayout_18 = QHBoxLayout()
        self.horizontalLayout_18.setObjectName(u"horizontalLayout_18")
        self.example_save_rate = QSpinBox(self.save_examples_settings)
        self.example_save_rate.setObjectName(u"example_save_rate")

        self.horizontalLayout_18.addWidget(self.example_save_rate)

        self.label_43 = QLabel(self.save_examples_settings)
        self.label_43.setObjectName(u"label_43")

        self.horizontalLayout_18.addWidget(self.label_43)

        self.horizontalSpacer_3 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_18.addItem(self.horizontalSpacer_3)


        self.formLayout_27.setLayout(0, QFormLayout.ItemRole.FieldRole, self.horizontalLayout_18)

        self.label_27 = QLabel(self.save_examples_settings)
        self.label_27.setObjectName(u"label_27")

        self.formLayout_27.setWidget(1, QFormLayout.ItemRole.LabelRole, self.label_27)

        self.horizontalLayout_19 = QHBoxLayout()
        self.horizontalLayout_19.setObjectName(u"horizontalLayout_19")
        self.num_examples = QSpinBox(self.save_examples_settings)
        self.num_examples.setObjectName(u"num_examples")

        self.horizontalLayout_19.addWidget(self.num_examples)

        self.horizontalSpacer_4 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_19.addItem(self.horizontalSpacer_4)


        self.formLayout_27.setLayout(1, QFormLayout.ItemRole.FieldRole, self.horizontalLayout_19)


        self.verticalLayout_18.addWidget(self.save_examples_settings)

        self.verticalSpacer_10 = QSpacerItem(20, 270, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.verticalLayout_18.addItem(self.verticalSpacer_10)

        self.scrollArea_7.setWidget(self.saving_tab)

        self.verticalLayout_17.addWidget(self.scrollArea_7)

        self.tabWidget_2.addTab(self._saving_tab, "")

        self.verticalLayout_54.addWidget(self.tabWidget_2)

        self.training_warning_label = QLabel(self.training_settings_container)
        self.training_warning_label.setObjectName(u"training_warning_label")
        self.training_warning_label.setAlignment(Qt.AlignCenter)

        self.verticalLayout_54.addWidget(self.training_warning_label)


        self.verticalLayout_20.addWidget(self.training_settings_container)

        self.scrollArea_3.setWidget(self.training_settings_container_)

        self.verticalLayout_55.addWidget(self.scrollArea_3)

        self.train_page_state_swap.addWidget(self.train_page_state_unlocked)
        self.train_page_state_locked = QWidget()
        self.train_page_state_locked.setObjectName(u"train_page_state_locked")
        self.verticalLayout_56 = QVBoxLayout(self.train_page_state_locked)
        self.verticalLayout_56.setObjectName(u"verticalLayout_56")
        self.verticalLayout_56.setContentsMargins(0, 0, 0, 0)
        self.train_page_locked_label = QLabel(self.train_page_state_locked)
        self.train_page_locked_label.setObjectName(u"train_page_locked_label")
        sizePolicy3.setHeightForWidth(self.train_page_locked_label.sizePolicy().hasHeightForWidth())
        self.train_page_locked_label.setSizePolicy(sizePolicy3)
        font15 = QFont()
        font15.setPointSize(12)
        font15.setBold(True)
        self.train_page_locked_label.setFont(font15)
        self.train_page_locked_label.setAlignment(Qt.AlignCenter)

        self.verticalLayout_56.addWidget(self.train_page_locked_label)

        self.train_page_state_swap.addWidget(self.train_page_state_locked)

        self.verticalLayout_22.addWidget(self.train_page_state_swap)

        self.train_functions_frame = QFrame(self.training_page)
        self.train_functions_frame.setObjectName(u"train_functions_frame")
        sizePolicy2.setHeightForWidth(self.train_functions_frame.sizePolicy().hasHeightForWidth())
        self.train_functions_frame.setSizePolicy(sizePolicy2)
        self.train_functions_frame.setFrameShape(QFrame.Panel)
        self.train_functions_frame.setFrameShadow(QFrame.Sunken)
        self.train_functions_frame.setLineWidth(1)
        self.verticalLayout_33 = QVBoxLayout(self.train_functions_frame)
        self.verticalLayout_33.setObjectName(u"verticalLayout_33")
        self.verticalLayout_33.setContentsMargins(3, 3, 3, 3)
        self.train_start = QPushButton(self.train_functions_frame)
        self.train_start.setObjectName(u"train_start")
        self.train_start.setFont(font15)

        self.verticalLayout_33.addWidget(self.train_start)

        self.train_pause = QPushButton(self.train_functions_frame)
        self.train_pause.setObjectName(u"train_pause")
        self.train_pause.setFont(font15)

        self.verticalLayout_33.addWidget(self.train_pause)

        self.train_stop = QPushButton(self.train_functions_frame)
        self.train_stop.setObjectName(u"train_stop")
        self.train_stop.setFont(font15)

        self.verticalLayout_33.addWidget(self.train_stop)


        self.verticalLayout_22.addWidget(self.train_functions_frame)

        self.settings_pages.addWidget(self.training_page)
        self.testing_page = QWidget()
        self.testing_page.setObjectName(u"testing_page")
        self.verticalLayout_43 = QVBoxLayout(self.testing_page)
        self.verticalLayout_43.setSpacing(3)
        self.verticalLayout_43.setObjectName(u"verticalLayout_43")
        self.verticalLayout_43.setContentsMargins(0, 0, 0, 0)
        self.scrollArea_10 = QScrollArea(self.testing_page)
        self.scrollArea_10.setObjectName(u"scrollArea_10")
        self.scrollArea_10.setFrameShadow(QFrame.Raised)
        self.scrollArea_10.setWidgetResizable(True)
        self.testing_settings_container = QWidget()
        self.testing_settings_container.setObjectName(u"testing_settings_container")
        self.testing_settings_container.setGeometry(QRect(0, 0, 437, 768))
        self.verticalLayout_44 = QVBoxLayout(self.testing_settings_container)
        self.verticalLayout_44.setObjectName(u"verticalLayout_44")
        self.verticalLayout_44.setContentsMargins(0, 0, 0, 0)
        self.frame_38 = QFrame(self.testing_settings_container)
        self.frame_38.setObjectName(u"frame_38")
        self.frame_38.setFrameShape(QFrame.StyledPanel)
        self.frame_38.setFrameShadow(QFrame.Raised)
        self.verticalLayout_41 = QVBoxLayout(self.frame_38)
        self.verticalLayout_41.setSpacing(3)
        self.verticalLayout_41.setObjectName(u"verticalLayout_41")
        self.verticalLayout_41.setContentsMargins(3, 3, 3, 3)
        self.frame_80 = QFrame(self.frame_38)
        self.frame_80.setObjectName(u"frame_80")
        self.frame_80.setFrameShape(QFrame.StyledPanel)
        self.frame_80.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_69 = QHBoxLayout(self.frame_80)
        self.horizontalLayout_69.setObjectName(u"horizontalLayout_69")
        self.horizontalLayout_69.setContentsMargins(0, 0, 0, 0)
        self.label_123 = QLabel(self.frame_80)
        self.label_123.setObjectName(u"label_123")
        sizePolicy17 = QSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Maximum)
        sizePolicy17.setHorizontalStretch(0)
        sizePolicy17.setVerticalStretch(0)
        sizePolicy17.setHeightForWidth(self.label_123.sizePolicy().hasHeightForWidth())
        self.label_123.setSizePolicy(sizePolicy17)
        self.label_123.setFont(font4)

        self.horizontalLayout_69.addWidget(self.label_123)

        self.line_23 = QFrame(self.frame_80)
        self.line_23.setObjectName(u"line_23")
        self.line_23.setFrameShadow(QFrame.Raised)
        self.line_23.setFrameShape(QFrame.Shape.HLine)

        self.horizontalLayout_69.addWidget(self.line_23)


        self.verticalLayout_41.addWidget(self.frame_80)

        self.test_experiment_path_frame = QFrame(self.frame_38)
        self.test_experiment_path_frame.setObjectName(u"test_experiment_path_frame")
        self.test_experiment_path_frame.setFrameShape(QFrame.StyledPanel)
        self.test_experiment_path_frame.setFrameShadow(QFrame.Raised)
        self.verticalLayout_24 = QVBoxLayout(self.test_experiment_path_frame)
        self.verticalLayout_24.setObjectName(u"verticalLayout_24")
        self.verticalLayout_24.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_39 = QHBoxLayout()
        self.horizontalLayout_39.setSpacing(6)
        self.horizontalLayout_39.setObjectName(u"horizontalLayout_39")
        self.test_experiment_path = QLineEdit(self.test_experiment_path_frame)
        self.test_experiment_path.setObjectName(u"test_experiment_path")

        self.horizontalLayout_39.addWidget(self.test_experiment_path)

        self.test_browse_experiment = QPushButton(self.test_experiment_path_frame)
        self.test_browse_experiment.setObjectName(u"test_browse_experiment")

        self.horizontalLayout_39.addWidget(self.test_browse_experiment)


        self.verticalLayout_24.addLayout(self.horizontalLayout_39)


        self.verticalLayout_41.addWidget(self.test_experiment_path_frame)

        self.frame_82 = QFrame(self.frame_38)
        self.frame_82.setObjectName(u"frame_82")
        self.frame_82.setFrameShape(QFrame.StyledPanel)
        self.frame_82.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_71 = QHBoxLayout(self.frame_82)
        self.horizontalLayout_71.setObjectName(u"horizontalLayout_71")
        self.horizontalLayout_71.setContentsMargins(0, 0, 0, 0)
        self.label_98 = QLabel(self.frame_82)
        self.label_98.setObjectName(u"label_98")
        sizePolicy1.setHeightForWidth(self.label_98.sizePolicy().hasHeightForWidth())
        self.label_98.setSizePolicy(sizePolicy1)
        self.label_98.setFont(font4)
        self.label_98.setFrameShape(QFrame.NoFrame)
        self.label_98.setFrameShadow(QFrame.Raised)

        self.horizontalLayout_71.addWidget(self.label_98)

        self.line_25 = QFrame(self.frame_82)
        self.line_25.setObjectName(u"line_25")
        self.line_25.setFrameShadow(QFrame.Raised)
        self.line_25.setFrameShape(QFrame.Shape.HLine)

        self.horizontalLayout_71.addWidget(self.line_25)


        self.verticalLayout_41.addWidget(self.frame_82)

        self.frame_68 = QFrame(self.frame_38)
        self.frame_68.setObjectName(u"frame_68")
        self.frame_68.setFrameShape(QFrame.StyledPanel)
        self.frame_68.setFrameShadow(QFrame.Raised)
        self.formLayout_25 = QFormLayout(self.frame_68)
        self.formLayout_25.setObjectName(u"formLayout_25")
        self.formLayout_25.setContentsMargins(0, 0, 0, 0)
        self.label_58 = QLabel(self.frame_68)
        self.label_58.setObjectName(u"label_58")

        self.formLayout_25.setWidget(1, QFormLayout.ItemRole.LabelRole, self.label_58)

        self.horizontalLayout_45 = QHBoxLayout()
        self.horizontalLayout_45.setObjectName(u"horizontalLayout_45")
        self.test_iterations_slider = QSlider(self.frame_68)
        self.test_iterations_slider.setObjectName(u"test_iterations_slider")
        self.test_iterations_slider.setMinimum(1)
        self.test_iterations_slider.setMaximum(999)
        self.test_iterations_slider.setValue(10)
        self.test_iterations_slider.setOrientation(Qt.Horizontal)

        self.horizontalLayout_45.addWidget(self.test_iterations_slider)

        self.test_iterations = QSpinBox(self.frame_68)
        self.test_iterations.setObjectName(u"test_iterations")
        self.test_iterations.setMinimum(1)
        self.test_iterations.setMaximum(99999)
        self.test_iterations.setSingleStep(5)
        self.test_iterations.setValue(10)

        self.horizontalLayout_45.addWidget(self.test_iterations)


        self.formLayout_25.setLayout(1, QFormLayout.ItemRole.FieldRole, self.horizontalLayout_45)

        self.label_92 = QLabel(self.frame_68)
        self.label_92.setObjectName(u"label_92")

        self.formLayout_25.setWidget(0, QFormLayout.ItemRole.LabelRole, self.label_92)

        self.horizontalLayout_65 = QHBoxLayout()
        self.horizontalLayout_65.setSpacing(6)
        self.horizontalLayout_65.setObjectName(u"horizontalLayout_65")
        self.test_load_epoch = QSpinBox(self.frame_68)
        self.test_load_epoch.setObjectName(u"test_load_epoch")
        self.test_load_epoch.setMinimum(1)
        self.test_load_epoch.setMaximum(999999)

        self.horizontalLayout_65.addWidget(self.test_load_epoch)

        self.test_get_most_recent = QPushButton(self.frame_68)
        self.test_get_most_recent.setObjectName(u"test_get_most_recent")
        sizePolicy8.setHeightForWidth(self.test_get_most_recent.sizePolicy().hasHeightForWidth())
        self.test_get_most_recent.setSizePolicy(sizePolicy8)

        self.horizontalLayout_65.addWidget(self.test_get_most_recent)


        self.formLayout_25.setLayout(0, QFormLayout.ItemRole.FieldRole, self.horizontalLayout_65)


        self.verticalLayout_41.addWidget(self.frame_68)

        self.frame_81 = QFrame(self.frame_38)
        self.frame_81.setObjectName(u"frame_81")
        self.frame_81.setFrameShape(QFrame.StyledPanel)
        self.frame_81.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_70 = QHBoxLayout(self.frame_81)
        self.horizontalLayout_70.setObjectName(u"horizontalLayout_70")
        self.horizontalLayout_70.setContentsMargins(0, 0, 0, 0)
        self.test_override_dataset = QCheckBox(self.frame_81)
        self.test_override_dataset.setObjectName(u"test_override_dataset")
        sizePolicy8.setHeightForWidth(self.test_override_dataset.sizePolicy().hasHeightForWidth())
        self.test_override_dataset.setSizePolicy(sizePolicy8)
        self.test_override_dataset.setFont(font4)

        self.horizontalLayout_70.addWidget(self.test_override_dataset)

        self.line_24 = QFrame(self.frame_81)
        self.line_24.setObjectName(u"line_24")
        self.line_24.setFrameShadow(QFrame.Raised)
        self.line_24.setFrameShape(QFrame.Shape.HLine)

        self.horizontalLayout_70.addWidget(self.line_24)


        self.verticalLayout_41.addWidget(self.frame_81)

        self.test_dataset_path_frame = QFrame(self.frame_38)
        self.test_dataset_path_frame.setObjectName(u"test_dataset_path_frame")
        self.test_dataset_path_frame.setFrameShape(QFrame.StyledPanel)
        self.test_dataset_path_frame.setFrameShadow(QFrame.Raised)
        self.verticalLayout_61 = QVBoxLayout(self.test_dataset_path_frame)
        self.verticalLayout_61.setObjectName(u"verticalLayout_61")
        self.verticalLayout_61.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_64 = QHBoxLayout()
        self.horizontalLayout_64.setSpacing(6)
        self.horizontalLayout_64.setObjectName(u"horizontalLayout_64")
        self.test_dataset_path = QLineEdit(self.test_dataset_path_frame)
        self.test_dataset_path.setObjectName(u"test_dataset_path")

        self.horizontalLayout_64.addWidget(self.test_dataset_path)

        self.test_browse_dataset = QPushButton(self.test_dataset_path_frame)
        self.test_browse_dataset.setObjectName(u"test_browse_dataset")

        self.horizontalLayout_64.addWidget(self.test_browse_dataset)


        self.verticalLayout_61.addLayout(self.horizontalLayout_64)


        self.verticalLayout_41.addWidget(self.test_dataset_path_frame)

        self.verticalSpacer = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.verticalLayout_41.addItem(self.verticalSpacer)


        self.verticalLayout_44.addWidget(self.frame_38)

        self.scrollArea_10.setWidget(self.testing_settings_container)

        self.verticalLayout_43.addWidget(self.scrollArea_10)

        self.test_warning_label = QLabel(self.testing_page)
        self.test_warning_label.setObjectName(u"test_warning_label")
        self.test_warning_label.setAlignment(Qt.AlignCenter)

        self.verticalLayout_43.addWidget(self.test_warning_label)

        self.frame_79 = QFrame(self.testing_page)
        self.frame_79.setObjectName(u"frame_79")
        self.frame_79.setFrameShape(QFrame.Panel)
        self.frame_79.setFrameShadow(QFrame.Sunken)
        self.frame_79.setLineWidth(2)
        self.verticalLayout_57 = QVBoxLayout(self.frame_79)
        self.verticalLayout_57.setObjectName(u"verticalLayout_57")
        self.verticalLayout_57.setContentsMargins(3, 3, 3, 3)
        self.test_start = QPushButton(self.frame_79)
        self.test_start.setObjectName(u"test_start")
        self.test_start.setFont(font15)

        self.verticalLayout_57.addWidget(self.test_start)

        self.test_progress_label = QLabel(self.frame_79)
        self.test_progress_label.setObjectName(u"test_progress_label")
        sizePolicy.setHeightForWidth(self.test_progress_label.sizePolicy().hasHeightForWidth())
        self.test_progress_label.setSizePolicy(sizePolicy)
        self.test_progress_label.setAlignment(Qt.AlignCenter)

        self.verticalLayout_57.addWidget(self.test_progress_label)


        self.verticalLayout_43.addWidget(self.frame_79)

        self.settings_pages.addWidget(self.testing_page)
        self.review_page = QWidget()
        self.review_page.setObjectName(u"review_page")
        self.verticalLayout_45 = QVBoxLayout(self.review_page)
        self.verticalLayout_45.setObjectName(u"verticalLayout_45")
        self.verticalLayout_45.setContentsMargins(0, 0, 0, 0)
        self.scrollArea_11 = QScrollArea(self.review_page)
        self.scrollArea_11.setObjectName(u"scrollArea_11")
        self.scrollArea_11.setFrameShadow(QFrame.Raised)
        self.scrollArea_11.setWidgetResizable(True)
        self.review_settings_container = QWidget()
        self.review_settings_container.setObjectName(u"review_settings_container")
        self.review_settings_container.setGeometry(QRect(0, 0, 437, 861))
        self.verticalLayout_46 = QVBoxLayout(self.review_settings_container)
        self.verticalLayout_46.setObjectName(u"verticalLayout_46")
        self.frame_89 = QFrame(self.review_settings_container)
        self.frame_89.setObjectName(u"frame_89")
        self.frame_89.setFrameShape(QFrame.StyledPanel)
        self.frame_89.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_80 = QHBoxLayout(self.frame_89)
        self.horizontalLayout_80.setObjectName(u"horizontalLayout_80")
        self.horizontalLayout_80.setContentsMargins(0, 0, 0, 0)
        self.label_100 = QLabel(self.frame_89)
        self.label_100.setObjectName(u"label_100")
        sizePolicy1.setHeightForWidth(self.label_100.sizePolicy().hasHeightForWidth())
        self.label_100.setSizePolicy(sizePolicy1)
        self.label_100.setFont(font4)
        self.label_100.setFrameShape(QFrame.NoFrame)
        self.label_100.setFrameShadow(QFrame.Raised)

        self.horizontalLayout_80.addWidget(self.label_100)

        self.line_32 = QFrame(self.frame_89)
        self.line_32.setObjectName(u"line_32")
        self.line_32.setFrameShadow(QFrame.Raised)
        self.line_32.setFrameShape(QFrame.Shape.HLine)

        self.horizontalLayout_80.addWidget(self.line_32)


        self.verticalLayout_46.addWidget(self.frame_89)

        self.horizontalLayout_81 = QHBoxLayout()
        self.horizontalLayout_81.setSpacing(6)
        self.horizontalLayout_81.setObjectName(u"horizontalLayout_81")
        self.review_experiment_path = QLineEdit(self.review_settings_container)
        self.review_experiment_path.setObjectName(u"review_experiment_path")

        self.horizontalLayout_81.addWidget(self.review_experiment_path)

        self.review_browse_experiment = QPushButton(self.review_settings_container)
        self.review_browse_experiment.setObjectName(u"review_browse_experiment")

        self.horizontalLayout_81.addWidget(self.review_browse_experiment)


        self.verticalLayout_46.addLayout(self.horizontalLayout_81)

        self.review_load_experiment = QPushButton(self.review_settings_container)
        self.review_load_experiment.setObjectName(u"review_load_experiment")
        font16 = QFont()
        font16.setPointSize(12)
        font16.setBold(False)
        self.review_load_experiment.setFont(font16)

        self.verticalLayout_46.addWidget(self.review_load_experiment)

        self.frame_93 = QFrame(self.review_settings_container)
        self.frame_93.setObjectName(u"frame_93")
        self.frame_93.setFrameShape(QFrame.StyledPanel)
        self.frame_93.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_82 = QHBoxLayout(self.frame_93)
        self.horizontalLayout_82.setObjectName(u"horizontalLayout_82")
        self.horizontalLayout_82.setContentsMargins(0, 0, 0, 0)
        self.label_102 = QLabel(self.frame_93)
        self.label_102.setObjectName(u"label_102")
        sizePolicy1.setHeightForWidth(self.label_102.sizePolicy().hasHeightForWidth())
        self.label_102.setSizePolicy(sizePolicy1)
        self.label_102.setFont(font4)
        self.label_102.setFrameShape(QFrame.NoFrame)
        self.label_102.setFrameShadow(QFrame.Raised)

        self.horizontalLayout_82.addWidget(self.label_102)

        self.line_39 = QFrame(self.frame_93)
        self.line_39.setObjectName(u"line_39")
        self.line_39.setFrameShadow(QFrame.Raised)
        self.line_39.setFrameShape(QFrame.Shape.HLine)

        self.horizontalLayout_82.addWidget(self.line_39)

        self.review_select_train_config = QComboBox(self.frame_93)
        self.review_select_train_config.setObjectName(u"review_select_train_config")

        self.horizontalLayout_82.addWidget(self.review_select_train_config)


        self.verticalLayout_46.addWidget(self.frame_93)

        self.review_train_config = QTextEdit(self.review_settings_container)
        self.review_train_config.setObjectName(u"review_train_config")
        self.review_train_config.setFrameShape(QFrame.Panel)
        self.review_train_config.setLineWidth(1)
        self.review_train_config.setReadOnly(True)
        self.review_train_config.setTextInteractionFlags(Qt.NoTextInteraction)

        self.verticalLayout_46.addWidget(self.review_train_config)

        self.scrollArea_11.setWidget(self.review_settings_container)

        self.verticalLayout_45.addWidget(self.scrollArea_11)

        self.settings_pages.addWidget(self.review_page)
        self.utilities_page = QWidget()
        self.utilities_page.setObjectName(u"utilities_page")
        self.verticalLayout_47 = QVBoxLayout(self.utilities_page)
        self.verticalLayout_47.setObjectName(u"verticalLayout_47")
        self.verticalLayout_47.setContentsMargins(0, 0, 0, 0)
        self.scrollArea_12 = QScrollArea(self.utilities_page)
        self.scrollArea_12.setObjectName(u"scrollArea_12")
        self.scrollArea_12.setFrameShadow(QFrame.Raised)
        self.scrollArea_12.setWidgetResizable(True)
        self.utilities_settings_container = QWidget()
        self.utilities_settings_container.setObjectName(u"utilities_settings_container")
        self.utilities_settings_container.setGeometry(QRect(0, 0, 420, 1634))
        self.verticalLayout_48 = QVBoxLayout(self.utilities_settings_container)
        self.verticalLayout_48.setObjectName(u"verticalLayout_48")
        self.frame_95 = QFrame(self.utilities_settings_container)
        self.frame_95.setObjectName(u"frame_95")
        sizePolicy.setHeightForWidth(self.frame_95.sizePolicy().hasHeightForWidth())
        self.frame_95.setSizePolicy(sizePolicy)
        self.frame_95.setFrameShape(QFrame.StyledPanel)
        self.frame_95.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_83 = QHBoxLayout(self.frame_95)
        self.horizontalLayout_83.setObjectName(u"horizontalLayout_83")
        self.horizontalLayout_83.setContentsMargins(0, 0, 0, 0)
        self.line_42 = QFrame(self.frame_95)
        self.line_42.setObjectName(u"line_42")
        self.line_42.setFrameShadow(QFrame.Raised)
        self.line_42.setFrameShape(QFrame.Shape.HLine)

        self.horizontalLayout_83.addWidget(self.line_42)

        self.label_106 = QLabel(self.frame_95)
        self.label_106.setObjectName(u"label_106")
        sizePolicy17.setHeightForWidth(self.label_106.sizePolicy().hasHeightForWidth())
        self.label_106.setSizePolicy(sizePolicy17)
        self.label_106.setFont(font4)
        self.label_106.setFrameShape(QFrame.NoFrame)
        self.label_106.setFrameShadow(QFrame.Raised)

        self.horizontalLayout_83.addWidget(self.label_106)

        self.line_40 = QFrame(self.frame_95)
        self.line_40.setObjectName(u"line_40")
        self.line_40.setFrameShadow(QFrame.Raised)
        self.line_40.setFrameShape(QFrame.Shape.HLine)

        self.horizontalLayout_83.addWidget(self.line_40)


        self.verticalLayout_48.addWidget(self.frame_95)

        self.convert_to_onnx = QPushButton(self.utilities_settings_container)
        self.convert_to_onnx.setObjectName(u"convert_to_onnx")
        self.convert_to_onnx.setCheckable(True)

        self.verticalLayout_48.addWidget(self.convert_to_onnx)

        self.utils_convert_onnx_frame = QFrame(self.utilities_settings_container)
        self.utils_convert_onnx_frame.setObjectName(u"utils_convert_onnx_frame")
        self.utils_convert_onnx_frame.setFrameShape(QFrame.Panel)
        self.utils_convert_onnx_frame.setFrameShadow(QFrame.Sunken)
        self.utils_convert_onnx_layout = QVBoxLayout(self.utils_convert_onnx_frame)
        self.utils_convert_onnx_layout.setObjectName(u"utils_convert_onnx_layout")
        self.utils_convert_onnx_layout.setContentsMargins(2, 2, 2, 2)
        self.label_120 = QLabel(self.utils_convert_onnx_frame)
        self.label_120.setObjectName(u"label_120")
        sizePolicy2.setHeightForWidth(self.label_120.sizePolicy().hasHeightForWidth())
        self.label_120.setSizePolicy(sizePolicy2)
        self.label_120.setAlignment(Qt.AlignCenter)

        self.utils_convert_onnx_layout.addWidget(self.label_120)

        self.frame_108 = QFrame(self.utils_convert_onnx_frame)
        self.frame_108.setObjectName(u"frame_108")
        self.frame_108.setFrameShape(QFrame.StyledPanel)
        self.frame_108.setFrameShadow(QFrame.Raised)
        self.formLayout_33 = QFormLayout(self.frame_108)
        self.formLayout_33.setObjectName(u"formLayout_33")
        self.formLayout_33.setContentsMargins(0, 0, 0, 0)
        self.label_111 = QLabel(self.frame_108)
        self.label_111.setObjectName(u"label_111")

        self.formLayout_33.setWidget(0, QFormLayout.ItemRole.LabelRole, self.label_111)

        self.horizontalLayout_93 = QHBoxLayout()
        self.horizontalLayout_93.setObjectName(u"horizontalLayout_93")
        self.convert_onnx_experiment = QLineEdit(self.frame_108)
        self.convert_onnx_experiment.setObjectName(u"convert_onnx_experiment")

        self.horizontalLayout_93.addWidget(self.convert_onnx_experiment)

        self.convert_onnx_browse_experiment = QPushButton(self.frame_108)
        self.convert_onnx_browse_experiment.setObjectName(u"convert_onnx_browse_experiment")
        sizePolicy18 = QSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        sizePolicy18.setHorizontalStretch(0)
        sizePolicy18.setVerticalStretch(0)
        sizePolicy18.setHeightForWidth(self.convert_onnx_browse_experiment.sizePolicy().hasHeightForWidth())
        self.convert_onnx_browse_experiment.setSizePolicy(sizePolicy18)

        self.horizontalLayout_93.addWidget(self.convert_onnx_browse_experiment)


        self.formLayout_33.setLayout(0, QFormLayout.ItemRole.FieldRole, self.horizontalLayout_93)

        self.label_112 = QLabel(self.frame_108)
        self.label_112.setObjectName(u"label_112")

        self.formLayout_33.setWidget(1, QFormLayout.ItemRole.LabelRole, self.label_112)

        self.horizontalLayout_95 = QHBoxLayout()
        self.horizontalLayout_95.setObjectName(u"horizontalLayout_95")
        self.convert_onnx_load_epoch = QSpinBox(self.frame_108)
        self.convert_onnx_load_epoch.setObjectName(u"convert_onnx_load_epoch")
        self.convert_onnx_load_epoch.setMinimum(1)
        self.convert_onnx_load_epoch.setMaximum(999999)

        self.horizontalLayout_95.addWidget(self.convert_onnx_load_epoch)

        self.convert_onnx_load_latest_epoch = QPushButton(self.frame_108)
        self.convert_onnx_load_latest_epoch.setObjectName(u"convert_onnx_load_latest_epoch")
        sizePolicy8.setHeightForWidth(self.convert_onnx_load_latest_epoch.sizePolicy().hasHeightForWidth())
        self.convert_onnx_load_latest_epoch.setSizePolicy(sizePolicy8)

        self.horizontalLayout_95.addWidget(self.convert_onnx_load_latest_epoch)


        self.formLayout_33.setLayout(1, QFormLayout.ItemRole.FieldRole, self.horizontalLayout_95)

        self.label_118 = QLabel(self.frame_108)
        self.label_118.setObjectName(u"label_118")

        self.formLayout_33.setWidget(2, QFormLayout.ItemRole.LabelRole, self.label_118)

        self.convert_onnx_in_channels = QSpinBox(self.frame_108)
        self.convert_onnx_in_channels.setObjectName(u"convert_onnx_in_channels")
        self.convert_onnx_in_channels.setMinimum(3)
        self.convert_onnx_in_channels.setMaximum(3)
        self.convert_onnx_in_channels.setValue(3)

        self.formLayout_33.setWidget(2, QFormLayout.ItemRole.FieldRole, self.convert_onnx_in_channels)

        self.label_113 = QLabel(self.frame_108)
        self.label_113.setObjectName(u"label_113")

        self.formLayout_33.setWidget(3, QFormLayout.ItemRole.LabelRole, self.label_113)

        self.convert_onnx_crop_size = QComboBox(self.frame_108)
        self.convert_onnx_crop_size.setObjectName(u"convert_onnx_crop_size")

        self.formLayout_33.setWidget(3, QFormLayout.ItemRole.FieldRole, self.convert_onnx_crop_size)

        self.label_114 = QLabel(self.frame_108)
        self.label_114.setObjectName(u"label_114")

        self.formLayout_33.setWidget(4, QFormLayout.ItemRole.LabelRole, self.label_114)

        self.convert_onnx_device = QComboBox(self.frame_108)
        self.convert_onnx_device.setObjectName(u"convert_onnx_device")

        self.formLayout_33.setWidget(4, QFormLayout.ItemRole.FieldRole, self.convert_onnx_device)

        self.label_115 = QLabel(self.frame_108)
        self.label_115.setObjectName(u"label_115")

        self.formLayout_33.setWidget(5, QFormLayout.ItemRole.LabelRole, self.label_115)

        self.convert_onnx_opset_version = QComboBox(self.frame_108)
        self.convert_onnx_opset_version.setObjectName(u"convert_onnx_opset_version")

        self.formLayout_33.setWidget(5, QFormLayout.ItemRole.FieldRole, self.convert_onnx_opset_version)


        self.utils_convert_onnx_layout.addWidget(self.frame_108)

        self.frame_109 = QFrame(self.utils_convert_onnx_frame)
        self.frame_109.setObjectName(u"frame_109")
        self.frame_109.setFrameShape(QFrame.StyledPanel)
        self.frame_109.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_105 = QHBoxLayout(self.frame_109)
        self.horizontalLayout_105.setObjectName(u"horizontalLayout_105")
        self.horizontalLayout_105.setContentsMargins(30, 0, 30, 0)
        self.convert_onnx_export_params = QCheckBox(self.frame_109)
        self.convert_onnx_export_params.setObjectName(u"convert_onnx_export_params")
        sizePolicy8.setHeightForWidth(self.convert_onnx_export_params.sizePolicy().hasHeightForWidth())
        self.convert_onnx_export_params.setSizePolicy(sizePolicy8)
        self.convert_onnx_export_params.setChecked(True)

        self.horizontalLayout_105.addWidget(self.convert_onnx_export_params)

        self.convert_onnx_fold_constants = QCheckBox(self.frame_109)
        self.convert_onnx_fold_constants.setObjectName(u"convert_onnx_fold_constants")
        sizePolicy8.setHeightForWidth(self.convert_onnx_fold_constants.sizePolicy().hasHeightForWidth())
        self.convert_onnx_fold_constants.setSizePolicy(sizePolicy8)
        self.convert_onnx_fold_constants.setChecked(True)

        self.horizontalLayout_105.addWidget(self.convert_onnx_fold_constants)


        self.utils_convert_onnx_layout.addWidget(self.frame_109)

        self.convert_onnx_start = QPushButton(self.utils_convert_onnx_frame)
        self.convert_onnx_start.setObjectName(u"convert_onnx_start")

        self.utils_convert_onnx_layout.addWidget(self.convert_onnx_start)


        self.verticalLayout_48.addWidget(self.utils_convert_onnx_frame)

        self.test_onnx_model = QPushButton(self.utilities_settings_container)
        self.test_onnx_model.setObjectName(u"test_onnx_model")
        self.test_onnx_model.setCheckable(True)

        self.verticalLayout_48.addWidget(self.test_onnx_model)

        self.utils_test_onnx_frame = QFrame(self.utilities_settings_container)
        self.utils_test_onnx_frame.setObjectName(u"utils_test_onnx_frame")
        self.utils_test_onnx_frame.setFrameShape(QFrame.Panel)
        self.utils_test_onnx_frame.setFrameShadow(QFrame.Sunken)
        self.verticalLayout_62 = QVBoxLayout(self.utils_test_onnx_frame)
        self.verticalLayout_62.setObjectName(u"verticalLayout_62")
        self.verticalLayout_62.setContentsMargins(2, 2, 2, 2)
        self.label_119 = QLabel(self.utils_test_onnx_frame)
        self.label_119.setObjectName(u"label_119")
        sizePolicy2.setHeightForWidth(self.label_119.sizePolicy().hasHeightForWidth())
        self.label_119.setSizePolicy(sizePolicy2)
        self.label_119.setAlignment(Qt.AlignCenter)

        self.verticalLayout_62.addWidget(self.label_119)

        self.frame_110 = QFrame(self.utils_test_onnx_frame)
        self.frame_110.setObjectName(u"frame_110")
        self.frame_110.setFrameShape(QFrame.StyledPanel)
        self.frame_110.setFrameShadow(QFrame.Raised)
        self.formLayout_32 = QFormLayout(self.frame_110)
        self.formLayout_32.setObjectName(u"formLayout_32")
        self.formLayout_32.setContentsMargins(0, 0, 0, 0)
        self.label_116 = QLabel(self.frame_110)
        self.label_116.setObjectName(u"label_116")

        self.formLayout_32.setWidget(0, QFormLayout.ItemRole.LabelRole, self.label_116)

        self.horizontalLayout_106 = QHBoxLayout()
        self.horizontalLayout_106.setObjectName(u"horizontalLayout_106")
        self.test_onnx_model_path = QLineEdit(self.frame_110)
        self.test_onnx_model_path.setObjectName(u"test_onnx_model_path")

        self.horizontalLayout_106.addWidget(self.test_onnx_model_path)

        self.browse_onnx_model_path = QPushButton(self.frame_110)
        self.browse_onnx_model_path.setObjectName(u"browse_onnx_model_path")
        sizePolicy18.setHeightForWidth(self.browse_onnx_model_path.sizePolicy().hasHeightForWidth())
        self.browse_onnx_model_path.setSizePolicy(sizePolicy18)

        self.horizontalLayout_106.addWidget(self.browse_onnx_model_path)


        self.formLayout_32.setLayout(0, QFormLayout.ItemRole.FieldRole, self.horizontalLayout_106)

        self.label_117 = QLabel(self.frame_110)
        self.label_117.setObjectName(u"label_117")

        self.formLayout_32.setWidget(1, QFormLayout.ItemRole.LabelRole, self.label_117)

        self.horizontalLayout_107 = QHBoxLayout()
        self.horizontalLayout_107.setObjectName(u"horizontalLayout_107")
        self.test_onnx_image_dir = QLineEdit(self.frame_110)
        self.test_onnx_image_dir.setObjectName(u"test_onnx_image_dir")

        self.horizontalLayout_107.addWidget(self.test_onnx_image_dir)

        self.browse_onnx_image_dir = QPushButton(self.frame_110)
        self.browse_onnx_image_dir.setObjectName(u"browse_onnx_image_dir")

        self.horizontalLayout_107.addWidget(self.browse_onnx_image_dir)


        self.formLayout_32.setLayout(1, QFormLayout.ItemRole.FieldRole, self.horizontalLayout_107)


        self.verticalLayout_62.addWidget(self.frame_110)

        self.test_onnx_start = QPushButton(self.utils_test_onnx_frame)
        self.test_onnx_start.setObjectName(u"test_onnx_start")

        self.verticalLayout_62.addWidget(self.test_onnx_start)


        self.verticalLayout_48.addWidget(self.utils_test_onnx_frame)

        self.frame_96 = QFrame(self.utilities_settings_container)
        self.frame_96.setObjectName(u"frame_96")
        self.frame_96.setFrameShape(QFrame.StyledPanel)
        self.frame_96.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_84 = QHBoxLayout(self.frame_96)
        self.horizontalLayout_84.setObjectName(u"horizontalLayout_84")
        self.horizontalLayout_84.setContentsMargins(0, 0, 0, 0)
        self.line_43 = QFrame(self.frame_96)
        self.line_43.setObjectName(u"line_43")
        self.line_43.setFrameShadow(QFrame.Raised)
        self.line_43.setFrameShape(QFrame.Shape.HLine)

        self.horizontalLayout_84.addWidget(self.line_43)

        self.label_107 = QLabel(self.frame_96)
        self.label_107.setObjectName(u"label_107")
        sizePolicy1.setHeightForWidth(self.label_107.sizePolicy().hasHeightForWidth())
        self.label_107.setSizePolicy(sizePolicy1)
        self.label_107.setFont(font4)
        self.label_107.setFrameShape(QFrame.NoFrame)
        self.label_107.setFrameShadow(QFrame.Raised)

        self.horizontalLayout_84.addWidget(self.label_107)

        self.line_41 = QFrame(self.frame_96)
        self.line_41.setObjectName(u"line_41")
        self.line_41.setFrameShadow(QFrame.Raised)
        self.line_41.setFrameShape(QFrame.Shape.HLine)

        self.horizontalLayout_84.addWidget(self.line_41)


        self.verticalLayout_48.addWidget(self.frame_96)

        self.pair_images = QPushButton(self.utilities_settings_container)
        self.pair_images.setObjectName(u"pair_images")
        self.pair_images.setCheckable(True)

        self.verticalLayout_48.addWidget(self.pair_images)

        self.utils_pair_images_frame = QFrame(self.utilities_settings_container)
        self.utils_pair_images_frame.setObjectName(u"utils_pair_images_frame")
        self.utils_pair_images_frame.setFrameShape(QFrame.Panel)
        self.utils_pair_images_frame.setFrameShadow(QFrame.Sunken)
        self.utils_pair_images_layout = QVBoxLayout(self.utils_pair_images_frame)
        self.utils_pair_images_layout.setSpacing(3)
        self.utils_pair_images_layout.setObjectName(u"utils_pair_images_layout")
        self.utils_pair_images_layout.setContentsMargins(2, 2, 2, 2)
        self.label_89 = QLabel(self.utils_pair_images_frame)
        self.label_89.setObjectName(u"label_89")
        sizePolicy2.setHeightForWidth(self.label_89.sizePolicy().hasHeightForWidth())
        self.label_89.setSizePolicy(sizePolicy2)
        self.label_89.setAlignment(Qt.AlignCenter)

        self.utils_pair_images_layout.addWidget(self.label_89)

        self.frame_83 = QFrame(self.utils_pair_images_frame)
        self.frame_83.setObjectName(u"frame_83")
        self.frame_83.setFrameShape(QFrame.StyledPanel)
        self.frame_83.setFrameShadow(QFrame.Raised)
        self.formLayout_23 = QFormLayout(self.frame_83)
        self.formLayout_23.setObjectName(u"formLayout_23")
        self.formLayout_23.setHorizontalSpacing(6)
        self.formLayout_23.setContentsMargins(-1, 0, 0, 0)
        self.label_64 = QLabel(self.frame_83)
        self.label_64.setObjectName(u"label_64")

        self.formLayout_23.setWidget(0, QFormLayout.ItemRole.LabelRole, self.label_64)

        self.horizontalLayout_85 = QHBoxLayout()
        self.horizontalLayout_85.setObjectName(u"horizontalLayout_85")
        self.pair_images_input_a = QLineEdit(self.frame_83)
        self.pair_images_input_a.setObjectName(u"pair_images_input_a")

        self.horizontalLayout_85.addWidget(self.pair_images_input_a)

        self.browse_pair_input_a = QPushButton(self.frame_83)
        self.browse_pair_input_a.setObjectName(u"browse_pair_input_a")
        sizePolicy18.setHeightForWidth(self.browse_pair_input_a.sizePolicy().hasHeightForWidth())
        self.browse_pair_input_a.setSizePolicy(sizePolicy18)

        self.horizontalLayout_85.addWidget(self.browse_pair_input_a)


        self.formLayout_23.setLayout(0, QFormLayout.ItemRole.FieldRole, self.horizontalLayout_85)

        self.label_66 = QLabel(self.frame_83)
        self.label_66.setObjectName(u"label_66")

        self.formLayout_23.setWidget(1, QFormLayout.ItemRole.LabelRole, self.label_66)

        self.horizontalLayout_86 = QHBoxLayout()
        self.horizontalLayout_86.setObjectName(u"horizontalLayout_86")
        self.pair_images_input_b = QLineEdit(self.frame_83)
        self.pair_images_input_b.setObjectName(u"pair_images_input_b")

        self.horizontalLayout_86.addWidget(self.pair_images_input_b)

        self.browse_pair_input_b = QPushButton(self.frame_83)
        self.browse_pair_input_b.setObjectName(u"browse_pair_input_b")
        sizePolicy18.setHeightForWidth(self.browse_pair_input_b.sizePolicy().hasHeightForWidth())
        self.browse_pair_input_b.setSizePolicy(sizePolicy18)

        self.horizontalLayout_86.addWidget(self.browse_pair_input_b)


        self.formLayout_23.setLayout(1, QFormLayout.ItemRole.FieldRole, self.horizontalLayout_86)

        self.label_67 = QLabel(self.frame_83)
        self.label_67.setObjectName(u"label_67")

        self.formLayout_23.setWidget(2, QFormLayout.ItemRole.LabelRole, self.label_67)

        self.horizontalLayout_87 = QHBoxLayout()
        self.horizontalLayout_87.setObjectName(u"horizontalLayout_87")
        self.pair_images_output = QLineEdit(self.frame_83)
        self.pair_images_output.setObjectName(u"pair_images_output")

        self.horizontalLayout_87.addWidget(self.pair_images_output)

        self.browse_pair_output = QPushButton(self.frame_83)
        self.browse_pair_output.setObjectName(u"browse_pair_output")
        sizePolicy18.setHeightForWidth(self.browse_pair_output.sizePolicy().hasHeightForWidth())
        self.browse_pair_output.setSizePolicy(sizePolicy18)

        self.horizontalLayout_87.addWidget(self.browse_pair_output)


        self.formLayout_23.setLayout(2, QFormLayout.ItemRole.FieldRole, self.horizontalLayout_87)


        self.utils_pair_images_layout.addWidget(self.frame_83)

        self.frame_102 = QFrame(self.utils_pair_images_frame)
        self.frame_102.setObjectName(u"frame_102")
        self.frame_102.setFrameShape(QFrame.StyledPanel)
        self.frame_102.setFrameShadow(QFrame.Raised)
        self.formLayout_29 = QFormLayout(self.frame_102)
        self.formLayout_29.setObjectName(u"formLayout_29")
        self.label_77 = QLabel(self.frame_102)
        self.label_77.setObjectName(u"label_77")

        self.formLayout_29.setWidget(0, QFormLayout.ItemRole.LabelRole, self.label_77)

        self.pair_images_direction = QComboBox(self.frame_102)
        self.pair_images_direction.setObjectName(u"pair_images_direction")

        self.formLayout_29.setWidget(0, QFormLayout.ItemRole.FieldRole, self.pair_images_direction)

        self.horizontalLayout_94 = QHBoxLayout()
        self.horizontalLayout_94.setObjectName(u"horizontalLayout_94")
        self.pair_images_do_scaling = QCheckBox(self.frame_102)
        self.pair_images_do_scaling.setObjectName(u"pair_images_do_scaling")
        sizePolicy8.setHeightForWidth(self.pair_images_do_scaling.sizePolicy().hasHeightForWidth())
        self.pair_images_do_scaling.setSizePolicy(sizePolicy8)
        self.pair_images_do_scaling.setLayoutDirection(Qt.LeftToRight)

        self.horizontalLayout_94.addWidget(self.pair_images_do_scaling)

        self.pair_images_scale = QComboBox(self.frame_102)
        self.pair_images_scale.setObjectName(u"pair_images_scale")

        self.horizontalLayout_94.addWidget(self.pair_images_scale)


        self.formLayout_29.setLayout(1, QFormLayout.ItemRole.FieldRole, self.horizontalLayout_94)

        self.label_97 = QLabel(self.frame_102)
        self.label_97.setObjectName(u"label_97")

        self.formLayout_29.setWidget(1, QFormLayout.ItemRole.LabelRole, self.label_97)


        self.utils_pair_images_layout.addWidget(self.frame_102)

        self.frame_98 = QFrame(self.utils_pair_images_frame)
        self.frame_98.setObjectName(u"frame_98")
        self.frame_98.setFrameShape(QFrame.StyledPanel)
        self.frame_98.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_88 = QHBoxLayout(self.frame_98)
        self.horizontalLayout_88.setObjectName(u"horizontalLayout_88")
        self.horizontalLayout_88.setContentsMargins(0, 0, 0, 0)
        self.start_image_pairing = QPushButton(self.frame_98)
        self.start_image_pairing.setObjectName(u"start_image_pairing")

        self.horizontalLayout_88.addWidget(self.start_image_pairing)

        self.preview_image_pairing = QPushButton(self.frame_98)
        self.preview_image_pairing.setObjectName(u"preview_image_pairing")
        sizePolicy8.setHeightForWidth(self.preview_image_pairing.sizePolicy().hasHeightForWidth())
        self.preview_image_pairing.setSizePolicy(sizePolicy8)

        self.horizontalLayout_88.addWidget(self.preview_image_pairing)


        self.utils_pair_images_layout.addWidget(self.frame_98)


        self.verticalLayout_48.addWidget(self.utils_pair_images_frame)

        self.frame_97 = QFrame(self.utilities_settings_container)
        self.frame_97.setObjectName(u"frame_97")
        self.frame_97.setFrameShape(QFrame.StyledPanel)
        self.frame_97.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_16 = QHBoxLayout(self.frame_97)
        self.horizontalLayout_16.setObjectName(u"horizontalLayout_16")
        self.horizontalLayout_16.setContentsMargins(0, 0, 0, 0)
        self.sort_images = QPushButton(self.frame_97)
        self.sort_images.setObjectName(u"sort_images")
        self.sort_images.setCheckable(True)

        self.horizontalLayout_16.addWidget(self.sort_images)

        self.remove_sorting_tags = QPushButton(self.frame_97)
        self.remove_sorting_tags.setObjectName(u"remove_sorting_tags")
        self.remove_sorting_tags.setCheckable(True)

        self.horizontalLayout_16.addWidget(self.remove_sorting_tags)

        self.copy_image_sort = QPushButton(self.frame_97)
        self.copy_image_sort.setObjectName(u"copy_image_sort")
        self.copy_image_sort.setCheckable(True)

        self.horizontalLayout_16.addWidget(self.copy_image_sort)


        self.verticalLayout_48.addWidget(self.frame_97)

        self.utils_sort_images_frame = QFrame(self.utilities_settings_container)
        self.utils_sort_images_frame.setObjectName(u"utils_sort_images_frame")
        self.utils_sort_images_frame.setFrameShape(QFrame.Panel)
        self.utils_sort_images_frame.setFrameShadow(QFrame.Sunken)
        self.utils_sort_images_layout = QVBoxLayout(self.utils_sort_images_frame)
        self.utils_sort_images_layout.setObjectName(u"utils_sort_images_layout")
        self.utils_sort_images_layout.setContentsMargins(2, 2, 2, 2)
        self.label_90 = QLabel(self.utils_sort_images_frame)
        self.label_90.setObjectName(u"label_90")
        sizePolicy2.setHeightForWidth(self.label_90.sizePolicy().hasHeightForWidth())
        self.label_90.setSizePolicy(sizePolicy2)
        self.label_90.setAlignment(Qt.AlignCenter)

        self.utils_sort_images_layout.addWidget(self.label_90)

        self.frame_99 = QFrame(self.utils_sort_images_frame)
        self.frame_99.setObjectName(u"frame_99")
        self.frame_99.setFrameShape(QFrame.StyledPanel)
        self.frame_99.setFrameShadow(QFrame.Raised)
        self.formLayout_24 = QFormLayout(self.frame_99)
        self.formLayout_24.setObjectName(u"formLayout_24")
        self.formLayout_24.setContentsMargins(0, 0, 0, 0)
        self.label_76 = QLabel(self.frame_99)
        self.label_76.setObjectName(u"label_76")

        self.formLayout_24.setWidget(0, QFormLayout.ItemRole.LabelRole, self.label_76)

        self.horizontalLayout_89 = QHBoxLayout()
        self.horizontalLayout_89.setObjectName(u"horizontalLayout_89")
        self.sort_images_input_dir = QLineEdit(self.frame_99)
        self.sort_images_input_dir.setObjectName(u"sort_images_input_dir")

        self.horizontalLayout_89.addWidget(self.sort_images_input_dir)

        self.browse_sort_input_dir = QPushButton(self.frame_99)
        self.browse_sort_input_dir.setObjectName(u"browse_sort_input_dir")
        sizePolicy18.setHeightForWidth(self.browse_sort_input_dir.sizePolicy().hasHeightForWidth())
        self.browse_sort_input_dir.setSizePolicy(sizePolicy18)

        self.horizontalLayout_89.addWidget(self.browse_sort_input_dir)


        self.formLayout_24.setLayout(0, QFormLayout.ItemRole.FieldRole, self.horizontalLayout_89)

        self.label_78 = QLabel(self.frame_99)
        self.label_78.setObjectName(u"label_78")

        self.formLayout_24.setWidget(1, QFormLayout.ItemRole.LabelRole, self.label_78)

        self.sort_images_type = QComboBox(self.frame_99)
        self.sort_images_type.setObjectName(u"sort_images_type")

        self.formLayout_24.setWidget(1, QFormLayout.ItemRole.FieldRole, self.sort_images_type)

        self.label_81 = QLabel(self.frame_99)
        self.label_81.setObjectName(u"label_81")

        self.formLayout_24.setWidget(2, QFormLayout.ItemRole.LabelRole, self.label_81)

        self.sort_images_direction = QComboBox(self.frame_99)
        self.sort_images_direction.setObjectName(u"sort_images_direction")

        self.formLayout_24.setWidget(2, QFormLayout.ItemRole.FieldRole, self.sort_images_direction)


        self.utils_sort_images_layout.addWidget(self.frame_99)

        self.frame_90 = QFrame(self.utils_sort_images_frame)
        self.frame_90.setObjectName(u"frame_90")
        self.frame_90.setFrameShape(QFrame.StyledPanel)
        self.frame_90.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_96 = QHBoxLayout(self.frame_90)
        self.horizontalLayout_96.setObjectName(u"horizontalLayout_96")
        self.horizontalLayout_96.setContentsMargins(0, 0, 0, 0)
        self.start_sort_images = QPushButton(self.frame_90)
        self.start_sort_images.setObjectName(u"start_sort_images")

        self.horizontalLayout_96.addWidget(self.start_sort_images)

        self.preview_image_sorting = QPushButton(self.frame_90)
        self.preview_image_sorting.setObjectName(u"preview_image_sorting")
        sizePolicy8.setHeightForWidth(self.preview_image_sorting.sizePolicy().hasHeightForWidth())
        self.preview_image_sorting.setSizePolicy(sizePolicy8)

        self.horizontalLayout_96.addWidget(self.preview_image_sorting)


        self.utils_sort_images_layout.addWidget(self.frame_90)


        self.verticalLayout_48.addWidget(self.utils_sort_images_frame)

        self.utils_remove_sorting_tags_frame = QFrame(self.utilities_settings_container)
        self.utils_remove_sorting_tags_frame.setObjectName(u"utils_remove_sorting_tags_frame")
        self.utils_remove_sorting_tags_frame.setFrameShape(QFrame.Panel)
        self.utils_remove_sorting_tags_frame.setFrameShadow(QFrame.Sunken)
        self.utils_remove_sorting_tags_layout = QVBoxLayout(self.utils_remove_sorting_tags_frame)
        self.utils_remove_sorting_tags_layout.setObjectName(u"utils_remove_sorting_tags_layout")
        self.utils_remove_sorting_tags_layout.setContentsMargins(2, 2, 2, 2)
        self.label_91 = QLabel(self.utils_remove_sorting_tags_frame)
        self.label_91.setObjectName(u"label_91")
        sizePolicy2.setHeightForWidth(self.label_91.sizePolicy().hasHeightForWidth())
        self.label_91.setSizePolicy(sizePolicy2)
        self.label_91.setAlignment(Qt.AlignCenter)

        self.utils_remove_sorting_tags_layout.addWidget(self.label_91)

        self.frame_100 = QFrame(self.utils_remove_sorting_tags_frame)
        self.frame_100.setObjectName(u"frame_100")
        self.frame_100.setFrameShape(QFrame.StyledPanel)
        self.frame_100.setFrameShadow(QFrame.Raised)
        self.formLayout_26 = QFormLayout(self.frame_100)
        self.formLayout_26.setObjectName(u"formLayout_26")
        self.formLayout_26.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_90 = QHBoxLayout()
        self.horizontalLayout_90.setObjectName(u"horizontalLayout_90")
        self.remove_sort_tags_input = QLineEdit(self.frame_100)
        self.remove_sort_tags_input.setObjectName(u"remove_sort_tags_input")

        self.horizontalLayout_90.addWidget(self.remove_sort_tags_input)

        self.browse_remove_sort_tags_input = QPushButton(self.frame_100)
        self.browse_remove_sort_tags_input.setObjectName(u"browse_remove_sort_tags_input")

        self.horizontalLayout_90.addWidget(self.browse_remove_sort_tags_input)


        self.formLayout_26.setLayout(0, QFormLayout.ItemRole.FieldRole, self.horizontalLayout_90)

        self.label_82 = QLabel(self.frame_100)
        self.label_82.setObjectName(u"label_82")

        self.formLayout_26.setWidget(0, QFormLayout.ItemRole.LabelRole, self.label_82)


        self.utils_remove_sorting_tags_layout.addWidget(self.frame_100)

        self.frame_103 = QFrame(self.utils_remove_sorting_tags_frame)
        self.frame_103.setObjectName(u"frame_103")
        self.frame_103.setFrameShape(QFrame.StyledPanel)
        self.frame_103.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_97 = QHBoxLayout(self.frame_103)
        self.horizontalLayout_97.setObjectName(u"horizontalLayout_97")
        self.horizontalLayout_97.setContentsMargins(0, 0, 0, 0)
        self.start_remove_sort_tags = QPushButton(self.frame_103)
        self.start_remove_sort_tags.setObjectName(u"start_remove_sort_tags")

        self.horizontalLayout_97.addWidget(self.start_remove_sort_tags)

        self.preview_remove_sort_tags = QPushButton(self.frame_103)
        self.preview_remove_sort_tags.setObjectName(u"preview_remove_sort_tags")
        sizePolicy8.setHeightForWidth(self.preview_remove_sort_tags.sizePolicy().hasHeightForWidth())
        self.preview_remove_sort_tags.setSizePolicy(sizePolicy8)

        self.horizontalLayout_97.addWidget(self.preview_remove_sort_tags)


        self.utils_remove_sorting_tags_layout.addWidget(self.frame_103)


        self.verticalLayout_48.addWidget(self.utils_remove_sorting_tags_frame)

        self.utils_copy_sort_frame = QFrame(self.utilities_settings_container)
        self.utils_copy_sort_frame.setObjectName(u"utils_copy_sort_frame")
        self.utils_copy_sort_frame.setFrameShape(QFrame.Panel)
        self.utils_copy_sort_frame.setFrameShadow(QFrame.Sunken)
        self.utils_copy_sort_layout = QVBoxLayout(self.utils_copy_sort_frame)
        self.utils_copy_sort_layout.setObjectName(u"utils_copy_sort_layout")
        self.utils_copy_sort_layout.setContentsMargins(2, 2, 2, 2)
        self.label_96 = QLabel(self.utils_copy_sort_frame)
        self.label_96.setObjectName(u"label_96")
        sizePolicy2.setHeightForWidth(self.label_96.sizePolicy().hasHeightForWidth())
        self.label_96.setSizePolicy(sizePolicy2)
        self.label_96.setAlignment(Qt.AlignCenter)

        self.utils_copy_sort_layout.addWidget(self.label_96)

        self.frame_101 = QFrame(self.utils_copy_sort_frame)
        self.frame_101.setObjectName(u"frame_101")
        self.frame_101.setFrameShape(QFrame.StyledPanel)
        self.frame_101.setFrameShadow(QFrame.Raised)
        self.formLayout_28 = QFormLayout(self.frame_101)
        self.formLayout_28.setObjectName(u"formLayout_28")
        self.formLayout_28.setContentsMargins(0, 0, 0, 0)
        self.label_83 = QLabel(self.frame_101)
        self.label_83.setObjectName(u"label_83")

        self.formLayout_28.setWidget(0, QFormLayout.ItemRole.LabelRole, self.label_83)

        self.horizontalLayout_91 = QHBoxLayout()
        self.horizontalLayout_91.setObjectName(u"horizontalLayout_91")
        self.copy_sort_copy_from = QLineEdit(self.frame_101)
        self.copy_sort_copy_from.setObjectName(u"copy_sort_copy_from")

        self.horizontalLayout_91.addWidget(self.copy_sort_copy_from)

        self.browse_copy_sort_copy_from = QPushButton(self.frame_101)
        self.browse_copy_sort_copy_from.setObjectName(u"browse_copy_sort_copy_from")

        self.horizontalLayout_91.addWidget(self.browse_copy_sort_copy_from)


        self.formLayout_28.setLayout(0, QFormLayout.ItemRole.FieldRole, self.horizontalLayout_91)

        self.label_88 = QLabel(self.frame_101)
        self.label_88.setObjectName(u"label_88")

        self.formLayout_28.setWidget(1, QFormLayout.ItemRole.LabelRole, self.label_88)

        self.horizontalLayout_92 = QHBoxLayout()
        self.horizontalLayout_92.setObjectName(u"horizontalLayout_92")
        self.copy_sort_copy_to = QLineEdit(self.frame_101)
        self.copy_sort_copy_to.setObjectName(u"copy_sort_copy_to")

        self.horizontalLayout_92.addWidget(self.copy_sort_copy_to)

        self.browse_copy_sort_copy_to = QPushButton(self.frame_101)
        self.browse_copy_sort_copy_to.setObjectName(u"browse_copy_sort_copy_to")

        self.horizontalLayout_92.addWidget(self.browse_copy_sort_copy_to)


        self.formLayout_28.setLayout(1, QFormLayout.ItemRole.FieldRole, self.horizontalLayout_92)


        self.utils_copy_sort_layout.addWidget(self.frame_101)

        self.frame_104 = QFrame(self.utils_copy_sort_frame)
        self.frame_104.setObjectName(u"frame_104")
        self.frame_104.setFrameShape(QFrame.StyledPanel)
        self.frame_104.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_98 = QHBoxLayout(self.frame_104)
        self.horizontalLayout_98.setObjectName(u"horizontalLayout_98")
        self.horizontalLayout_98.setContentsMargins(0, 0, 0, 0)
        self.start_copy_sort = QPushButton(self.frame_104)
        self.start_copy_sort.setObjectName(u"start_copy_sort")

        self.horizontalLayout_98.addWidget(self.start_copy_sort)

        self.preview_copy_sort = QPushButton(self.frame_104)
        self.preview_copy_sort.setObjectName(u"preview_copy_sort")
        sizePolicy8.setHeightForWidth(self.preview_copy_sort.sizePolicy().hasHeightForWidth())
        self.preview_copy_sort.setSizePolicy(sizePolicy8)

        self.horizontalLayout_98.addWidget(self.preview_copy_sort)


        self.utils_copy_sort_layout.addWidget(self.frame_104)


        self.verticalLayout_48.addWidget(self.utils_copy_sort_frame)

        self.split_dataset = QPushButton(self.utilities_settings_container)
        self.split_dataset.setObjectName(u"split_dataset")
        self.split_dataset.setCheckable(True)

        self.verticalLayout_48.addWidget(self.split_dataset)

        self.utils_split_dataset_frame = QFrame(self.utilities_settings_container)
        self.utils_split_dataset_frame.setObjectName(u"utils_split_dataset_frame")
        self.utils_split_dataset_frame.setFrameShape(QFrame.Panel)
        self.utils_split_dataset_frame.setFrameShadow(QFrame.Sunken)
        self.utils_split_dataset_layout = QVBoxLayout(self.utils_split_dataset_frame)
        self.utils_split_dataset_layout.setObjectName(u"utils_split_dataset_layout")
        self.utils_split_dataset_layout.setContentsMargins(2, 2, 2, 2)
        self.frame_106 = QFrame(self.utils_split_dataset_frame)
        self.frame_106.setObjectName(u"frame_106")
        self.frame_106.setFrameShape(QFrame.StyledPanel)
        self.frame_106.setFrameShadow(QFrame.Raised)
        self.formLayout_31 = QFormLayout(self.frame_106)
        self.formLayout_31.setObjectName(u"formLayout_31")
        self.formLayout_31.setContentsMargins(0, 0, 0, 0)
        self.label_109 = QLabel(self.frame_106)
        self.label_109.setObjectName(u"label_109")

        self.formLayout_31.setWidget(0, QFormLayout.ItemRole.LabelRole, self.label_109)

        self.horizontalLayout_102 = QHBoxLayout()
        self.horizontalLayout_102.setObjectName(u"horizontalLayout_102")
        self.split_dataset_input = QLineEdit(self.frame_106)
        self.split_dataset_input.setObjectName(u"split_dataset_input")

        self.horizontalLayout_102.addWidget(self.split_dataset_input)

        self.browse_split_dataset_input = QPushButton(self.frame_106)
        self.browse_split_dataset_input.setObjectName(u"browse_split_dataset_input")

        self.horizontalLayout_102.addWidget(self.browse_split_dataset_input)


        self.formLayout_31.setLayout(0, QFormLayout.ItemRole.FieldRole, self.horizontalLayout_102)

        self.label_110 = QLabel(self.frame_106)
        self.label_110.setObjectName(u"label_110")

        self.formLayout_31.setWidget(1, QFormLayout.ItemRole.LabelRole, self.label_110)

        self.horizontalLayout_103 = QHBoxLayout()
        self.horizontalLayout_103.setObjectName(u"horizontalLayout_103")
        self.split_dataset_output = QLineEdit(self.frame_106)
        self.split_dataset_output.setObjectName(u"split_dataset_output")

        self.horizontalLayout_103.addWidget(self.split_dataset_output)

        self.browse_split_dataset_output = QPushButton(self.frame_106)
        self.browse_split_dataset_output.setObjectName(u"browse_split_dataset_output")

        self.horizontalLayout_103.addWidget(self.browse_split_dataset_output)


        self.formLayout_31.setLayout(1, QFormLayout.ItemRole.FieldRole, self.horizontalLayout_103)


        self.utils_split_dataset_layout.addWidget(self.frame_106)

        self.frame_105 = QFrame(self.utils_split_dataset_frame)
        self.frame_105.setObjectName(u"frame_105")
        self.frame_105.setFrameShape(QFrame.StyledPanel)
        self.frame_105.setFrameShadow(QFrame.Raised)
        self.formLayout_30 = QFormLayout(self.frame_105)
        self.formLayout_30.setObjectName(u"formLayout_30")
        self.formLayout_30.setContentsMargins(0, 0, 0, 0)
        self.label_79 = QLabel(self.frame_105)
        self.label_79.setObjectName(u"label_79")

        self.formLayout_30.setWidget(0, QFormLayout.ItemRole.LabelRole, self.label_79)

        self.horizontalLayout_99 = QHBoxLayout()
        self.horizontalLayout_99.setObjectName(u"horizontalLayout_99")
        self.split_dataset_test_slider = QSlider(self.frame_105)
        self.split_dataset_test_slider.setObjectName(u"split_dataset_test_slider")
        self.split_dataset_test_slider.setMaximum(100)
        self.split_dataset_test_slider.setValue(10)
        self.split_dataset_test_slider.setOrientation(Qt.Horizontal)

        self.horizontalLayout_99.addWidget(self.split_dataset_test_slider)

        self.split_dataset_test = QSpinBox(self.frame_105)
        self.split_dataset_test.setObjectName(u"split_dataset_test")
        self.split_dataset_test.setMaximum(100)
        self.split_dataset_test.setValue(10)

        self.horizontalLayout_99.addWidget(self.split_dataset_test)


        self.formLayout_30.setLayout(0, QFormLayout.ItemRole.FieldRole, self.horizontalLayout_99)

        self.label_99 = QLabel(self.frame_105)
        self.label_99.setObjectName(u"label_99")

        self.formLayout_30.setWidget(1, QFormLayout.ItemRole.LabelRole, self.label_99)

        self.label_108 = QLabel(self.frame_105)
        self.label_108.setObjectName(u"label_108")

        self.formLayout_30.setWidget(2, QFormLayout.ItemRole.LabelRole, self.label_108)

        self.horizontalLayout_100 = QHBoxLayout()
        self.horizontalLayout_100.setObjectName(u"horizontalLayout_100")
        self.split_dataset_train_slider = QSlider(self.frame_105)
        self.split_dataset_train_slider.setObjectName(u"split_dataset_train_slider")
        self.split_dataset_train_slider.setMaximum(100)
        self.split_dataset_train_slider.setValue(80)
        self.split_dataset_train_slider.setOrientation(Qt.Horizontal)

        self.horizontalLayout_100.addWidget(self.split_dataset_train_slider)

        self.split_dataset_train = QSpinBox(self.frame_105)
        self.split_dataset_train.setObjectName(u"split_dataset_train")
        self.split_dataset_train.setMaximum(100)
        self.split_dataset_train.setValue(80)

        self.horizontalLayout_100.addWidget(self.split_dataset_train)


        self.formLayout_30.setLayout(1, QFormLayout.ItemRole.FieldRole, self.horizontalLayout_100)

        self.horizontalLayout_101 = QHBoxLayout()
        self.horizontalLayout_101.setObjectName(u"horizontalLayout_101")
        self.split_dataset_val_slider = QSlider(self.frame_105)
        self.split_dataset_val_slider.setObjectName(u"split_dataset_val_slider")
        self.split_dataset_val_slider.setMaximum(100)
        self.split_dataset_val_slider.setValue(10)
        self.split_dataset_val_slider.setOrientation(Qt.Horizontal)

        self.horizontalLayout_101.addWidget(self.split_dataset_val_slider)

        self.split_dataset_val = QSpinBox(self.frame_105)
        self.split_dataset_val.setObjectName(u"split_dataset_val")
        self.split_dataset_val.setMaximum(100)
        self.split_dataset_val.setValue(10)

        self.horizontalLayout_101.addWidget(self.split_dataset_val)


        self.formLayout_30.setLayout(2, QFormLayout.ItemRole.FieldRole, self.horizontalLayout_101)


        self.utils_split_dataset_layout.addWidget(self.frame_105)

        self.frame_107 = QFrame(self.utils_split_dataset_frame)
        self.frame_107.setObjectName(u"frame_107")
        self.frame_107.setFrameShape(QFrame.StyledPanel)
        self.frame_107.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_104 = QHBoxLayout(self.frame_107)
        self.horizontalLayout_104.setObjectName(u"horizontalLayout_104")
        self.horizontalLayout_104.setContentsMargins(0, 0, 0, 0)
        self.start_split_dataset = QPushButton(self.frame_107)
        self.start_split_dataset.setObjectName(u"start_split_dataset")

        self.horizontalLayout_104.addWidget(self.start_split_dataset)

        self.preview_split_dataset = QPushButton(self.frame_107)
        self.preview_split_dataset.setObjectName(u"preview_split_dataset")
        sizePolicy8.setHeightForWidth(self.preview_split_dataset.sizePolicy().hasHeightForWidth())
        self.preview_split_dataset.setSizePolicy(sizePolicy8)

        self.horizontalLayout_104.addWidget(self.preview_split_dataset)


        self.utils_split_dataset_layout.addWidget(self.frame_107)


        self.verticalLayout_48.addWidget(self.utils_split_dataset_frame)

        self.verticalSpacer_15 = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.verticalLayout_48.addItem(self.verticalSpacer_15)

        self.scrollArea_12.setWidget(self.utilities_settings_container)

        self.verticalLayout_47.addWidget(self.scrollArea_12)

        self.utils_warning_label = QLabel(self.utilities_page)
        self.utils_warning_label.setObjectName(u"utils_warning_label")
        self.utils_warning_label.setAlignment(Qt.AlignCenter)

        self.verticalLayout_47.addWidget(self.utils_warning_label)

        self.image_utils_warning = QLabel(self.utilities_page)
        self.image_utils_warning.setObjectName(u"image_utils_warning")
        sizePolicy.setHeightForWidth(self.image_utils_warning.sizePolicy().hasHeightForWidth())
        self.image_utils_warning.setSizePolicy(sizePolicy)
        self.image_utils_warning.setFont(font10)
        self.image_utils_warning.setAlignment(Qt.AlignCenter)

        self.verticalLayout_47.addWidget(self.image_utils_warning)

        self.settings_pages.addWidget(self.utilities_page)
        self.settings_page = QWidget()
        self.settings_page.setObjectName(u"settings_page")
        self.verticalLayout_49 = QVBoxLayout(self.settings_page)
        self.verticalLayout_49.setObjectName(u"verticalLayout_49")
        self.verticalLayout_49.setContentsMargins(0, 0, 0, 0)
        self.scrollArea_13 = QScrollArea(self.settings_page)
        self.scrollArea_13.setObjectName(u"scrollArea_13")
        self.scrollArea_13.setFrameShadow(QFrame.Raised)
        self.scrollArea_13.setWidgetResizable(True)
        self.setting_settings_container = QWidget()
        self.setting_settings_container.setObjectName(u"setting_settings_container")
        self.setting_settings_container.setGeometry(QRect(0, 0, 437, 861))
        self.verticalLayout_50 = QVBoxLayout(self.setting_settings_container)
        self.verticalLayout_50.setObjectName(u"verticalLayout_50")
        self.verticalLayout_50.setContentsMargins(0, 0, 0, 0)
        self.frame_53 = QFrame(self.setting_settings_container)
        self.frame_53.setObjectName(u"frame_53")
        self.frame_53.setFrameShape(QFrame.StyledPanel)
        self.frame_53.setFrameShadow(QFrame.Raised)
        self.verticalLayout_42 = QVBoxLayout(self.frame_53)
        self.verticalLayout_42.setObjectName(u"verticalLayout_42")
        self.always_on_top = QCheckBox(self.frame_53)
        self.always_on_top.setObjectName(u"always_on_top")
        self.always_on_top.setChecked(True)

        self.verticalLayout_42.addWidget(self.always_on_top)

        self.line_38 = QFrame(self.frame_53)
        self.line_38.setObjectName(u"line_38")
        self.line_38.setFrameShape(QFrame.Shape.HLine)
        self.line_38.setFrameShadow(QFrame.Shadow.Sunken)

        self.verticalLayout_42.addWidget(self.line_38)

        self.frame_94 = QFrame(self.frame_53)
        self.frame_94.setObjectName(u"frame_94")
        self.frame_94.setFrameShape(QFrame.StyledPanel)
        self.frame_94.setFrameShadow(QFrame.Raised)
        self.formLayout_9 = QFormLayout(self.frame_94)
        self.formLayout_9.setObjectName(u"formLayout_9")
        self.formLayout_9.setContentsMargins(0, 0, 0, 0)
        self.label_40 = QLabel(self.frame_94)
        self.label_40.setObjectName(u"label_40")

        self.formLayout_9.setWidget(0, QFormLayout.ItemRole.LabelRole, self.label_40)

        self.review_graph_sample_rate = QSpinBox(self.frame_94)
        self.review_graph_sample_rate.setObjectName(u"review_graph_sample_rate")
        self.review_graph_sample_rate.setMinimum(1)
        self.review_graph_sample_rate.setMaximum(99999)
        self.review_graph_sample_rate.setSingleStep(5)
        self.review_graph_sample_rate.setValue(50)

        self.formLayout_9.setWidget(0, QFormLayout.ItemRole.FieldRole, self.review_graph_sample_rate)

        self.label_44 = QLabel(self.frame_94)
        self.label_44.setObjectName(u"label_44")

        self.formLayout_9.setWidget(1, QFormLayout.ItemRole.LabelRole, self.label_44)

        self.horizontalLayout_31 = QHBoxLayout()
        self.horizontalLayout_31.setObjectName(u"horizontalLayout_31")
        self.update_frequency_slider = QSlider(self.frame_94)
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

        self.update_frequency = QSpinBox(self.frame_94)
        self.update_frequency.setObjectName(u"update_frequency")
        self.update_frequency.setMinimum(1)
        self.update_frequency.setMaximum(99999)
        self.update_frequency.setSingleStep(10)
        self.update_frequency.setValue(50)

        self.horizontalLayout_31.addWidget(self.update_frequency)


        self.formLayout_9.setLayout(1, QFormLayout.ItemRole.FieldRole, self.horizontalLayout_31)


        self.verticalLayout_42.addWidget(self.frame_94)

        self.line_26 = QFrame(self.frame_53)
        self.line_26.setObjectName(u"line_26")
        self.line_26.setFrameShape(QFrame.Shape.HLine)
        self.line_26.setFrameShadow(QFrame.Shadow.Sunken)

        self.verticalLayout_42.addWidget(self.line_26)

        self.verticalSpacer_13 = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.verticalLayout_42.addItem(self.verticalSpacer_13)


        self.verticalLayout_50.addWidget(self.frame_53)

        self.scrollArea_13.setWidget(self.setting_settings_container)

        self.verticalLayout_49.addWidget(self.scrollArea_13)

        self.settings_pages.addWidget(self.settings_page)

        self.verticalLayout_13.addWidget(self.settings_pages)


        self.verticalLayout_2.addWidget(self.train_settings)


        self.horizontalLayout_5.addWidget(self.frame_2)

        self.settings_dock.setWidget(self.dockWidgetContents_8)
        MainWindow.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.settings_dock)

        self.retranslateUi(MainWindow)

        self.centralwidget_pages.setCurrentIndex(2)
        self.settings_pages.setCurrentIndex(5)
        self.direction.setCurrentIndex(-1)
        self.tabWidget_2.setCurrentIndex(0)
        self.pair_images_scale.setCurrentIndex(-1)


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
        self.test_a_real_label.setText(QCoreApplication.translate("MainWindow", u"A_real", None))
        self.test_b_fake_label.setText(QCoreApplication.translate("MainWindow", u"B_fake", None))
        self.test_b_real_label.setText(QCoreApplication.translate("MainWindow", u"B_real", None))
        self.test_info_label.setText(QCoreApplication.translate("MainWindow", u"Info", None))
        self.label_93.setText(QCoreApplication.translate("MainWindow", u"Test Version", None))
        self.label_80.setText(QCoreApplication.translate("MainWindow", u"Image Scale", None))
        self.test_image_scale_reset.setText(QCoreApplication.translate("MainWindow", u"Reset Scale", None))
        self.label_94.setText(QCoreApplication.translate("MainWindow", u"Sort by", None))
        self.label_23.setText(QCoreApplication.translate("MainWindow", u"Example Images", None))
        self.label_25.setText(QCoreApplication.translate("MainWindow", u"Loss Graphs", None))
        self.label_121.setText(QCoreApplication.translate("MainWindow", u"Preview Results", None))
        self.label_122.setText(QCoreApplication.translate("MainWindow", u"Preview Images", None))
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
        self.label_104.setText(QCoreApplication.translate("MainWindow", u"Output Settings", None))
        self.label_2.setText(QCoreApplication.translate("MainWindow", u"Output Root", None))
        self.browse_output.setText(QCoreApplication.translate("MainWindow", u"Browse", None))
        self.label_3.setText(QCoreApplication.translate("MainWindow", u"Experiment Name", None))
        self.label_4.setText(QCoreApplication.translate("MainWindow", u"Version", None))
        self.label_105.setText(QCoreApplication.translate("MainWindow", u"Generator Architecture", None))
        self.label_87.setText(QCoreApplication.translate("MainWindow", u"Upsample Type", None))
#if QT_CONFIG(tooltip)
        self.label_124.setToolTip(QCoreApplication.translate("MainWindow", u"Number of features on the first conv layer.", None))
#endif // QT_CONFIG(tooltip)
        self.label_124.setText(QCoreApplication.translate("MainWindow", u"Features", None))
        self.label_125.setText(QCoreApplication.translate("MainWindow", u"Down Layers", None))
        self.label_126.setText(QCoreApplication.translate("MainWindow", u"Block Type", None))
        self.label_103.setText(QCoreApplication.translate("MainWindow", u"Discriminator Architecture", None))
        self.label_84.setText(QCoreApplication.translate("MainWindow", u"Layer Count", None))
        self.label_85.setText(QCoreApplication.translate("MainWindow", u"Base Channels", None))
        self.label_86.setText(QCoreApplication.translate("MainWindow", u"Max Channels", None))
        self.load_from_config.setText(QCoreApplication.translate("MainWindow", u"Load Settings from Config File", None))
        self.label_101.setText(QCoreApplication.translate("MainWindow", u"Dataset Files", None))
        self.label.setText(QCoreApplication.translate("MainWindow", u"Dataset Root", None))
        self.browse_dataset.setText(QCoreApplication.translate("MainWindow", u"Browse", None))
        self.label_10.setText(QCoreApplication.translate("MainWindow", u"Load Size", None))
        self.label_11.setText(QCoreApplication.translate("MainWindow", u"Crop Size", None))
        self.label_9.setText(QCoreApplication.translate("MainWindow", u"Input Channels", None))
        self.label_95.setText(QCoreApplication.translate("MainWindow", u"Loading", None))
        self.label_5.setText(QCoreApplication.translate("MainWindow", u"Direction", None))
        self.direction.setCurrentText("")
        self.label_8.setText(QCoreApplication.translate("MainWindow", u"Batch Size", None))
        self.groupBox_2.setTitle(QCoreApplication.translate("MainWindow", u"Augmentations", None))
        self.show_input_transforms.setText(QCoreApplication.translate("MainWindow", u"Input", None))
        self.label_74.setText(QCoreApplication.translate("MainWindow", u"Colorjitter", None))
        self.colorjitter_chance.setSuffix(QCoreApplication.translate("MainWindow", u"%", None))
        self.label_75.setText(QCoreApplication.translate("MainWindow", u"Colorjitter Range", None))
        self.label_127.setText(QCoreApplication.translate("MainWindow", u"Gaussian Noise", None))
        self.gaussnoise_chance.setSuffix(QCoreApplication.translate("MainWindow", u"%", None))
        self.label_128.setText(QCoreApplication.translate("MainWindow", u"Gaussian Range", None))
        self.label_129.setText(QCoreApplication.translate("MainWindow", u"Motion Blur", None))
        self.motionblur_chance.setSuffix(QCoreApplication.translate("MainWindow", u"%", None))
        self.label_130.setText(QCoreApplication.translate("MainWindow", u"Motion Blur Limit", None))
        self.label_131.setText(QCoreApplication.translate("MainWindow", u"Random Gamma", None))
        self.randgamma_chance.setSuffix(QCoreApplication.translate("MainWindow", u"%", None))
        self.label_133.setText(QCoreApplication.translate("MainWindow", u"Gamma Limits", None))
        self.label_132.setText(QCoreApplication.translate("MainWindow", u"Grayscale", None))
        self.grayscale_chance.setSuffix(QCoreApplication.translate("MainWindow", u"%", None))
        self.label_134.setText(QCoreApplication.translate("MainWindow", u"Grayscale Method", None))
        self.label_135.setText(QCoreApplication.translate("MainWindow", u"Compression", None))
        self.label_136.setText(QCoreApplication.translate("MainWindow", u"Compression Type", None))
        self.label_137.setText(QCoreApplication.translate("MainWindow", u"Compression Quality", None))
        self.compression_chance.setSuffix(QCoreApplication.translate("MainWindow", u"%", None))
        self.show_both_transforms.setText(QCoreApplication.translate("MainWindow", u"Both", None))
        self.label_71.setText(QCoreApplication.translate("MainWindow", u"Horizontal Flip", None))
        self.h_flip_chance.setSuffix(QCoreApplication.translate("MainWindow", u"%", None))
        self.label_72.setText(QCoreApplication.translate("MainWindow", u"Vertical Flip", None))
        self.label_73.setText(QCoreApplication.translate("MainWindow", u"90\u00b0 Rotation", None))
        self.v_flip_chance.setSuffix(QCoreApplication.translate("MainWindow", u"%", None))
        self.rot90_chance.setSuffix(QCoreApplication.translate("MainWindow", u"%", None))
        self.label_138.setText(QCoreApplication.translate("MainWindow", u"Elastic Xform", None))
        self.elastic_transform_chance.setSuffix(QCoreApplication.translate("MainWindow", u"%", None))
        self.label_139.setText(QCoreApplication.translate("MainWindow", u"Elastic Alpha", None))
        self.label_140.setText(QCoreApplication.translate("MainWindow", u"Elastic Sigma", None))
        self.elastic_transform_alpha.setSuffix("")
        self.elastic_transform_sigma.setSuffix("")
        self.label_141.setText(QCoreApplication.translate("MainWindow", u"Optical Distort", None))
        self.optical_distortion_chance.setSuffix(QCoreApplication.translate("MainWindow", u"%", None))
        self.label_142.setText(QCoreApplication.translate("MainWindow", u"Distortion Limits", None))
        self.label_143.setText(QCoreApplication.translate("MainWindow", u"Distortion Mode", None))
        self.label_144.setText(QCoreApplication.translate("MainWindow", u"Coarse Dropout", None))
        self.label_145.setText(QCoreApplication.translate("MainWindow", u"Hole Count", None))
        self.label_146.setText(QCoreApplication.translate("MainWindow", u"Hole Height", None))
        self.label_147.setText(QCoreApplication.translate("MainWindow", u"Hole Width", None))
        self.coarse_dropout_chance.setSuffix(QCoreApplication.translate("MainWindow", u"%", None))
        self.continue_train.setText(QCoreApplication.translate("MainWindow", u"Continue Train", None))
        self.label_16.setText(QCoreApplication.translate("MainWindow", u"Load Epoch", None))
        self.label_57.setText(QCoreApplication.translate("MainWindow", u"Duration", None))
        self.label_32.setText(QCoreApplication.translate("MainWindow", u"Epochs", None))
        self.label_31.setText(QCoreApplication.translate("MainWindow", u"Epochs Decay", None))
        self.label_60.setText(QCoreApplication.translate("MainWindow", u"Learning Rate", None))
        self.label_7.setText(QCoreApplication.translate("MainWindow", u"Initial", None))
        self.label_33.setText(QCoreApplication.translate("MainWindow", u"Target", None))
        self.label_59.setText(QCoreApplication.translate("MainWindow", u"Optimizer", None))
        self.label_14.setText(QCoreApplication.translate("MainWindow", u"Beta1", None))
        self.tabWidget_2.setTabText(self.tabWidget_2.indexOf(self._generator_tab), QCoreApplication.translate("MainWindow", u"Generator", None))
        self.separate_lr_schedules.setText(QCoreApplication.translate("MainWindow", u"Seperate Learning Schedules", None))
        self.label_61.setText(QCoreApplication.translate("MainWindow", u"Duration", None))
        self.label_34.setText(QCoreApplication.translate("MainWindow", u"Epochs", None))
        self.label_35.setText(QCoreApplication.translate("MainWindow", u"Epochs Decay", None))
        self.label_62.setText(QCoreApplication.translate("MainWindow", u"Learning Rate", None))
        self.label_12.setText(QCoreApplication.translate("MainWindow", u"Initial", None))
        self.label_36.setText(QCoreApplication.translate("MainWindow", u"Target", None))
        self.label_63.setText(QCoreApplication.translate("MainWindow", u"Optimizer", None))
        self.label_15.setText(QCoreApplication.translate("MainWindow", u"Beta1", None))
        self.tabWidget_2.setTabText(self.tabWidget_2.indexOf(self._discriminator_tab), QCoreApplication.translate("MainWindow", u"Discriminator", None))
        self.label_13.setText(QCoreApplication.translate("MainWindow", u"L1", None))
        self.label_17.setText(QCoreApplication.translate("MainWindow", u"Sobel", None))
        self.label_37.setText(QCoreApplication.translate("MainWindow", u"Laplacian", None))
        self.label_38.setText(QCoreApplication.translate("MainWindow", u"VGG", None))
        self.label_28.setText(QCoreApplication.translate("MainWindow", u"GAN", None))
        self.label_29.setText(QCoreApplication.translate("MainWindow", u"L2", None))
        self.log_losses.setText(QCoreApplication.translate("MainWindow", u"Log Losses During Training", None))
        self.label_18.setText(QCoreApplication.translate("MainWindow", u"Log Dump Frequency (epochs)", None))
        self.tabWidget_2.setTabText(self.tabWidget_2.indexOf(self._loss_tab), QCoreApplication.translate("MainWindow", u"Loss", None))
        self.save_model.setText(QCoreApplication.translate("MainWindow", u"Save Checkpoints", None))
        self.label_24.setText(QCoreApplication.translate("MainWindow", u"Frequency", None))
        self.label_42.setText(QCoreApplication.translate("MainWindow", u"Epochs", None))
        self.save_examples.setText(QCoreApplication.translate("MainWindow", u"Save Example Images", None))
        self.label_26.setText(QCoreApplication.translate("MainWindow", u"Frequency", None))
        self.label_43.setText(QCoreApplication.translate("MainWindow", u"Epochs", None))
        self.label_27.setText(QCoreApplication.translate("MainWindow", u"Count", None))
        self.tabWidget_2.setTabText(self.tabWidget_2.indexOf(self._saving_tab), QCoreApplication.translate("MainWindow", u"Saving", None))
        self.training_warning_label.setText(QCoreApplication.translate("MainWindow", u"_training_warning_label_", None))
        self.train_page_locked_label.setText(QCoreApplication.translate("MainWindow", u"_TRAIN_PAGE_LOCKED_", None))
        self.train_start.setText(QCoreApplication.translate("MainWindow", u"Begin Training", None))
        self.train_pause.setText(QCoreApplication.translate("MainWindow", u"Pause Training", None))
        self.train_stop.setText(QCoreApplication.translate("MainWindow", u"Stop Training", None))
        self.label_123.setText(QCoreApplication.translate("MainWindow", u"Experiment", None))
        self.test_experiment_path.setPlaceholderText(QCoreApplication.translate("MainWindow", u"Enter experiment directory path...", None))
        self.test_browse_experiment.setText(QCoreApplication.translate("MainWindow", u"Browse", None))
        self.label_98.setText(QCoreApplication.translate("MainWindow", u"Test Settings", None))
        self.label_58.setText(QCoreApplication.translate("MainWindow", u"Test Iterations", None))
        self.label_92.setText(QCoreApplication.translate("MainWindow", u"Load Epoch", None))
        self.test_get_most_recent.setText(QCoreApplication.translate("MainWindow", u"Most Recent", None))
        self.test_override_dataset.setText(QCoreApplication.translate("MainWindow", u"Override Dataset", None))
        self.test_dataset_path.setPlaceholderText(QCoreApplication.translate("MainWindow", u"Enter dataset directory path...", None))
        self.test_browse_dataset.setText(QCoreApplication.translate("MainWindow", u"Browse", None))
        self.test_warning_label.setText(QCoreApplication.translate("MainWindow", u"_test_warning_label_", None))
        self.test_start.setText(QCoreApplication.translate("MainWindow", u"Begin Test", None))
        self.test_progress_label.setText("")
        self.label_100.setText(QCoreApplication.translate("MainWindow", u"Review Experiment", None))
        self.review_experiment_path.setPlaceholderText(QCoreApplication.translate("MainWindow", u"Enter experiment directory path...", None))
        self.review_browse_experiment.setText(QCoreApplication.translate("MainWindow", u"Browse", None))
        self.review_load_experiment.setText(QCoreApplication.translate("MainWindow", u"Load Experiment", None))
        self.label_102.setText(QCoreApplication.translate("MainWindow", u"Train Config", None))
        self.label_106.setText(QCoreApplication.translate("MainWindow", u"ONNX Tools", None))
        self.convert_to_onnx.setText(QCoreApplication.translate("MainWindow", u"Convert to ONNX", None))
        self.label_120.setText(QCoreApplication.translate("MainWindow", u"Convert Model Settings", None))
        self.label_111.setText(QCoreApplication.translate("MainWindow", u"Experiment", None))
        self.convert_onnx_experiment.setPlaceholderText(QCoreApplication.translate("MainWindow", u"Enter path to experiment directory...", None))
        self.convert_onnx_browse_experiment.setText(QCoreApplication.translate("MainWindow", u"Browse", None))
        self.label_112.setText(QCoreApplication.translate("MainWindow", u"Load Epoch", None))
        self.convert_onnx_load_latest_epoch.setText(QCoreApplication.translate("MainWindow", u"Latest", None))
        self.label_118.setText(QCoreApplication.translate("MainWindow", u"In Channels", None))
        self.label_113.setText(QCoreApplication.translate("MainWindow", u"Width/Height", None))
        self.label_114.setText(QCoreApplication.translate("MainWindow", u"Target Device", None))
        self.label_115.setText(QCoreApplication.translate("MainWindow", u"Opset Version", None))
        self.convert_onnx_export_params.setText(QCoreApplication.translate("MainWindow", u"Export Params", None))
        self.convert_onnx_fold_constants.setText(QCoreApplication.translate("MainWindow", u"Fold Constants", None))
        self.convert_onnx_start.setText(QCoreApplication.translate("MainWindow", u"Convert Model", None))
        self.test_onnx_model.setText(QCoreApplication.translate("MainWindow", u"Test ONNX Model", None))
        self.label_119.setText(QCoreApplication.translate("MainWindow", u"Test ONNX Model Settings", None))
        self.label_116.setText(QCoreApplication.translate("MainWindow", u"Model", None))
        self.test_onnx_model_path.setPlaceholderText(QCoreApplication.translate("MainWindow", u"Enter path to ONNX model...", None))
        self.browse_onnx_model_path.setText(QCoreApplication.translate("MainWindow", u"Browse", None))
        self.label_117.setText(QCoreApplication.translate("MainWindow", u"Test Images", None))
        self.test_onnx_image_dir.setPlaceholderText(QCoreApplication.translate("MainWindow", u"Enter path to folder of test images...", None))
        self.browse_onnx_image_dir.setText(QCoreApplication.translate("MainWindow", u"Browse", None))
        self.test_onnx_start.setText(QCoreApplication.translate("MainWindow", u"Test Model", None))
        self.label_107.setText(QCoreApplication.translate("MainWindow", u"Image Tools", None))
        self.pair_images.setText(QCoreApplication.translate("MainWindow", u"Pair Images", None))
        self.label_89.setText(QCoreApplication.translate("MainWindow", u"Image Pairing Settings", None))
        self.label_64.setText(QCoreApplication.translate("MainWindow", u"Input A", None))
        self.pair_images_input_a.setPlaceholderText(QCoreApplication.translate("MainWindow", u"Enter path to folder of A images...", None))
        self.browse_pair_input_a.setText(QCoreApplication.translate("MainWindow", u"Browse", None))
        self.label_66.setText(QCoreApplication.translate("MainWindow", u"Input B", None))
        self.pair_images_input_b.setPlaceholderText(QCoreApplication.translate("MainWindow", u"Enter path to folder of B images...", None))
        self.browse_pair_input_b.setText(QCoreApplication.translate("MainWindow", u"Browse", None))
        self.label_67.setText(QCoreApplication.translate("MainWindow", u"Output", None))
        self.pair_images_output.setPlaceholderText(QCoreApplication.translate("MainWindow", u"Enter path to output directory...", None))
        self.browse_pair_output.setText(QCoreApplication.translate("MainWindow", u"Browse", None))
        self.label_77.setText(QCoreApplication.translate("MainWindow", u"Direction", None))
        self.pair_images_do_scaling.setText("")
        self.pair_images_scale.setCurrentText("")
        self.label_97.setText(QCoreApplication.translate("MainWindow", u"Scale Images", None))
        self.start_image_pairing.setText(QCoreApplication.translate("MainWindow", u"Start", None))
        self.preview_image_pairing.setText(QCoreApplication.translate("MainWindow", u"Preview", None))
        self.sort_images.setText(QCoreApplication.translate("MainWindow", u"Sort Images", None))
        self.remove_sorting_tags.setText(QCoreApplication.translate("MainWindow", u"Remove Sorting Tags", None))
        self.copy_image_sort.setText(QCoreApplication.translate("MainWindow", u"Copy Sort", None))
        self.label_90.setText(QCoreApplication.translate("MainWindow", u"Image Sorting Settings", None))
        self.label_76.setText(QCoreApplication.translate("MainWindow", u"Input", None))
        self.sort_images_input_dir.setPlaceholderText(QCoreApplication.translate("MainWindow", u"Enter path to input directory...", None))
        self.browse_sort_input_dir.setText(QCoreApplication.translate("MainWindow", u"Browse", None))
        self.label_78.setText(QCoreApplication.translate("MainWindow", u"Type", None))
        self.label_81.setText(QCoreApplication.translate("MainWindow", u"Direction", None))
        self.start_sort_images.setText(QCoreApplication.translate("MainWindow", u"Start", None))
        self.preview_image_sorting.setText(QCoreApplication.translate("MainWindow", u"Preview", None))
        self.label_91.setText(QCoreApplication.translate("MainWindow", u"Remove Tags Settings", None))
        self.remove_sort_tags_input.setPlaceholderText(QCoreApplication.translate("MainWindow", u"Enter path to input directory...", None))
        self.browse_remove_sort_tags_input.setText(QCoreApplication.translate("MainWindow", u"Browse", None))
        self.label_82.setText(QCoreApplication.translate("MainWindow", u"Input", None))
        self.start_remove_sort_tags.setText(QCoreApplication.translate("MainWindow", u"Start", None))
        self.preview_remove_sort_tags.setText(QCoreApplication.translate("MainWindow", u"Preview", None))
        self.label_96.setText(QCoreApplication.translate("MainWindow", u"Copy Sort Settings", None))
        self.label_83.setText(QCoreApplication.translate("MainWindow", u"From", None))
        self.copy_sort_copy_from.setPlaceholderText(QCoreApplication.translate("MainWindow", u"Enter path to folder of sorted images...", None))
        self.browse_copy_sort_copy_from.setText(QCoreApplication.translate("MainWindow", u"Browse", None))
        self.label_88.setText(QCoreApplication.translate("MainWindow", u"To", None))
        self.copy_sort_copy_to.setPlaceholderText(QCoreApplication.translate("MainWindow", u"Enter path to folder of images to sort...", None))
        self.browse_copy_sort_copy_to.setText(QCoreApplication.translate("MainWindow", u"Browse", None))
        self.start_copy_sort.setText(QCoreApplication.translate("MainWindow", u"Start", None))
        self.preview_copy_sort.setText(QCoreApplication.translate("MainWindow", u"Preview", None))
        self.split_dataset.setText(QCoreApplication.translate("MainWindow", u"Split Dataset Images", None))
        self.label_109.setText(QCoreApplication.translate("MainWindow", u"Input", None))
        self.split_dataset_input.setPlaceholderText(QCoreApplication.translate("MainWindow", u"Enter path to image directory...", None))
        self.browse_split_dataset_input.setText(QCoreApplication.translate("MainWindow", u"Browse", None))
        self.label_110.setText(QCoreApplication.translate("MainWindow", u"Output", None))
        self.split_dataset_output.setPlaceholderText(QCoreApplication.translate("MainWindow", u"Enter path to output directory...", None))
        self.browse_split_dataset_output.setText(QCoreApplication.translate("MainWindow", u"Browse", None))
        self.label_79.setText(QCoreApplication.translate("MainWindow", u"Test", None))
        self.split_dataset_test.setSuffix(QCoreApplication.translate("MainWindow", u"%", None))
        self.label_99.setText(QCoreApplication.translate("MainWindow", u"Train", None))
        self.label_108.setText(QCoreApplication.translate("MainWindow", u"Validate", None))
        self.split_dataset_train.setSuffix(QCoreApplication.translate("MainWindow", u"%", None))
        self.split_dataset_val.setSuffix(QCoreApplication.translate("MainWindow", u"%", None))
        self.start_split_dataset.setText(QCoreApplication.translate("MainWindow", u"Start", None))
        self.preview_split_dataset.setText(QCoreApplication.translate("MainWindow", u"Preview", None))
        self.utils_warning_label.setText(QCoreApplication.translate("MainWindow", u"_utils_warning_label_", None))
        self.image_utils_warning.setText(QCoreApplication.translate("MainWindow", u"Please carefully read the documentation for each of these tools, found at\n"
"(NectarGAN/docs/toolbox.md) before using them.", None))
        self.always_on_top.setText(QCoreApplication.translate("MainWindow", u"Always on Top", None))
        self.label_40.setText(QCoreApplication.translate("MainWindow", u"Review Graph Sample Rate", None))
        self.label_44.setText(QCoreApplication.translate("MainWindow", u"Training Update Rate (Iters)", None))
    # retranslateUi

