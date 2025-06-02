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
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDoubleSpinBox,
    QFormLayout, QFrame, QGroupBox, QHBoxLayout,
    QLabel, QLineEdit, QMainWindow, QMenu,
    QMenuBar, QProgressBar, QPushButton, QSizePolicy,
    QSlider, QSpacerItem, QSpinBox, QStatusBar,
    QVBoxLayout, QWidget)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(800, 793)
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
        self.verticalLayout_2 = QVBoxLayout(self.frame)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.groupBox = QGroupBox(self.frame)
        self.groupBox.setObjectName(u"groupBox")
        self.verticalLayout_3 = QVBoxLayout(self.groupBox)
        self.verticalLayout_3.setObjectName(u"verticalLayout_3")
        self.verticalLayout_3.setContentsMargins(3, 3, 3, 3)
        self.frame_2 = QFrame(self.groupBox)
        self.frame_2.setObjectName(u"frame_2")
        self.frame_2.setFrameShape(QFrame.StyledPanel)
        self.frame_2.setFrameShadow(QFrame.Raised)
        self.formLayout = QFormLayout(self.frame_2)
        self.formLayout.setObjectName(u"formLayout")
        self.formLayout.setContentsMargins(0, 0, -1, 0)
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


        self.verticalLayout_3.addWidget(self.frame_2)

        self.init_from_last_train = QPushButton(self.groupBox)
        self.init_from_last_train.setObjectName(u"init_from_last_train")

        self.verticalLayout_3.addWidget(self.init_from_last_train)


        self.verticalLayout_2.addWidget(self.groupBox)


        self.verticalLayout.addWidget(self.frame)

        self.groupBox_3 = QGroupBox(self.centralwidget)
        self.groupBox_3.setObjectName(u"groupBox_3")
        self.verticalLayout_4 = QVBoxLayout(self.groupBox_3)
        self.verticalLayout_4.setObjectName(u"verticalLayout_4")
        self.verticalLayout_4.setContentsMargins(3, 3, 3, 3)
        self.frame_3 = QFrame(self.groupBox_3)
        self.frame_3.setObjectName(u"frame_3")
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
        self.dataset_root = QLineEdit(self.frame_3)
        self.dataset_root.setObjectName(u"dataset_root")

        self.horizontalLayout_2.addWidget(self.dataset_root)

        self.browse_dataset = QPushButton(self.frame_3)
        self.browse_dataset.setObjectName(u"browse_dataset")

        self.horizontalLayout_2.addWidget(self.browse_dataset)


        self.formLayout_3.setLayout(0, QFormLayout.ItemRole.FieldRole, self.horizontalLayout_2)


        self.verticalLayout_4.addWidget(self.frame_3)

        self.frame_4 = QFrame(self.groupBox_3)
        self.frame_4.setObjectName(u"frame_4")
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
        self.load_size.setMinimum(64)
        self.load_size.setMaximum(2048)
        self.load_size.setSingleStep(64)
        self.load_size.setValue(512)

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
        self.crop_size.setMinimum(64)
        self.crop_size.setMaximum(1024)
        self.crop_size.setSingleStep(64)
        self.crop_size.setValue(256)

        self.formLayout_5.setWidget(0, QFormLayout.ItemRole.FieldRole, self.crop_size)


        self.horizontalLayout_9.addWidget(self.frame_6)


        self.verticalLayout_4.addWidget(self.frame_4)


        self.verticalLayout.addWidget(self.groupBox_3)

        self.groupBox_2 = QGroupBox(self.centralwidget)
        self.groupBox_2.setObjectName(u"groupBox_2")
        self.formLayout_2 = QFormLayout(self.groupBox_2)
        self.formLayout_2.setObjectName(u"formLayout_2")
        self.formLayout_2.setContentsMargins(3, 3, 3, 3)
        self.label_5 = QLabel(self.groupBox_2)
        self.label_5.setObjectName(u"label_5")

        self.formLayout_2.setWidget(0, QFormLayout.ItemRole.LabelRole, self.label_5)

        self.direction = QComboBox(self.groupBox_2)
        self.direction.setObjectName(u"direction")

        self.formLayout_2.setWidget(0, QFormLayout.ItemRole.FieldRole, self.direction)

        self.label_7 = QLabel(self.groupBox_2)
        self.label_7.setObjectName(u"label_7")

        self.formLayout_2.setWidget(1, QFormLayout.ItemRole.LabelRole, self.label_7)

        self.horizontalLayout_6 = QHBoxLayout()
        self.horizontalLayout_6.setObjectName(u"horizontalLayout_6")
        self.lr_slider = QSlider(self.groupBox_2)
        self.lr_slider.setObjectName(u"lr_slider")
        self.lr_slider.setMaximum(100)
        self.lr_slider.setValue(2)
        self.lr_slider.setOrientation(Qt.Horizontal)

        self.horizontalLayout_6.addWidget(self.lr_slider)

        self.learning_rate = QDoubleSpinBox(self.groupBox_2)
        self.learning_rate.setObjectName(u"learning_rate")
        self.learning_rate.setDecimals(5)
        self.learning_rate.setMinimum(0.000000000000000)
        self.learning_rate.setMaximum(0.100000000000000)
        self.learning_rate.setSingleStep(0.000100000000000)
        self.learning_rate.setValue(0.000200000000000)

        self.horizontalLayout_6.addWidget(self.learning_rate)


        self.formLayout_2.setLayout(1, QFormLayout.ItemRole.FieldRole, self.horizontalLayout_6)

        self.label_6 = QLabel(self.groupBox_2)
        self.label_6.setObjectName(u"label_6")

        self.formLayout_2.setWidget(2, QFormLayout.ItemRole.LabelRole, self.label_6)

        self.horizontalLayout_5 = QHBoxLayout()
        self.horizontalLayout_5.setObjectName(u"horizontalLayout_5")
        self.num_epochs_slider = QSlider(self.groupBox_2)
        self.num_epochs_slider.setObjectName(u"num_epochs_slider")
        self.num_epochs_slider.setMinimum(1)
        self.num_epochs_slider.setMaximum(500)
        self.num_epochs_slider.setValue(100)
        self.num_epochs_slider.setOrientation(Qt.Horizontal)

        self.horizontalLayout_5.addWidget(self.num_epochs_slider)

        self.num_epochs = QSpinBox(self.groupBox_2)
        self.num_epochs.setObjectName(u"num_epochs")
        self.num_epochs.setMinimum(1)
        self.num_epochs.setMaximum(500)
        self.num_epochs.setSingleStep(5)
        self.num_epochs.setValue(100)

        self.horizontalLayout_5.addWidget(self.num_epochs)


        self.formLayout_2.setLayout(2, QFormLayout.ItemRole.FieldRole, self.horizontalLayout_5)

        self.label_8 = QLabel(self.groupBox_2)
        self.label_8.setObjectName(u"label_8")

        self.formLayout_2.setWidget(3, QFormLayout.ItemRole.LabelRole, self.label_8)

        self.horizontalLayout_7 = QHBoxLayout()
        self.horizontalLayout_7.setObjectName(u"horizontalLayout_7")
        self.batch_size_slider = QSlider(self.groupBox_2)
        self.batch_size_slider.setObjectName(u"batch_size_slider")
        self.batch_size_slider.setMinimum(1)
        self.batch_size_slider.setMaximum(32)
        self.batch_size_slider.setValue(1)
        self.batch_size_slider.setOrientation(Qt.Horizontal)

        self.horizontalLayout_7.addWidget(self.batch_size_slider)

        self.batch_size = QSpinBox(self.groupBox_2)
        self.batch_size.setObjectName(u"batch_size")
        self.batch_size.setMinimum(1)
        self.batch_size.setMaximum(32)

        self.horizontalLayout_7.addWidget(self.batch_size)


        self.formLayout_2.setLayout(3, QFormLayout.ItemRole.FieldRole, self.horizontalLayout_7)

        self.label_9 = QLabel(self.groupBox_2)
        self.label_9.setObjectName(u"label_9")

        self.formLayout_2.setWidget(4, QFormLayout.ItemRole.LabelRole, self.label_9)

        self.horizontalLayout_8 = QHBoxLayout()
        self.horizontalLayout_8.setObjectName(u"horizontalLayout_8")
        self.worker_count_slider = QSlider(self.groupBox_2)
        self.worker_count_slider.setObjectName(u"worker_count_slider")
        self.worker_count_slider.setMinimum(1)
        self.worker_count_slider.setMaximum(32)
        self.worker_count_slider.setValue(4)
        self.worker_count_slider.setOrientation(Qt.Horizontal)

        self.horizontalLayout_8.addWidget(self.worker_count_slider)

        self.worker_count = QSpinBox(self.groupBox_2)
        self.worker_count.setObjectName(u"worker_count")
        self.worker_count.setMinimum(1)
        self.worker_count.setMaximum(32)

        self.horizontalLayout_8.addWidget(self.worker_count)


        self.formLayout_2.setLayout(4, QFormLayout.ItemRole.FieldRole, self.horizontalLayout_8)

        self.label_14 = QLabel(self.groupBox_2)
        self.label_14.setObjectName(u"label_14")

        self.formLayout_2.setWidget(5, QFormLayout.ItemRole.LabelRole, self.label_14)

        self.horizontalLayout_14 = QHBoxLayout()
        self.horizontalLayout_14.setObjectName(u"horizontalLayout_14")
        self.beta1_slider = QSlider(self.groupBox_2)
        self.beta1_slider.setObjectName(u"beta1_slider")
        self.beta1_slider.setMaximum(250)
        self.beta1_slider.setOrientation(Qt.Horizontal)

        self.horizontalLayout_14.addWidget(self.beta1_slider)

        self.beta1 = QDoubleSpinBox(self.groupBox_2)
        self.beta1.setObjectName(u"beta1")
        self.beta1.setMaximum(2.500000000000000)
        self.beta1.setSingleStep(0.100000000000000)
        self.beta1.setValue(0.500000000000000)

        self.horizontalLayout_14.addWidget(self.beta1)


        self.formLayout_2.setLayout(5, QFormLayout.ItemRole.FieldRole, self.horizontalLayout_14)


        self.verticalLayout.addWidget(self.groupBox_2)

        self.groupBox_4 = QGroupBox(self.centralwidget)
        self.groupBox_4.setObjectName(u"groupBox_4")
        self.formLayout_6 = QFormLayout(self.groupBox_4)
        self.formLayout_6.setObjectName(u"formLayout_6")
        self.formLayout_6.setContentsMargins(3, 3, 3, 3)
        self.label_12 = QLabel(self.groupBox_4)
        self.label_12.setObjectName(u"label_12")

        self.formLayout_6.setWidget(0, QFormLayout.ItemRole.LabelRole, self.label_12)

        self.horizontalLayout_10 = QHBoxLayout()
        self.horizontalLayout_10.setObjectName(u"horizontalLayout_10")
        self.l1_slider = QSlider(self.groupBox_4)
        self.l1_slider.setObjectName(u"l1_slider")
        self.l1_slider.setMaximum(500)
        self.l1_slider.setOrientation(Qt.Horizontal)

        self.horizontalLayout_10.addWidget(self.l1_slider)

        self.l1 = QDoubleSpinBox(self.groupBox_4)
        self.l1.setObjectName(u"l1")
        self.l1.setMaximum(500.000000000000000)
        self.l1.setSingleStep(5.000000000000000)
        self.l1.setValue(100.000000000000000)

        self.horizontalLayout_10.addWidget(self.l1)


        self.formLayout_6.setLayout(0, QFormLayout.ItemRole.FieldRole, self.horizontalLayout_10)

        self.label_13 = QLabel(self.groupBox_4)
        self.label_13.setObjectName(u"label_13")

        self.formLayout_6.setWidget(1, QFormLayout.ItemRole.LabelRole, self.label_13)

        self.horizontalLayout_11 = QHBoxLayout()
        self.horizontalLayout_11.setObjectName(u"horizontalLayout_11")
        self.sobel_slider = QSlider(self.groupBox_4)
        self.sobel_slider.setObjectName(u"sobel_slider")
        self.sobel_slider.setMaximum(100)
        self.sobel_slider.setOrientation(Qt.Horizontal)

        self.horizontalLayout_11.addWidget(self.sobel_slider)

        self.sobel = QDoubleSpinBox(self.groupBox_4)
        self.sobel.setObjectName(u"sobel")
        self.sobel.setMaximum(100.000000000000000)
        self.sobel.setValue(10.000000000000000)

        self.horizontalLayout_11.addWidget(self.sobel)


        self.formLayout_6.setLayout(1, QFormLayout.ItemRole.FieldRole, self.horizontalLayout_11)


        self.verticalLayout.addWidget(self.groupBox_4)

        self.groupBox_5 = QGroupBox(self.centralwidget)
        self.groupBox_5.setObjectName(u"groupBox_5")
        self.horizontalLayout_16 = QHBoxLayout(self.groupBox_5)
        self.horizontalLayout_16.setObjectName(u"horizontalLayout_16")
        self.horizontalLayout_16.setContentsMargins(3, 3, 3, 3)
        self.label_15 = QLabel(self.groupBox_5)
        self.label_15.setObjectName(u"label_15")
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_15.sizePolicy().hasHeightForWidth())
        self.label_15.setSizePolicy(sizePolicy)

        self.horizontalLayout_16.addWidget(self.label_15)

        self.continue_train = QCheckBox(self.groupBox_5)
        self.continue_train.setObjectName(u"continue_train")
        sizePolicy1 = QSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.continue_train.sizePolicy().hasHeightForWidth())
        self.continue_train.setSizePolicy(sizePolicy1)

        self.horizontalLayout_16.addWidget(self.continue_train)

        self.label_16 = QLabel(self.groupBox_5)
        self.label_16.setObjectName(u"label_16")
        sizePolicy.setHeightForWidth(self.label_16.sizePolicy().hasHeightForWidth())
        self.label_16.setSizePolicy(sizePolicy)

        self.horizontalLayout_16.addWidget(self.label_16)

        self.load_epoch = QSpinBox(self.groupBox_5)
        self.load_epoch.setObjectName(u"load_epoch")
        self.load_epoch.setMinimum(1)

        self.horizontalLayout_16.addWidget(self.load_epoch)

        self.horizontalSpacer = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_16.addItem(self.horizontalSpacer)


        self.verticalLayout.addWidget(self.groupBox_5)

        self.groupBox_6 = QGroupBox(self.centralwidget)
        self.groupBox_6.setObjectName(u"groupBox_6")
        self.horizontalLayout_17 = QHBoxLayout(self.groupBox_6)
        self.horizontalLayout_17.setObjectName(u"horizontalLayout_17")
        self.label_17 = QLabel(self.groupBox_6)
        self.label_17.setObjectName(u"label_17")
        sizePolicy.setHeightForWidth(self.label_17.sizePolicy().hasHeightForWidth())
        self.label_17.setSizePolicy(sizePolicy)

        self.horizontalLayout_17.addWidget(self.label_17)

        self.save_checkpoints = QCheckBox(self.groupBox_6)
        self.save_checkpoints.setObjectName(u"save_checkpoints")
        sizePolicy1.setHeightForWidth(self.save_checkpoints.sizePolicy().hasHeightForWidth())
        self.save_checkpoints.setSizePolicy(sizePolicy1)
        self.save_checkpoints.setChecked(True)

        self.horizontalLayout_17.addWidget(self.save_checkpoints)

        self.label_18 = QLabel(self.groupBox_6)
        self.label_18.setObjectName(u"label_18")
        sizePolicy.setHeightForWidth(self.label_18.sizePolicy().hasHeightForWidth())
        self.label_18.setSizePolicy(sizePolicy)

        self.horizontalLayout_17.addWidget(self.label_18)

        self.save_frequency = QSpinBox(self.groupBox_6)
        self.save_frequency.setObjectName(u"save_frequency")
        sizePolicy1.setHeightForWidth(self.save_frequency.sizePolicy().hasHeightForWidth())
        self.save_frequency.setSizePolicy(sizePolicy1)
        self.save_frequency.setMinimum(1)
        self.save_frequency.setValue(5)

        self.horizontalLayout_17.addWidget(self.save_frequency)

        self.horizontalSpacer_2 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_17.addItem(self.horizontalSpacer_2)


        self.verticalLayout.addWidget(self.groupBox_6)

        self.frame_7 = QFrame(self.centralwidget)
        self.frame_7.setObjectName(u"frame_7")
        self.frame_7.setFrameShape(QFrame.StyledPanel)
        self.frame_7.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_18 = QHBoxLayout(self.frame_7)
        self.horizontalLayout_18.setObjectName(u"horizontalLayout_18")
        self.test_button = QPushButton(self.frame_7)
        self.test_button.setObjectName(u"test_button")

        self.horizontalLayout_18.addWidget(self.test_button)

        self.begin_training = QPushButton(self.frame_7)
        self.begin_training.setObjectName(u"begin_training")

        self.horizontalLayout_18.addWidget(self.begin_training)


        self.verticalLayout.addWidget(self.frame_7)

        self.frame_8 = QFrame(self.centralwidget)
        self.frame_8.setObjectName(u"frame_8")
        self.frame_8.setFrameShape(QFrame.StyledPanel)
        self.frame_8.setFrameShadow(QFrame.Raised)
        self.verticalLayout_5 = QVBoxLayout(self.frame_8)
        self.verticalLayout_5.setObjectName(u"verticalLayout_5")
        self.train_start = QPushButton(self.frame_8)
        self.train_start.setObjectName(u"train_start")

        self.verticalLayout_5.addWidget(self.train_start)

        self.train_stop = QPushButton(self.frame_8)
        self.train_stop.setObjectName(u"train_stop")

        self.verticalLayout_5.addWidget(self.train_stop)

        self.train_progress = QProgressBar(self.frame_8)
        self.train_progress.setObjectName(u"train_progress")
        self.train_progress.setValue(24)

        self.verticalLayout_5.addWidget(self.train_progress)


        self.verticalLayout.addWidget(self.frame_8)

        self.verticalSpacer = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.verticalLayout.addItem(self.verticalSpacer)

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 800, 22))
        self.menuFile = QMenu(self.menubar)
        self.menuFile.setObjectName(u"menuFile")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.menubar.addAction(self.menuFile.menuAction())
        self.menuFile.addAction(self.actionLoad_From_Config)

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"MainWindow", None))
        self.actionLoad_From_Config.setText(QCoreApplication.translate("MainWindow", u"Load From Config", None))
        self.groupBox.setTitle(QCoreApplication.translate("MainWindow", u"Output", None))
        self.label_2.setText(QCoreApplication.translate("MainWindow", u"Output Directory", None))
        self.browse_output.setText(QCoreApplication.translate("MainWindow", u"Browse", None))
        self.label_3.setText(QCoreApplication.translate("MainWindow", u"Experiment Name", None))
        self.label_4.setText(QCoreApplication.translate("MainWindow", u"Version", None))
        self.init_from_last_train.setText(QCoreApplication.translate("MainWindow", u"Initialize settings from most recent train", None))
        self.groupBox_3.setTitle(QCoreApplication.translate("MainWindow", u"Dataset", None))
        self.label.setText(QCoreApplication.translate("MainWindow", u"Dataset Root", None))
        self.browse_dataset.setText(QCoreApplication.translate("MainWindow", u"Browse", None))
        self.label_10.setText(QCoreApplication.translate("MainWindow", u"Load Size", None))
        self.label_11.setText(QCoreApplication.translate("MainWindow", u"Crop Size", None))
        self.groupBox_2.setTitle(QCoreApplication.translate("MainWindow", u"Training", None))
        self.label_5.setText(QCoreApplication.translate("MainWindow", u"Direction", None))
        self.label_7.setText(QCoreApplication.translate("MainWindow", u"Learning Rate", None))
        self.label_6.setText(QCoreApplication.translate("MainWindow", u"Number of Epochs", None))
        self.label_8.setText(QCoreApplication.translate("MainWindow", u"Batch Size", None))
        self.label_9.setText(QCoreApplication.translate("MainWindow", u"Worker Count", None))
        self.label_14.setText(QCoreApplication.translate("MainWindow", u"Beta1", None))
        self.groupBox_4.setTitle(QCoreApplication.translate("MainWindow", u"Loss Scaling", None))
        self.label_12.setText(QCoreApplication.translate("MainWindow", u"L1", None))
        self.label_13.setText(QCoreApplication.translate("MainWindow", u"Sobel", None))
        self.groupBox_5.setTitle(QCoreApplication.translate("MainWindow", u"Continue Train", None))
        self.label_15.setText(QCoreApplication.translate("MainWindow", u"Continue From Checkpoint", None))
        self.continue_train.setText("")
        self.label_16.setText(QCoreApplication.translate("MainWindow", u"Load Epoch", None))
        self.groupBox_6.setTitle(QCoreApplication.translate("MainWindow", u"Save Model", None))
        self.label_17.setText(QCoreApplication.translate("MainWindow", u"Save Intermediate Checkpoints", None))
        self.save_checkpoints.setText("")
        self.label_18.setText(QCoreApplication.translate("MainWindow", u"Frequency (epochs)", None))
        self.test_button.setText(QCoreApplication.translate("MainWindow", u"Test", None))
        self.begin_training.setText(QCoreApplication.translate("MainWindow", u"Begin Training", None))
        self.train_start.setText(QCoreApplication.translate("MainWindow", u"Start", None))
        self.train_stop.setText(QCoreApplication.translate("MainWindow", u"Stop", None))
        self.menuFile.setTitle(QCoreApplication.translate("MainWindow", u"File", None))
    # retranslateUi

