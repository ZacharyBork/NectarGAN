import pathlib
from os import PathLike
from typing import Any

from PySide6.QtCore import QThread, Slot, QVariantAnimation
from PySide6.QtWidgets import (
    QWidget, QLabel, QPushButton, QFrame, QLineEdit, QComboBox, QProgressBar)

from pix2pix_graphical.toolbox.workers.utility_worker import (
    UtilityWorker, SignalHandler)

class UtilityPanel():
    def __init__(self, mainwidget: QWidget) -> None:
        self.mainwidget = mainwidget
        self.find = self.mainwidget.findChild
        self.init_utilities_widgets()

        self.warning_label = self.find(QLabel, 'utils_warning_label')
        self.pairing_progress = self.find(
            QProgressBar, 'image_pairing_progress')
        self.pairing_label = self.find(QLabel, 'pairing_starting_label')
        self.pairing_label.setHidden(True)
        self._init_pairing_starting_anim()

    def init_utilities_widgets(self) -> None:
        pair_images_frame = self.find(QFrame, 'utils_pair_images_frame')
        pair_images_frame.setHidden(True)
        self.find(QPushButton, 'pair_images'
            ).clicked.connect(lambda x : pair_images_frame.setHidden(not x))
        self.find(QPushButton, 'start_image_pairing'
            ).clicked.connect(self.pair_images)
        self.find(QComboBox, 'pair_images_direction'
            ).addItems(['AtoB', 'BtoA'])
        self.find(QProgressBar, 'image_pairing_progress').setHidden(True)
        
    def _warn(self, warning: str) -> None:
        self.warning_label.setText(warning)

    def _init_pairing_starting_anim(self) -> None:
        self.pairing_starting_anim = QVariantAnimation()
        self.pairing_starting_anim.setStartValue(0)
        self.pairing_starting_anim.setEndValue(1000)
        self.pairing_starting_anim.setDuration(500000)
        self.pairing_starting_anim.valueChanged.connect(
            self._pairing_stating_anim)

    ### ANIMATION ###

    def _pairing_stating_anim(self, val: int):
        text = 'Starting' + '.' * (val%4) 
        self.pairing_label.setText(text)

    ### WORKER ###

    def _start_worker(self, args: Any) -> None:
        self.utils_thread = QThread() 
        self.signalhandler = SignalHandler()
        self.worker = UtilityWorker(args=args, signalhandler=self.signalhandler)
        self.worker.moveToThread(self.utils_thread)

        self.signalhandler.progress.connect(self._pairing_progress)
        self.signalhandler.finished.connect(self.utils_thread.quit)
        self.signalhandler.finished.connect(self._cleanup)
                
        self.utils_thread.started.connect(self.worker.pair_images)
        self.utils_thread.finished.connect(self.utils_thread.deleteLater)
        self.utils_thread.start()

    def _cleanup(self) -> None:
        self.pairing_progress.setHidden(True)
        self.find(QFrame, 'utils_pair_images_frame').setEnabled(True)
        self.find(QPushButton, 'pair_images').setEnabled(True)

    ### ONNX UTILITIES ###

    def _pth_to_onnx(self, input_file: PathLike) -> None:
        '''Converts a .pth.tar model checkpoint to a .onnx file.'''
        pass

    def convert_to_onnx(self) -> None:
        '''Converts pre-trained model weights to ONNX format.'''
        pass

    def _run_onnx_inference(self) -> None:
        '''Runs ONNX model inference.'''
        pass

    def test_onnx_model(self) -> None:
        '''Tests the inference of an ONNX model.'''
        pass

    ### IMAGE PAIRING ###        

    @Slot(float)
    def _pairing_progress(self, progress: float) -> None:
        if not self.pairing_label.isHidden():
            self.pairing_starting_anim.stop()
            self.pairing_label.setHidden(True)
            self.pairing_progress.setHidden(False)
        self.pairing_progress.setValue(progress)

    def pair_images(self) -> None:
        '''Takes images from A|B folder and pairs them for pix2pix training.'''
        input_dirA = self.find(QLineEdit, 'pair_images_input_a').text()
        dirA = pathlib.Path(input_dirA)
        if not dirA.exists() or input_dirA == '': 
            self._warn('Please input an A directory for pairing.')
            return
        input_dirB = self.find(QLineEdit, 'pair_images_input_b').text()
        dirB = pathlib.Path(input_dirB)
        if not dirB.exists() or input_dirB == '': 
            self._warn('Please input a B directory for pairing.')
            return
        input_outdir = self.find(QLineEdit, 'pair_images_output').text()
        outdir = pathlib.Path(input_outdir)
        if not outdir.exists() or input_outdir == '': 
            self._warn('Please input an output directory for pairing.')
            return
        
        direction = self.find(QComboBox, 'pair_images_direction').currentText()
        args = []
        for fileA in list(dirA.iterdir()):
            args.append((fileA, dirB, outdir, direction)) 

        self.pairing_label.setHidden(False)
        self.pairing_starting_anim.start()
        self.pairing_progress.setValue(0)
        self.find(QFrame, 'utils_pair_images_frame').setEnabled(False)
        self.find(QPushButton, 'pair_images').setEnabled(False)
        self._start_worker(args)       

    ### IMAGE SORTING ###
    
    def sort_images(self) -> None:
        '''Sorts dataset images by various metrics.'''
        pass

    def strip_sorting_tag(self) -> None:
        '''Removes sorting tags added by `UtilityPanel.sort_images()`.'''
        pass

    ### DATASET SPLIT ###

    def split_dataset_images(self) -> None:
        '''Splits a folder of dataset images into [train, test, val].'''
        pass

