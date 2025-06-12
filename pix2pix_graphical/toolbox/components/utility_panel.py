import pathlib
import shutil
import random
from os import PathLike
from typing import Any, Literal

from PySide6.QtCore import QThread, QVariantAnimation
from PySide6.QtWidgets import (
    QWidget, QLabel, QPushButton, QFrame, QLineEdit, QComboBox, QProgressBar,
    QCheckBox, QHBoxLayout, QTextEdit, QSpinBox)

from pix2pix_graphical.toolbox.workers.utility_worker import (
    UtilityWorker, SignalHandler)

class UtilityPanel():
    def __init__(self, mainwidget: QWidget) -> None:
        self.mainwidget = mainwidget
        self.find = self.mainwidget.findChild
        self._init_utilities_widgets()
        self._init_progress_bars()

        self.warning_label = self.find(QLabel, 'utils_warning_label')
        self.change_log = self.find(QTextEdit, 'file_change_log')

    def _init_utilities_widgets(self) -> None:
        pair_images_frame = self.find(QFrame, 'utils_pair_images_frame')
        pair_images_frame.setHidden(True)
        self.find(QPushButton, 'pair_images'
            ).clicked.connect(lambda x : pair_images_frame.setHidden(not x))
        self.find(QPushButton, 'start_image_pairing'
            ).clicked.connect(self.pair_images)
        self.find(QPushButton, 'preview_image_pairing'
            ).clicked.connect(lambda : self.pair_images(dry_run=True))
        self.find(QComboBox, 'pair_images_direction'
            ).addItems(['AtoB', 'BtoA'])
        self.pair_do_scaling = self.find(QCheckBox, 'pair_images_do_scaling')
        self.pair_images_scale = self.find(QComboBox, 'pair_images_scale')
        self.pair_images_scale.addItems([
            '16²', '32²', '64²', '128²', '256²', '512²', '1024²'])    
        self.pair_images_scale.setEnabled(False)
        self.pair_images_scale.setCurrentIndex(4)
        self.pair_do_scaling.clicked.connect(
            lambda x : self.pair_images_scale.setEnabled(x))    

        self.find(QComboBox, 'sort_images_type').addItems([
            'Count (White Pixels)', 'Count (Black Pixels)', 
            'Mean Pixel Value', 'Contrast (Average Local)', 
            'Contrast (RMS)', 'Contrast (Haziness)'])
        self.find(QComboBox, 'sort_images_direction').addItems([
            'Ascending', 'Descending'])

        sort_images_frame = self.find(QFrame, 'utils_sort_images_frame')
        sort_images_frame.setHidden(True)
        self.find(QPushButton, 'sort_images'
            ).clicked.connect(lambda x : sort_images_frame.setHidden(not x))
        self.find(QPushButton, 'start_sort_images'
            ).clicked.connect(self.sort_images)
        self.find(QPushButton, 'preview_image_sorting'
            ).clicked.connect(lambda : self.sort_images(dry_run=True))
        
        remove_sorting_tags_frame = self.find(
            QFrame, 'utils_remove_sorting_tags_frame')
        remove_sorting_tags_frame.setHidden(True)
        self.find(QPushButton, 'remove_sorting_tags').clicked.connect(
            lambda x : remove_sorting_tags_frame.setHidden(not x))
        self.find(QPushButton, 'start_remove_sort_tags'
            ).clicked.connect(self.strip_sorting_tags)
        self.find(QPushButton, 'preview_remove_sort_tags'
            ).clicked.connect(lambda : self.strip_sorting_tags(dry_run=True))
        
        copy_sort_frame = self.find(QFrame, 'utils_copy_sort_frame')
        copy_sort_frame.setHidden(True)
        self.find(QPushButton, 'copy_image_sort'
            ).clicked.connect(lambda x : copy_sort_frame.setHidden(not x))
        
        split_dataset_frame = self.find(QFrame, 'utils_split_dataset_frame')
        split_dataset_frame.setHidden(True)
        self.find(QPushButton, 'split_dataset'
            ).clicked.connect(lambda x : split_dataset_frame.setHidden(not x))
        self.find(QPushButton, 'start_split_dataset'
            ).clicked.connect(self.split_dataset)
        self.find(QPushButton, 'preview_split_dataset'
            ).clicked.connect(lambda : self.split_dataset(dry_run=True))

    def _init_progress_bars(self) -> None:
        self.pairing_progress = self.find(
            QProgressBar, 'image_pairing_progress')
        self.pairing_label = self.find(QLabel, 'pairing_starting_label')
        self.pairing_progress.setHidden(True)
        self.pairing_label.setHidden(True)
        

        self.sort_images_progress = self.find(
            QProgressBar, 'sort_images_progress')
        self.sort_images_progress.setHidden(True)
        self.sorting_label = self.find(QLabel, 'sorting_starting_label')
        self.sorting_label.setHidden(True)
        
        self.remove_tags_progress = self.find(
            QProgressBar, 'remove_sort_tags_progress')
        self.remove_tags_progress.setHidden(True)
        self.remove_sort_tags_label = self.find(
            QLabel, 'remove_sort_tags_starting_label')
        self.remove_sort_tags_label.setHidden(True)
        
        self.copy_sort_progress = self.find(
            QProgressBar, 'copy_sort_progress')
        self.copy_sort_progress.setHidden(True)
        self.copy_sort_label = self.find(QLabel, 'copy_sort_starting_label')
        self.copy_sort_label.setHidden(True)
        
        self.split_dataset_progress = self.find(
            QProgressBar, 'split_dataset_progress')
        self.split_dataset_progress.setHidden(True)
        self.split_dataset_label = self.find(QLabel, 'split_dataset_starting_label')
        self.split_dataset_label.setHidden(True)

        self._init_starting_animation()
        
    def _warn(self, warning: str) -> None:
        self.warning_label.setText(warning)

    def _init_starting_animation(self) -> None:
        self.starting_anim = QVariantAnimation()
        self.starting_anim.setStartValue(0)
        self.starting_anim.setEndValue(1000)
        self.starting_anim.setDuration(500000)
        self.starting_anim.valueChanged.connect(
            self._starting_animation)

    ### ANIMATION ###

    def _starting_animation(self, val: int):
        text = 'Starting' + '.' * (val%4) 
        self.pairing_label.setText(text)
        self.sorting_label.setText(text)
        self.remove_sort_tags_label.setText(text)
        self.copy_sort_label.setText(text)

    ### WORKER ###

    def _start_worker(
            self, 
            args: Any, 
            task: Literal['pair', 'sort']
        ) -> None:
        self.utils_thread = QThread() 
        self.signalhandler = SignalHandler()
        self.worker = UtilityWorker(
            args=args, signalhandler=self.signalhandler)
        self.worker.moveToThread(self.utils_thread)

        match task:
            case 'pair': 
                self.signalhandler.progress.connect(self._pairing_progress)
                self.signalhandler.finished.connect(self._pairing_cleanup)
                self.utils_thread.started.connect(self.worker.pair_images)
            case 'sort': 
                self.signalhandler.progress.connect(self._sorting_progress)
                self.signalhandler.finished.connect(self._sorting_cleanup)
                self.utils_thread.started.connect(self.worker.sort_images)
        
        self.signalhandler.finished.connect(self.utils_thread.quit)
        self.utils_thread.finished.connect(self.utils_thread.deleteLater)
        self.utils_thread.start()

    def _pairing_cleanup(self) -> None:
        self.pairing_progress.setHidden(True)
        self.find(QFrame, 'utils_pair_images_frame').setEnabled(True)
        self.find(QPushButton, 'pair_images').setEnabled(True)

    def _sorting_cleanup(self) -> None:
        self.sort_images_progress.setHidden(True)
        self.find(QFrame, 'utils_sort_images_frame').setEnabled(True)
        self.find(QPushButton, 'sort_images').setEnabled(True)

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

    def _pairing_progress(self, progress: float) -> None:
        if not self._sorting_progress.isHidden():
            self.starting_anim.stop()
            self.pairing_label.setHidden(True)
            self.pairing_progress.setHidden(False)
        self.pairing_progress.setValue(progress)

    def pair_images(self, dry_run: bool=False) -> None:
        '''Takes images from A|B folder and pairs them for pix2pix training.'''
        self.change_log.clear()

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
        if self.pair_do_scaling.isChecked():
            scale = int(self.pair_images_scale.currentText()[0:-1])
        else: scale = -1
        
        args = []
        for fileA in list(dirA.iterdir()):
            args.append((fileA, dirB, outdir, direction, scale)) 

        if dry_run:
            for arg in args:
                name = arg[0]
                log_line = (
                    f'({pathlib.Path(dirA, name)} + '
                    f'{pathlib.Path(dirB, name)}) -> '
                    f'{pathlib.Path(outdir, name)}')
                self.change_log.append(log_line)
        else:
            self.pairing_label.setHidden(False)
            self.starting_anim.start()
            self.pairing_progress.setValue(0)
            self.find(QFrame, 'utils_pair_images_frame').setEnabled(False)
            self.find(QPushButton, 'pair_images').setEnabled(False)
            self._start_worker(args, task='pair')       

    ### IMAGE SORTING ###

    def _sorting_progress(self, progress: int) -> None:
        if not self.sorting_label.isHidden():
            self.starting_anim.stop()
            self.sorting_label.setHidden(True)
            self.sort_images_progress.setHidden(False)
        self.sort_images_progress.setValue(progress)
    
    def sort_images(self, dry_run: bool=False) -> None:
        '''Sorts dataset images by various metrics.'''
        self.change_log.clear()

        input_dir = self.find(QLineEdit, 'sort_images_input_dir').text()
        dir = pathlib.Path(input_dir)
        if not dir.exists() or input_dir == '': 
            self._warn('Please input a directory with images to sort.')
            return
        files = list(dir.iterdir())
        if len(files) == 0:
            self._warn('No files found.')
            return
        
        sort_type = self.find(QComboBox, 'sort_images_type').currentText()
        args = [(i, sort_type) for i in files]

        if dry_run:
            for file in files:
                self.change_log.append(file.as_posix())
        else:
            self.sorting_label.setHidden(False)
            self.starting_anim.start()
            self.sort_images_progress.setValue(0)
            self._start_worker(args, task='sort')  
        
    def strip_sorting_tags(self, dry_run: bool=False) -> None:
        '''Removes sorting tags added by `UtilityPanel.sort_images()`.'''
        self.change_log.clear()

        input_dir = self.find(QLineEdit, 'remove_sort_tags_input').text()
        dir = pathlib.Path(input_dir)
        if not dir.exists() or input_dir == '': 
            message = (
                f'Please input a directory of '
                f'sorted images to remove tags from.')
            self._warn(message)
            return
        files = list(dir.glob('*_*'))
        if len(files) == 0:
            self._warn('No files with sorting tags found.')
            return
                
        for file in files:
            split = file.name.split('_')[1:]
            name_name = '_'.join(split)
            new_path = pathlib.Path(file.parent.resolve(), name_name)
            if dry_run:
                log_line = f'{file.as_posix()} -> {new_path.as_posix()}'
                self.change_log.append(log_line)
            else: file.rename(new_path)

    ### DATASET SPLIT ###

    def split_dataset(self, dry_run: bool=False) -> None:
        '''Splits a folder of dataset images into [train, test, val].'''
        input_dir = self.find(QLineEdit, 'split_dataset_input').text()
        dir = pathlib.Path(input_dir)
        if not dir.exists() or input_dir == '': 
            self._warn('Please input a directory with images to split.')
            return
        files = list(dir.iterdir())
        if len(files) == 0:
            self._warn('No files to split.')
            return
        
        input_outdir = self.find(QLineEdit, 'split_dataset_output').text()
        outdir = pathlib.Path(input_outdir)
        if not outdir.exists() or input_outdir == '': 
            self._warn('Please input an output directory for the dataset.')
            return
        
        test_percent = self.find(QSpinBox, 'split_dataset_test').value()
        train_percent = self.find(QSpinBox, 'split_dataset_train').value()
        val_percent = self.find(QSpinBox, 'split_dataset_val').value()
        
        total = test_percent + train_percent + val_percent
        
        test_percent /= total
        train_percent /= total
        val_percent /= total
        
        test = pathlib.Path(outdir, 'test')
        train = pathlib.Path(outdir, 'train')
        val = pathlib.Path(outdir, 'val')
        if not dry_run:
            for i in [test, train, val]:
                try: i.mkdir()
                except Exception as e:
                    message = (
                        f'Unable to create output directory: '
                        f'{i.as_posix()}')
                    raise RuntimeError(message) from e

        for file in files:
            value = random.random()
            if value < test_percent: new_path = pathlib.Path(test, file.name)
            elif value > test_percent and value < test_percent + train_percent:
                new_path = pathlib.Path(train, file.name)
            else: new_path = pathlib.Path(val, file.name)
            
            if dry_run:
                log_line = f'{file.as_posix()} -> {new_path.as_posix()}'
                self.change_log.append(log_line)
            else:
                try: shutil.copy(file, new_path)
                except Exception as e:
                    message = f'Unable to copy file: {file.as_posix()}'
                    raise RuntimeError(message) from e