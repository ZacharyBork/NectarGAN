import shutil
import random
from pathlib import Path
from os import PathLike
from typing import Any, Literal

from PySide6.QtCore import QThread, QVariantAnimation
from PySide6.QtWidgets import (
    QWidget, QLabel, QPushButton, QFrame, QLineEdit, QComboBox, QProgressBar,
    QCheckBox, QTextEdit, QSpinBox)

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

    ### INIT HELPERS ###

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
        self.find(QPushButton, 'start_copy_sort'
            ).clicked.connect(self.copy_image_sort)
        self.find(QPushButton, 'preview_copy_sort'
            ).clicked.connect(lambda : self.copy_image_sort(dry_run=True))
        
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
        self.split_dataset_label = self.find(
            QLabel, 'split_dataset_starting_label')
        self.split_dataset_label.setHidden(True)

        self._init_starting_animation()

    def _init_starting_animation(self) -> None:
        self.starting_anim = QVariantAnimation()
        self.starting_anim.setStartValue(0)
        self.starting_anim.setEndValue(1000)
        self.starting_anim.setDuration(500000)
        self.starting_anim.valueChanged.connect(
            self._starting_animation)

    ### UTILITIES ###

    def _warn(self, warning: str) -> None:
        self.warning_label.setText(warning)

    def _validate_input_files(
            self, 
            lineedit_name: str,
            invalid_dir_warning: str,
            no_files_warning: str,
            key: str='*'
        ) -> tuple[bool, list[Path]]:
        input_dir = self.find(QLineEdit, lineedit_name).text()
        dir = Path(input_dir)
        if not dir.exists() or input_dir == '': 
            self._warn(invalid_dir_warning)
            return (False, [])
        files = list(dir.glob(key))
        if len(files) == 0:
            self._warn(no_files_warning)
            return (False, [])
        return (True, files)

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
        if not self.pairing_label.isHidden():
            self.starting_anim.stop()
            self.pairing_label.setHidden(True)
            self.pairing_progress.setHidden(False)
        self.pairing_progress.setValue(progress)

    def pair_images(self, dry_run: bool=False) -> None:
        '''Takes images from A|B folder and pairs them for pix2pix training.'''
        self.change_log.clear()

        valid, filesA = self._validate_input_files(
            'pair_images_input_a', 
            'Please input an A directory for pairing.', 
            'No files found in input directory A.')
        if not valid: return
        valid, filesB = self._validate_input_files(
            'pair_images_input_b', 
            'Please input an B directory for pairing.', 
            'No files found in input directory B.')
        if not valid: return

        input_outdir = self.find(QLineEdit, 'pair_images_output').text()
        outdir = Path(input_outdir)
        if not outdir.exists() or input_outdir == '': 
            self._warn('Please input an output directory for pairing.')
            return
        
        direction = self.find(QComboBox, 'pair_images_direction').currentText()
        if self.pair_do_scaling.isChecked():
            scale = int(self.pair_images_scale.currentText()[0:-1])
        else: scale = -1
        
        args = []
        for fileA, fileB in zip(filesA, filesB):
            args.append((fileA, fileB, outdir, direction, scale)) 

        if dry_run:
            for arg in args:
                x = (f'({arg[0]} + {arg[1]}) -> {Path(outdir, arg[0].name)}')
                self.change_log.append(x)
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
        valid, files = self._validate_input_files(
            'sort_images_input_dir', 
            'Please input a directory with images to sort.', 
            'No files found.')
        if not valid: return
        
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
        invalid_dir = (
            f'Please input a directory of '
            f'sorted images to remove tags from.')
        valid, files = self._validate_input_files(
            'remove_sort_tags_input', invalid_dir, 
            'No files with sorting tags found.', key='*_*')
        if not valid: return
                
        for file in files:
            split = file.name.split('_')[1:]
            name_name = '_'.join(split)
            new_path = Path(file.parent.resolve(), name_name)
            if dry_run:
                log_line = f'{file.as_posix()} -> {new_path.as_posix()}'
                self.change_log.append(log_line)
            else: file.rename(new_path)

    def copy_image_sort(self, dry_run: bool=False) -> None:
        valid, copyfrom_files = self._validate_input_files(
            'copy_sort_copy_from',
            'Please input a folder of sorted images to copy the order from.',
            'No sorted files found.')
        if not valid: return
        valid, copyto_files = self._validate_input_files(
            'copy_sort_copy_to',
            'Please input a folder of images to copy the order to.',
            'No files to sort.')
        if not valid: return
        copyto_dir = copyto_files[0].parent.resolve()
        for file in copyfrom_files:
            sorted_name = file.name
            original_name = '_'.join(file.name.split('_')[1:])
            other_file = Path(copyto_dir, original_name)
            if not other_file.exists():
                message = f'Unable to find match for file: {file.as_posix()}'
                self._warn(message)
                return
            new_path = Path(copyto_dir, sorted_name)
            if dry_run:
                log_line = f'{other_file.as_posix()} -> {new_path.as_posix()}'
                self.change_log.append(log_line)
            else: other_file.rename(new_path)

    ### DATASET SPLIT ###

    def _get_dataset_splits(self) -> None:
        total = 0
        chances = []
        for x in ['test', 'train', 'val']:
            value = self.find(QSpinBox, f'split_dataset_{x}').value()
            chances.append(value)
            total += value
        normalized = [i/total for i in chances]
        return normalized

    def _build_split_output_dirs(
            self,
            test: Path,
            train: Path,
            val: Path,
        ) -> None:
        for i in [test, train, val]:
            try: i.mkdir()
            except Exception as e:
                msg = (f'Unable to create output directory: {i.as_posix()}')
                raise RuntimeError(msg) from e
        return (test, train, val)

    def split_dataset(self, dry_run: bool=False) -> None:
        '''Splits a folder of dataset images into [train, test, val].'''
        valid, files = self._validate_input_files(
            'split_dataset_input',
            'Please input a directory with images to split.',
            'No files to split.')
        if not valid: return

        input_outdir = self.find(QLineEdit, 'split_dataset_output').text()
        outdir = Path(input_outdir)
        if not outdir.exists() or input_outdir == '': 
            self._warn('Please input an output directory for the dataset.')
            return

        test_percent, train_percent, val_percent = self._get_dataset_splits()
        test, train, val = (
            Path(outdir, 'test'), 
            Path(outdir, 'train'), 
            Path(outdir, 'val'))
        if not dry_run: self._build_split_output_dirs(test, train, val)

        for file in files:
            value = random.random()
            if value < test_percent: new_path = Path(test, file.name)
            elif value > test_percent and value < test_percent + train_percent:
                new_path = Path(train, file.name)
            else: new_path = Path(val, file.name)
            
            if dry_run:
                log_line = f'{file.as_posix()} -> {new_path.as_posix()}'
                self.change_log.append(log_line)
            else:
                try: shutil.copy(file, new_path)
                except Exception as e:
                    message = f'Unable to copy file: {file.as_posix()}'
                    raise RuntimeError(message) from e