import json
import pathlib
from typing import Any

from PySide6.QtCore import QObject, QThread, QSize, QVariantAnimation
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtWidgets import (
    QWidget, QFrame, QLabel, QHBoxLayout, QVBoxLayout, QCheckBox, QLineEdit,
    QFormLayout, QSizePolicy, QPushButton, QSlider, QComboBox, QSpinBox)

from nectargan.toolbox.helpers.config_helper import ConfigHelper
from nectargan.toolbox.workers.testerworker import TesterWorker
from nectargan.toolbox.components.log import OutputLog
import nectargan.toolbox.utils.common as utils

class TesterHelper(QObject):
    def __init__(
            self, 
            mainwidget: QWidget, 
            confighelper: ConfigHelper,
            log: OutputLog
        ) -> None:
        super().__init__()
        self.mainwidget = mainwidget
        self.confighelper = confighelper
        self.log = log
        self.find = self.mainwidget.findChild

        self.image_count = 30 
        self.image_load_size = 350

        self.test_thread: QThread | None = None
        self.worker: TesterWorker | None = None

        self.results_dir: pathlib.Path | None = None
        self.results: dict[str, Any] | None = None
        self.previous_tests: dict[str, pathlib.Path] = {}

        self.find(QPushButton, 'test_start').clicked.connect(self.start_test)
        self.find(QSlider, 'test_image_scale'
            ).valueChanged.connect(self.change_image_scale)
        self.find(QPushButton, 'test_image_scale_reset'
            ).clicked.connect(self.reset_image_scale)
        
        test_dataset_override = self.find(QFrame, 'test_dataset_path_frame')
        test_dataset_override.setHidden(True)
        self.find(QCheckBox, 'test_override_dataset').clicked.connect(
            lambda x : test_dataset_override.setHidden(not x))
        
        self.find(QPushButton, 'test_get_most_recent').clicked.connect(
            self._testing_get_latest_epoch)
        
        self.image_layout = self.find(QVBoxLayout, 'test_image_layout')
        self.image_labels = { 'A_real': [], 'B_fake': [], 'B_real': [] }
        self.warning_label = self.find(QLabel, 'test_warning_label')
        self.warning_label.setHidden(True)
        self.warning_anim: QVariantAnimation | None = None
        self._init_warning_animation()
        self.find(QLabel, 'test_progress_label').setHidden(True)
        self.find(QComboBox, 'previous_tests').activated.connect(
            self._load_test_from_dropdown)
        self._add_sorting_options()

    def _testing_get_latest_epoch(self) -> None:
        exp_path = self.find(QLineEdit, 'test_experiment_path').text()
        latest_epoch = utils.get_latest_checkpoint_epoch(exp_path)
        self.find(QSpinBox, 'test_load_epoch').setValue(latest_epoch)

    ### PREVIOUS TEST HANDLING ###

    def _load_test_from_dropdown(self) -> None:
        key = self.find(QComboBox, 'previous_tests').currentText()
        self.results_dir = self.previous_tests[key]
        self.load_test_results()

    def _update_prev_test_dropdown(self) -> None:
        keys = [i for i in self.previous_tests.keys()]
        dropdown = self.find(QComboBox, 'previous_tests')
        dropdown.clear()
        dropdown.addItems(keys)
        dropdown.setCurrentIndex(len(keys)-1)

    ### RESULT SORTING ###

    def _add_sorting_options(self) -> None:
        self.sort_type = self.find(QComboBox, 'test_sort_type')
        self.sort_type.addItems([
            'Test Iteration', 'Input File Name', 'L1 Loss',
            'Sobel Loss', 'Laplacian Loss', 'Total Loss'])
        self.sort_type.currentTextChanged.connect(self.load_test_results)

        self.sort_direction = self.find(QComboBox, 'test_sort_direction')
        self.sort_direction.addItems(['Ascending', 'Descending'])
        self.sort_direction.currentTextChanged.connect(self.load_test_results)

    def _get_total_loss(self, result: dict[str, Any]) -> float:
        l1 = result['losses']['L1']
        sobel = result['losses']['SOBEL']
        laplacian = result['losses']['LAPLACIAN']
        return l1 + sobel + laplacian

    def _sort_results(
            self, 
            results: list[dict[str, Any]]
        ) -> list[dict[str, Any]]:
        _type = self.find(QComboBox, 'test_sort_type').currentText()
        _direction = self.find(QComboBox, 'test_sort_direction').currentText()     
        keys = {
            'Test Iteration': lambda x: x['iteration'],
            'Input File Name': lambda x: x['test_data_path'],
            'L1 Loss': lambda x: x['losses']['L1'],
            'Sobel Loss': lambda x: x['losses']['SOBEL'],
            'Laplacian Loss': lambda x: x['losses']['LAPLACIAN'],
            'Total Loss': lambda x: self._get_total_loss(x)}   
        results = sorted(results, key=keys[_type])
        if _direction == 'Descending': results.reverse()
        return results

    ### UTILS ###

    def _clear_cached_image_labels(self) -> None:
        self.image_labels = { 'A_real': [], 'B_fake': [], 'B_real': [] }

    def get_all_image_labels(self) -> list[QLabel]:
        a_reals = self.image_labels['A_real']
        b_fakes = self.image_labels['B_fake']
        b_reals = self.image_labels['B_real']
        return a_reals + b_fakes + b_reals
    
    def _clear_image_layout(self) -> None:
        for i in reversed(range(self.image_layout.count())): 
            self.image_layout.itemAt(i).widget().setParent(None) 

    ### WARNING LABEL ###

    def _show_warning_label(self, val: int) -> None:
        show = not val > 0
        self.warning_label.setHidden(show)

    def _init_warning_animation(self) -> None:
        if not self.warning_anim is None: 
            self.warning_anim.deleteLater()
        self.warning_anim = QVariantAnimation()
        self.warning_anim.setStartValue(10000)
        self.warning_anim.setEndValue(0)
        self.warning_anim.setDuration(3000)
        self.warning_anim.valueChanged.connect(
            lambda x : self.warning_label.setHidden(not x > 0))

    def _warn(self, warning: str) -> None:
        self._init_warning_animation()
        self.warning_label.setText(warning)
        self.warning_anim.start()

    ### CALLBACKS ###

    def change_image_scale(self) -> None:
        image_labels = self.get_all_image_labels()
        scale = self.find(QSlider, 'test_image_scale').value()
        new_size = int(float(self.image_load_size * scale) * 0.01)
        
        column_label_size = QSize(
            new_size, self.find(QLabel, 'test_a_real_label').height())
        self.find(QLabel, 'test_a_real_label').setFixedSize(column_label_size)
        self.find(QLabel, 'test_b_fake_label').setFixedSize(column_label_size)
        self.find(QLabel, 'test_b_real_label').setFixedSize(column_label_size)

        for label in image_labels:
            label.setFixedSize(QSize(new_size, new_size))
            label.setScaledContents(True)
        

    def reset_image_scale(self) -> None:
        self.find(QSlider, 'test_image_scale').setValue(100)

    ### LOAD RESULTS ###

    def build_results_container(self) -> QHBoxLayout:
        frame = QFrame()
        frame.setObjectName('test_results')
        frame.setStyleSheet(
            '#test_results { border: 2px ridge rgb(100, 100, 100); }')
        
        layout = QHBoxLayout()
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)
        frame.setLayout(layout)
        self.image_layout.addWidget(frame)        
        
        return layout

    def build_info_container(self, log_entry: dict) -> QWidget:
        rows = [
            ('Test Iteration:', f'{log_entry["iteration"]}', 'test_iterations'),
            ('Test Image:', log_entry["test_data_path"], 'test_input_file'),
            ('L1 Loss:', f'{log_entry["losses"]["L1"]}', 'test_l1_loss'),
            ('Sobel Loss:', f'{log_entry["losses"]["SOBEL"]}', 'test_sobel_loss'),
            ('Laplacian Loss:', f'{log_entry["losses"]["LAPLACIAN"]}', 'test_laplacian_loss')
        ]

        layout = QFormLayout()
        for label, field, name in rows:
            _label = QLabel(label)
            _field = QLabel(field)
            _field.setObjectName(name)
            layout.addRow(_label, _field)

        frame = QWidget()
        frame.setObjectName('test_results_info')
        frame.setStyleSheet(
            '#test_results_info { border: 1px ridge rgb(100, 100, 100); }')
        frame.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        frame.setLayout(layout)

        return frame
    
    def _parse_test_log(self) -> None:
        if self.results_dir is None: return
        log = pathlib.Path(self.results_dir, 'log.json')
        try: 
            with open(log.as_posix(), 'r') as file:
                log_data = json.loads(file.read())
        except Exception as e:
            raise RuntimeError(f'Unable to open test log: {log}') from e
        self.results = self._sort_results(log_data['test']['results'])

    def load_test_results(self) -> None:
        progress_label = self.find(QLabel, 'test_progress_label')        
        progress_label.setText('Finished. Loading results...') 
        self._clear_cached_image_labels()
        self._clear_image_layout()
        self._parse_test_log()
        if self.results is None: return

        for result in self.results:
            layout = self.build_results_container()
            for key in ['A_real', 'B_fake', 'B_real']:
                label = QLabel()
                label.setObjectName(key)
                self.image_labels[key].append(label)
                path = pathlib.Path(
                    self.results_dir, f'{result["iteration"]}_{key}.png')
                pixmap = QPixmap(QImage(path.as_posix()))
                label.setPixmap(pixmap.scaledToWidth(self.image_load_size))
                layout.addWidget(label)
            
            info = self.build_info_container(result)
            layout.addWidget(info)

        self.change_image_scale()
        progress_label.setHidden(True)
        self.find(QPushButton, 'test_start').setHidden(False)
        
    ### PROCESS SIGNALS ###

    def _set_results_dir(self, results_dir: pathlib.Path) -> None:
        self.results_dir = results_dir
        self.previous_tests[results_dir.name] = results_dir
        self.load_test_results()
        self._update_prev_test_dropdown()

    def _report_progress(self, iteration: int) -> None:
        label_text = f'Iterations Completed: {iteration}/{self.image_count}'
        self.find(QLabel, 'test_progress_label').setText(label_text)     

    ### HANDLE UI OPTIONS ###

    def _build_worker(self) -> bool:
        exp_dir_path = self.find(QLineEdit, 'test_experiment_path').text()
        experiment_dir = pathlib.Path(exp_dir_path)
        if exp_dir_path == '' or not experiment_dir.exists(): 
            self._warn('Please enter a valid experiment directory path.')
            return False

        dataroot = None
        if self.find(QCheckBox, 'test_override_dataset').isChecked():
            dataroot_path = self.find(QLineEdit, 'test_dataset_path').text()
            dataroot = pathlib.Path(dataroot_path)
            if dataroot_path == '' or not dataroot.exists(): 
                self._warn('Please enter a valid dataset directory path.')
                return False
        elif self.find(QLineEdit, 'dataroot').text() == '':
            message = (
                f'Dataset root is empty and no override selected.\n'
                f'Please select a dataset to continue.')
            self._warn(message)
            return False
        
        load_epoch = self.find(QSpinBox, 'test_load_epoch').value()
        pth = pathlib.Path(experiment_dir, f'epoch{load_epoch}_netG.pth.tar')
        if not pth.exists(): 
            self._warn(f'Unable to locate checkpoint file:\n{pth.as_posix()}')
            return False

        self.image_count = self.find(QSpinBox, 'test_iterations').value()
        self.confighelper._build_launch_config() 
        self.worker = TesterWorker(
            config=self.confighelper.config, image_count=self.image_count,
            experiment_dir=experiment_dir, dataroot=dataroot,
            load_epoch=load_epoch)
        return True

    ### START TEST ###

    def start_test(self) -> None:
        success = self._build_worker()
        if not success: return

        self.find(QPushButton, 'test_start').setHidden(True)
        self._clear_cached_image_labels()
        self._clear_image_layout() 
        label_text = f'Iterations Completed: 0/{self.image_count}'
        progress_label = self.find(QLabel, 'test_progress_label')
        progress_label.setHidden(False)
        progress_label.setText(label_text)

        self.test_thread = QThread() 
        self.worker.moveToThread(self.test_thread)
        self.test_thread.started.connect(self.worker.run)
        
        self.worker.progress.connect(self._report_progress)
        self.worker.finished.connect(self.test_thread.quit)
        self.worker.finished.connect(self._set_results_dir)
        
        self.test_thread.finished.connect(self.test_thread.deleteLater)
        self.test_thread.start()
