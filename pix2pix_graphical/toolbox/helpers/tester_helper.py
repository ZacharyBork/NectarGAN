import json
import pathlib

from PySide6.QtCore import QObject, QThread, QSize
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtWidgets import (
    QWidget, QFrame, QLabel, QHBoxLayout, QVBoxLayout, 
    QFormLayout, QSizePolicy, QPushButton, QSlider, QComboBox)

from pix2pix_graphical.toolbox.workers.testerworker import TesterWorker
from pix2pix_graphical.toolbox.components.log import OutputLog

class TesterHelper(QObject):
    def __init__(self, mainwidget: QWidget, log: OutputLog) -> None:
        super().__init__()
        self.mainwidget = mainwidget
        self.log = log
        self.find = self.mainwidget.findChild

        self.image_layout = self.find(QVBoxLayout, 'test_image_layout')
        self.image_labels = { 'A_real': [], 'B_fake': [], 'B_real': [] }
        self.find(QLabel, 'test_progress_label').setHidden(True)
        self._add_sorting_options()

        self.test_thread: QThread | None=None
        self.worker: TesterWorker | None=None

        self.image_count = 30 
        self.image_load_size = 350

    ### RESULT SORTING ###

    def _add_sorting_options(self) -> None:
        self.sort_type = self.find(QComboBox, 'test_sort_type')
        self.sort_type.addItems([
            'Test Iteration', 'Input File Name', 'L1 Loss',
            'Sobel Loss', 'Laplacian Loss', 'Total Loss'])
        self.sort_type.currentTextChanged.connect(self._sort_results)

        self.sort_direction = self.find(QComboBox, 'test_sort_direction')
        self.sort_direction.addItems(['Ascending', 'Descending'])
        self.sort_direction.currentTextChanged.connect(self._sort_results)

    def _sort_results(self) -> None:
        sort_type = self.find(QComboBox, 'test_sort_type').currentText()
        sort_direction = self.find(QComboBox, 'test_sort_direction').currentText()
        
        containers = []
        keys = []
        for i in reversed(range(self.image_layout.count())): 
            container = self.image_layout.itemAt(i).widget()
            containers.append(container)
            match sort_type:
                case 'Test Iteration':
                    keys.append(container.findChild(QLabel, 'test_iterations').text())
                case 'Input File Name':
                    keys.append(container.findChild(QLabel, 'test_input_file'))
                case 'L1 Loss':
                    keys.append(container.findChild(QLabel, 'test_l1_loss'))
                case 'Sobel Loss':
                    keys.append(container.findChild(QLabel, 'test_sobel_loss'))
                case 'Laplacian Loss':
                    keys.append(container.findChild(QLabel, 'test_laplacian_loss'))
                case 'Total Loss':
                    l1 = float(container.findChild(QLabel, 'test_l1_loss'))
                    sobel = float(container.findChild(QLabel, 'test_sobel_loss'))
                    lap = float(container.findChild(QLabel, 'test_laplacian_loss'))
                    keys.append(l1 + sobel + lap)
            container.setParent(None) 
        
        paired = list(zip(keys, containers))
        paired.sort()
        if not sort_direction == 'Ascending': paired = reversed(paired)
        sorted_containers = [i[1] for i in paired]
        for i in sorted_containers: self.image_layout.addWidget(i)

            

    ### UTILS ###

    def _clear_cached_image_labels(self) -> None:
        self.image_labels = { 'A_real': [], 'B_fake': [], 'B_real': [] }

    def get_all_image_labels(self) -> list[QLabel]:
        a_reals = self.image_labels['A_real']
        b_fakes = self.image_labels['B_fake']
        b_reals = self.image_labels['B_real']
        return a_reals + b_fakes + b_reals

    ### CALLBACKS ###

    def change_image_scale(self) -> None:
        image_labels = self.get_all_image_labels()
        scale = self.find(QSlider, 'test_image_scale').value()
        for label in image_labels:
            new_size = int(float(self.image_load_size * scale) * 0.01)
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
            ('Test Iteration:', f'{log_entry['iteration']}', 'test_iterations'),
            ('Test Image:', log_entry['test_data_path'], 'test_input_file'),
            ('L1 Loss:', f'{log_entry['losses']['L1']}', 'test_l1_loss'),
            ('Sobel Loss:', f'{log_entry['losses']['SOBEL']}', 'test_sobel_loss'),
            ('Laplacian Loss:', f'{log_entry['losses']['LAPLACIAN']}', 'test_laplacian_loss')
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

    def load_test_results(self, results_dir: pathlib.Path) -> None:
        progress_label = self.find(QLabel, 'test_progress_label')        
        progress_label.setText('Finished. Loading results...') 
        log = pathlib.Path(results_dir, 'log.json')
        try: 
            with open(log.as_posix(), 'r') as file:
                log_data = json.loads(file.read())
        except Exception as e:
            raise RuntimeError(f'Unable to open test log: {log}') from e
        for i in range(self.image_count):
            layout = self.build_results_container()
            keys = ['A_real', 'B_fake', 'B_real']
            for key in keys:
                label = QLabel()
                label.setObjectName(key)
                self.image_labels[key].append(label)
                path = pathlib.Path(results_dir, f'{i+1}_{key}.png')
                pixmap = QPixmap(QImage(path.as_posix()))
                label.setPixmap(pixmap.scaledToWidth(self.image_load_size))
                layout.addWidget(label)

            log_entry = log_data['test']['results'][i]
            info = self.build_info_container(log_entry)
            layout.addWidget(info)
        progress_label.setHidden(True)
        self.find(QPushButton, 'test_start').setHidden(False)
            
    ### REPORT PROGRESS ###

    def _report_progress(self, iteration: int) -> None:
        label_text = f'Iterations Completed: {iteration}/{self.image_count}'
        self.find(QLabel, 'test_progress_label').setText(label_text)     

    ### START TEST ###

    def start_test(self) -> None:
        self.find(QPushButton, 'test_start').setHidden(True)
        self._clear_cached_image_labels()
        for i in reversed(range(self.image_layout.count())): 
            self.image_layout.itemAt(i).widget().setParent(None) 
        label_text = f'Iterations Completed: 0/{self.image_count}'
        progress_label = self.find(QLabel, 'test_progress_label')
        progress_label.setHidden(False)
        progress_label.setText(label_text)

        self.test_thread = QThread() 
        self.worker = TesterWorker(config=None, image_count=self.image_count)
        self.worker.moveToThread(self.test_thread)

        self.test_thread.started.connect(self.worker.run)
        
        self.worker.progress.connect(self._report_progress)
        self.worker.finished.connect(self.test_thread.quit)
        self.worker.finished.connect(self.load_test_results)
        
        self.test_thread.finished.connect(self.test_thread.deleteLater)
        self.test_thread.start()


