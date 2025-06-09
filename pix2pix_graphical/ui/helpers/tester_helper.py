import json
import pathlib

from PySide6.QtCore import QObject, QThread
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtWidgets import (
    QWidget, QFrame, QLabel, QHBoxLayout, QVBoxLayout, 
    QFormLayout, QSizePolicy, QPushButton)

from pix2pix_graphical.ui.workers.testerworker import TesterWorker
from pix2pix_graphical.ui.components.log import OutputLog

class TesterHelper(QObject):
    def __init__(self, mainwidget: QWidget, log: OutputLog) -> None:
        super().__init__()
        self.mainwidget = mainwidget
        self.log = log
        self.find = self.mainwidget.findChild

        self.image_layout = self.find(QVBoxLayout, 'test_image_layout')
        self.find(QLabel, 'test_progress_label').setHidden(True)

        self.test_thread: QThread | None=None
        self.worker: TesterWorker | None=None

        self.image_count = 30 
        self.image_load_size = 350

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
            ('Test Iteration:', f'{log_entry['iteration']}'),
            ('Test Image:', log_entry['test_data_path']),
            ('L1 Loss:', f'{log_entry['losses']['L1']}'),
            ('Sobel Loss:', f'{log_entry['losses']['SOBEL']}'),
            ('Laplacian Loss:', f'{log_entry['losses']['LAPLACIAN']}')
        ]

        layout = QFormLayout()
        for label, field in rows:
            layout.addRow( QLabel(label), QLabel(field))

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
            paths = [
                pathlib.Path(results_dir, f'{i+1}_A_real.png'), 
                pathlib.Path(results_dir, f'{i+1}_B_fake.png'),
                pathlib.Path(results_dir, f'{i+1}_B_real.png')]
            for path in paths:
                label = QLabel()
                pixmap = QPixmap(QImage(path.as_posix()))
                label.setPixmap(pixmap.scaledToWidth(self.image_load_size))
                layout.addWidget(label)

            log_entry = log_data['test']['results'][i]
            info = self.build_info_container(log_entry)
            layout.addWidget(info)
        progress_label.setHidden(True)
        self.find(QPushButton, 'test_start').setHidden(False)
            
    def _report_progress(self, iteration: int) -> None:
        label_text = f'Iterations Completed: {iteration}/{self.image_count}'
        self.find(QLabel, 'test_progress_label').setText(label_text)     

    def start_test(self) -> None:
        self.find(QPushButton, 'test_start').setHidden(True)
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


