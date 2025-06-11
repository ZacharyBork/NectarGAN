import json
import math
from pathlib import Path

from PySide6.QtGui import QPixmap, QImage
from PySide6.QtWidgets import (
    QWidget, QFrame, QLineEdit, QLabel, QVBoxLayout, QHBoxLayout, QTextEdit,
    QComboBox
)

from pix2pix_graphical.toolbox.widgets.graph import Graph

class ReviewPanel():
    def __init__(self, mainwidget: QWidget | None=None):
        self.mainwidget = mainwidget
        self.find = self.mainwidget.findChild
        self.images_layout = self.find(QVBoxLayout, 'review_images_layout')
        self.graphs_layout = self.find(QVBoxLayout, 'review_graphs_layout')
        self.train_config_display = self.find(
            QTextEdit, 'review_train_config')
        self.select_config = self.find(
            QComboBox, 'review_select_train_config')

        self.image_size = 300
        self.image_labels = []

        self.graph_sample_rate = 50

    ### UTILITIES ###

    def _clear_review_panel(self) -> None:
        for i in reversed(range(self.images_layout.count())): 
            self.images_layout.itemAt(i).widget().setParent(None) 
        for i in reversed(range(self.graphs_layout.count())): 
            self.graphs_layout.itemAt(i).widget().setParent(None) 

    def set_graph_sample_rate(self, sample_rate: int) -> None:
        self.graph_sample_rate = max(1, sample_rate)

    ### EXAMPLE IMAGES ###

    def _get_image_sets(
            self, 
            source_directory: Path
        ) -> list[list[Path]]:
        '''Sorts training example images into their sets.
        
        Since the example image count is configurable and might even change
        between different training sessions of the same model, this function 
        has to do some kind of strange looping logic to ensure every image is 
        found and ordered correctly. There is probably a better way to do this 
        but every other way I tried kept getting broken by edge cases.

        Args:
            source_directory : A `pathlib.Path` pointing to the directory of
                example images to load (i.e. `my_cool_experiment/examples`).

        Returns:
            list[list[pathlib.Path]] : A list of lists, each of which contains 
                three Paths to the individual images in a [x, y, y_fake] group.
        '''
        images = []
        for image in source_directory.glob('*.png'):
            split = image.name.split('_')
            epoch = split[0].replace('epoch', '')
            index = split[1]
            value = int(f'{epoch}{index}')
            images.append((value, image.as_posix()))

        images = sorted(images, key=lambda x: x[0])
        self.image_sets = []
        group = []
        for idx, image in enumerate(images):
            group.append(image[1])
            if (idx+1) % 3 == 0:
                self.image_sets.append(group)
                group = []

    def _build_image_set(
            self,
            epoch: int,
            images: list[Path]
        ) -> None:
        '''Builds a layout with an epoch label and a set of 3 image labels.
        
        Args:
            epoch : The epoch that the images belong to.
            images : The set of x, y, y_fake images to add.
        '''
        layout = QHBoxLayout()
        layout.setSpacing(2)
        layout.setContentsMargins(2, 2, 2, 2)

        # Default sort is alphabetical which puts real before fake so
        # we "swizzle" the image list to correct the order for the loop
        images = [images[0], images[2], images[1]]
        
        epoch_label = QLabel(f'Epoch: {epoch}')
        layout.addWidget(epoch_label)
        for image in images:
            label = QLabel('')
            pixmap = QPixmap(QImage(image)).scaledToWidth(self.image_size)
            label.setPixmap(pixmap)
            layout.addWidget(label)
            self.image_labels.append(label)
        
        frame = QFrame()
        frame.setLayout(layout)
        self.images_layout.addWidget(frame)

    def _build_image_sets(self) -> None:
        '''Builds image label layouts for all image sets.'''
        for idx, set in enumerate(self.image_sets):
            self._build_image_set(epoch=idx+1, images=set)

    ### GRAPH LOSSES ###

    def _graph_losses(self, loss_log: Path) -> None:
        with open(loss_log.as_posix(), 'r') as file:
            log_data = json.loads(file.read())
        losses = log_data['LOSSMANAGER_LOG']['loss_functions']
        dataset_length = log_data['LOSSMANAGER_LOG']['dataset_length']
        
        for key, value in losses.items():
            loss, weights = value['loss'], value['weights']
            
            graph = Graph(key)
            graph.add_line('loss', color=(255, 0, 0))
            graph.add_line('weight', color=(0, 255, 0))
            for i in range(len(loss)):
                if not i % self.graph_sample_rate == 0: continue
                step = ((i % dataset_length) / dataset_length)
                step += math.floor(i / dataset_length)
                graph.update_plot('loss', loss[i], step)
                graph.update_plot('weight', weights[i], step)
            
            layout = QVBoxLayout()
            layout.setSpacing(2)
            layout.setContentsMargins(2, 2, 2, 2)

            frame = QFrame()
            frame.setLayout(layout)
            
            label = QLabel(key)
            layout.addWidget(label)
            layout.addWidget(graph)

            self.graphs_layout.addWidget(frame)

    ### DISPLAY TRAIN CONFIG ###

    def _get_config_files(self, experiment_dir: Path) -> None:
        self.config_files = sorted(list(experiment_dir.glob('*config.json')))
        if len(self.config_files) == 0: return
        self.select_config.addItems([i.name for i in self.config_files])
        self.select_config.setCurrentIndex(len(self.config_files)-1)

    def load_train_config(self) -> None:
        index = self.select_config.currentIndex()
        config_file = self.config_files[index]
        with open(config_file.as_posix(), 'r') as file:
            data = json.loads(file.read())
        self.train_config_display.clear()
        self.train_config_display.setText(json.dumps(data, indent=4))

    ### LOAD EXPERIMENT ###

    def load_experiment(self) -> None:
        self._clear_review_panel()

        experiment_dir = Path(
            self.find(QLineEdit, 'review_experiment_path').text())
        examples_dir = Path(experiment_dir, 'examples')
        loss_log = Path(experiment_dir, 'loss_log.json')
        
        self._get_image_sets(examples_dir)
        self._build_image_sets()

        self._graph_losses(loss_log)

        self._get_config_files(experiment_dir)
        self.load_train_config()

