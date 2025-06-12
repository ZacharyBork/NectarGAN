from PySide6.QtWidgets import (
    QWidget, QPushButton, QFrame, QCheckBox, QSpinBox, QDoubleSpinBox, QSlider,
    QStackedWidget, QSplitter, QLabel, QLCDNumber, QLineEdit, QVBoxLayout,
    QProgressBar
)
import PySide6.QtGui as QtGui
from PySide6.QtCore import Qt

import pix2pix_graphical.toolbox.utils.common as utils
from pix2pix_graphical.toolbox.utils.callbacks import Callbacks

from pix2pix_graphical.toolbox.components.settings_dock import SettingsDock
from pix2pix_graphical.toolbox.helpers.trainer_helper import TrainerHelper
from pix2pix_graphical.toolbox.widgets.imagelabel import ImageLabel
from pix2pix_graphical.toolbox.widgets.graph import Graph


class InitHelper():
    def __init__(self, mainwidget: QWidget) -> None:
        self.mainwidget = mainwidget
        self.find = self.mainwidget.findChild
        self.callbacks = Callbacks(mainwidget=mainwidget)

    def _init_training_controls(
            self,
            trainerhelper: TrainerHelper, 
            settings_dock: SettingsDock
        ) -> None:
        '''Inits the training control buttons and progress bars.'''
        self.find(QProgressBar, 'train_progress').setValue(0)
        self.find(QProgressBar, 'epoch_progress').setValue(0)
        self.find(QPushButton, 'train_start').clicked.connect(
            lambda : self.callbacks._start_train(trainerhelper, settings_dock))
        self.find(QPushButton, 'train_stop').clicked.connect(
            lambda : self.callbacks._stop_train(trainerhelper, settings_dock))
        self.find(QPushButton, 'train_stop').setHidden(True)

        self.find(QPushButton, 'train_pause').setHidden(True)
        self.find(QPushButton, 'train_pause').clicked.connect(
            trainerhelper.pause_train)

    def _swap_image_labels(self) -> None:
        '''Swaps image labels from the .ui file with custom QLabel wrapper.'''
        for name in ['x_label', 'y_label', 'y_fake_label']:
            old_label = self.find(QLabel, name)
            layout = old_label.parentWidget().layout()

            new_label = ImageLabel(
                name=name, pixmap=None, 
                parent=self.mainwidget)
            new_label.setObjectName(name)
            new_label.setMinimumSize(old_label.minimumSize())
            new_label.setSizePolicy(old_label.sizePolicy())
            new_label.setAlignment(old_label.alignment()) 

            layout.replaceWidget(old_label, new_label)
            old_label.deleteLater()

    def _init_close_settings_buttons(self) -> None:
        '''Inits the button that lets the user close the settings dock.'''
        button = self.find(QPushButton, 'close_experiment_settings')
        utils.set_button_icon('caret-line-left-bold.svg', button)
        button.clicked.connect(self.callbacks._close_settings_dock)

    def _init_training_setting(self) -> None:
        '''Initializes widgets related to training config settings.'''
        self.find(QFrame, 'disc_schedule_box').setHidden(True)
        self.find(QCheckBox, 'separate_lr_schedules').clicked.connect(
            self.callbacks._split_lr_schedules)  
        continue_train_settings = self.find(QFrame, 'continue_train_settings')
        continue_train_settings.setHidden(True)
        self.find(QCheckBox, 'continue_train').clicked.connect(
            lambda value : continue_train_settings.setHidden(not value))  
        
        save_model_settings = self.find(QFrame, 'save_model_settings')
        self.find(QCheckBox, 'save_model').clicked.connect(
            lambda value : save_model_settings.setHidden(not value))    

        save_examples_settings = self.find(QFrame, 'save_examples_settings')
        self.find(QCheckBox, 'save_examples').clicked.connect(
            lambda value : save_examples_settings.setHidden(not value))
        
        visdom_settings = self.find(QFrame, 'visdom_settings')
        visdom_settings.setHidden(True)
        self.find(QCheckBox, 'visdom_enable').clicked.connect(
            lambda value : visdom_settings.setHidden(not value))

    def _init_current_epoch_display(self) -> None:
        '''Initializes the current epoch display QLCD widget.'''
        current_epoch = self.find(QLCDNumber, 'current_epoch')
        current_epoch.setSegmentStyle(QLCDNumber.SegmentStyle.Flat)
        palette = current_epoch.palette()
        palette.setColor(
            QtGui.QPalette.ColorRole.WindowText, QtGui.QColor('black'))
        current_epoch.setPalette(palette)

    def _init_browse_dir_buttons(self) -> None:
        '''Inits buttons for browsing for system directory paths'''
        pairs = [
            ('browse_output', 'output_directory'),
            ('browse_dataset', 'dataroot'),
            ('test_browse_experiment', 'test_experiment_path'),
            ('test_browse_dataset', 'test_dataset_path'),
            ('review_browse_experiment', 'review_experiment_path'),
            ('browse_pair_input_a', 'pair_images_input_a'),
            ('browse_pair_input_b', 'pair_images_input_b'),
            ('browse_pair_output', 'pair_images_output')
        ]
        for button_name, line_edit_name in pairs:
            button = self.find(QPushButton, button_name)
            line_edit = self.find(QLineEdit, line_edit_name)
            button.clicked.connect(
                lambda x, y=line_edit: utils.browse_directory(y))

    def _init_misc_settings(self, trainerhelper: TrainerHelper) -> None:
        '''Init widgets in the general `Settings` page.'''
        self.mainwidget.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)
        self.find(QCheckBox, 'always_on_top').clicked.connect(
            lambda x : self.callbacks._set_always_on_top(x))

        update_freq = self.find(QSpinBox, 'update_frequency')
        update_freq.valueChanged.connect(
            lambda x=update_freq.value() : 
                trainerhelper._change_update_frequency(x))

    def _set_start_page_indices(self) -> None:
        '''Sets indices of stacked widgets in the UI to correct start pages.'''
        self.find(QStackedWidget, 'centralwidget_pages').setCurrentIndex(0)
        self.find(QStackedWidget, 'settings_pages').setCurrentIndex(0)
        self.find(QStackedWidget, 'train_page_state_swap').setCurrentIndex(0)
        splitter = self.find(QSplitter, 'visualizer_splitter')
        splitter.setSizes([1000 - 250, 250])

    ### WIDGET LINKERS ###

    def _sliders_to_spinboxes(self) -> None:
        '''Links spinboxes to their respective sliders and vice versa.'''
        sliders_to_link = [
            # (slider, spinbox, multiplier)
            ('update_frequency_slider', 'update_frequency', 1.0),

            ('batch_size_slider', 'batch_size', 1.0),
            ('h_flip_chance_slider', 'h_flip_chance', 1.0),
            ('v_flip_chance_slider', 'v_flip_chance', 1.0),
            ('rot90_chance_slider', 'rot90_chance', 1.0),
            ('colorjitter_chance_slider', 'colorjitter_chance', 1.0),

            ('gen_lr_initial_slider', 'gen_lr_initial', 0.0001),
            ('gen_lr_target_slider', 'gen_lr_target', 0.0001),
            ('gen_optim_beta1_slider', 'gen_optim_beta1', 0.01),
            
            ('disc_lr_initial_slider', 'disc_lr_initial', 0.0001),
            ('disc_lr_target_slider', 'disc_lr_target', 0.0001),
            ('disc_optim_beta1_slider', 'disc_optim_beta1', 0.01),
            
            ('lambda_l1_slider', 'lambda_l1', 1.0),
            ('lambda_sobel_slider', 'lambda_sobel', 1.0),
            ('lambda_laplacian_slider', 'lambda_laplacian', 1.0),
            ('lambda_vgg_slider', 'lambda_vgg', 1.0)]

        for x, y, z in sliders_to_link:
            slider = self.find(QSlider, x)
            spinbox = self.find(QSpinBox, y)
            if spinbox == None: spinbox = self.find(QDoubleSpinBox, y)
            slider.sliderMoved.connect(
                lambda value, s=spinbox, sl=slider, m=z: 
                    self.callbacks._update_spinbox_from_slider(s, sl, m))
            spinbox.valueChanged.connect(
                lambda value, s=spinbox, sl=slider, m=z: 
                    self.callbacks._update_spinbox_from_slider(sl, s, 1/m))
            
    ### GRAPHS ###

    def _init_performance_graph(self) -> Graph:
        '''Initializes widgets related to the performance graph.
        
        Returns:
            Graph : The created perfomance graph.
        '''
        performance = Graph(window_title='performance')
        performance.add_line(name='epoch_time', color=(255, 0, 0))
        performance.setObjectName('performance_graph')
        layout = self.mainwidget.findChild(
            QVBoxLayout, 'performance_graph_layout')
        layout.addWidget(performance)

        self.find(
            QPushButton, 'performance_graph_reset_framing'
        ).clicked.connect(lambda : self.graphs['performance'].reframe_graph())
        self.find(
            QPushButton, 'performance_graph_clear'
        ).clicked.connect(lambda : self.graphs['performance'].reset_graph())

        return performance
    
    def _init_loss_graphs(self) -> tuple[Graph, Graph]:
        '''Inits the generator and discriminator loss graphs.
        
        Returns:
            tuple : The loss graphs as: (`loss_g`, `loss_d`)
        '''
        loss_g = Graph(window_title='loss_g', bottom_label='Epoch')
        loss_g.add_line(name='G_GAN', color=(255, 0, 0))
        loss_g.add_line(name='G_L1', color=(0, 255, 0))
        loss_g.add_line(name='G_SOBEL', color=(0, 0, 255))
        loss_g.add_line(name='G_LAP', color=(255, 255, 0))
        loss_g.add_line(name='G_VGG', color=(255, 0, 255))
        loss_g.setObjectName('loss_g_graph')
        layout = self.mainwidget.findChild(QVBoxLayout, 'loss_g_graph_layout')
        layout.addWidget(loss_g)

        loss_d = Graph(window_title='loss_d', bottom_label='Epoch')
        loss_d.add_line(name='D_real', color=(0, 255, 0))
        loss_d.add_line(name='D_fake', color=(255, 0, 0))
        loss_d.setObjectName('loss_d_graph')
        layout = self.mainwidget.findChild(QVBoxLayout, 'loss_d_graph_layout')
        layout.addWidget(loss_d)

        return loss_g, loss_d

    def _build_graphs(self) -> None:
        '''Inits the various graphs in the interfaces and adds their lines.'''
        performance = self._init_performance_graph()
        loss_g, loss_d = self._init_loss_graphs()
        
        self.graphs = {
            'performance': performance,
            'loss_g': loss_g,
            'loss_d': loss_d
        }

    def init(
            self, 
            trainerhelper: TrainerHelper, 
            settings_dock: SettingsDock
        ) -> None:
        # Link callbacks for training controls
        self._init_training_controls(trainerhelper, settings_dock) 
        self._sliders_to_spinboxes()   # Link sliders to their spinboxes
        self._swap_image_labels()      # Replace QLabels with custom wrappers
        self._init_close_settings_buttons()
        self._init_training_setting()
        self._init_browse_dir_buttons()
        self._init_misc_settings(trainerhelper=trainerhelper)
        self._set_start_page_indices()
            