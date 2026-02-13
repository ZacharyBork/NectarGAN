import sys
import json
from pathlib import Path
from importlib.resources import files
from typing import Any

from PySide6.QtWidgets import (
    QWidget, QPushButton, QApplication, QFileDialog, QLineEdit, QMessageBox, 
    QLabel, QVBoxLayout, QHBoxLayout, QFrame, QCheckBox, QSlider, QRadioButton,
    QSizePolicy)
from PySide6.QtUiTools import QUiLoader
from PySide6.QtCore import Qt, QFile, QObject, QTimer, QEvent
from PySide6.QtGui import QShortcut, QKeySequence, QPixmap

from nectargan.annotations.creator.src.widgets import InteractiveImageDisplay

class Interface(QObject):    
    def __init__(self, schema_version: int=1) -> None:
        super().__init__()
        self.schema_version = schema_version
        self.current_index = -1
        self.current_image: Path = None 
        self.query_widgets: dict[str, QWidget] = {}
        self.allow_override = False
        self.image_loaded = False

    def eventFilter(self, obj: QObject, event: QEvent) -> None:
        if obj is self.mainwidget and event.type() == QEvent.Type.Resize:
            if self.image_loaded: 
                self.image_display.draw_image(self.current_image)
        return super().eventFilter(obj, event)

    def _get_ui_file(self) -> QFile:
        root = Path(__file__).parent.resolve()
        file = Path(root, 'ui/annotation_creator.ui')
        if not file.exists():
            msg = f'Unable to locate UI file: {file.resolve().as_posix()}'
            raise FileNotFoundError(msg)
        return QFile(file.resolve().as_posix())

    def _init_mainwidget(self) -> None:
        '''Initializes a Qt main widget from the UI file.'''
        loader = QUiLoader()
        file = self._get_ui_file()
        
        file.open(QFile.ReadOnly)
        self.mainwidget = loader.load(file)
        file.close()
        self.mainwidget.setWindowTitle('Annotation Creator')
        self.mainwidget.installEventFilter(self)

    def _set_stylesheet(self) -> None:
        path = files('nectargan.toolbox.resources').joinpath('stylesheet.qss')
        file = Path(path)
        if not file.exists():
            msg = f'Unable to locate stylesheet: {file.resolve().as_posix()}'
            raise FileNotFoundError(msg)
        with open(file.resolve().as_posix(), 'r') as file:
            stylesheet = file.read()
            self.app.setStyleSheet(stylesheet)
            
    def _set_ui_state(self, state: str) -> None:
        match state:
            case 'config':
                self.main_frame.setDisabled(True)
                self.main_frame.setHidden(True)
                self.config_frame.setDisabled(False)
                self.config_frame.setHidden(False)
            case 'active':
                self.main_frame.setDisabled(False)
                self.main_frame.setHidden(False)
                self.config_frame.setDisabled(True)
                self.config_frame.setHidden(True)
                
    ### EXTRA CAPTIONS ###
    
    def _remove_extra_caption(self, x: QPushButton) -> None:
        x.parentWidget().deleteLater()
    
    def _add_extra_caption(self) -> None:
        layout = self.find(QVBoxLayout, 'extra_caption_layout')
        frame = QFrame()
        caption_layout = QHBoxLayout()
        caption_box = QLineEdit()
        
        remove_button = QPushButton(text='-')
        remove_button.setSizePolicy(
            QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Preferred)
        remove_button.clicked.connect(
            lambda : self._remove_extra_caption(x=remove_button))
        
        caption_layout.addWidget(caption_box)
        caption_layout.addWidget(remove_button)
        
        frame.setLayout(caption_layout)
        layout.addWidget(frame)
        
    def _get_extra_captions(self) -> list[str]:
        captions = []
        
        extra_captions_layout = self.find(QVBoxLayout, 'extra_caption_layout')   
        caption_count = extra_captions_layout.count()    
        
        for i in range(caption_count): 
            layout = extra_captions_layout.itemAt(i).widget().layout()
            for j in range(layout.count()):
                widget = layout.itemAt(j).widget()
                if isinstance(widget, QLineEdit):
                    current_text = widget.text()
                    if current_text != '':
                        captions.append(current_text)

        return captions
    
    def _destroy_extra_captions(self) -> None:
        extra_captions_layout = self.find(QVBoxLayout, 'extra_caption_layout') 
        for i in range(extra_captions_layout.count()): 
            extra_captions_layout.itemAt(i).widget().deleteLater()

    ### UTILS ###

    def _warn(self, message: str) -> None:
        QMessageBox.warning(
            None, 'Warning', message, QMessageBox.StandardButton.Ok)

    def _update_remaining(self) -> None:
        remaining = str(len(self.image_files))
        self.find(QLabel, 'images_remaining').setText(remaining)
        
    def _update_caption_override(self) -> None:
        enabled = self.find(QCheckBox, 'override_caption').isChecked()
        self.allow_override = enabled
        self.caption_text.setEnabled(enabled)

    def _update_example_caption(self) -> None:
        if self.allow_override: return
        
        caption = ''
        count = len(self.query_widgets.keys())
        for idx, (key, value) in enumerate(self.query_widgets.items()):
            for x in self.queries:
                if x['title'] == key: query = x
            
            match query['type']:
                case 'checkbox':
                    caption_key = 'caption_true' if value.isChecked() \
                        else 'caption_false'
                    caption += f'{query['settings'][caption_key]}'
                case 'slider':
                    current = str(value.value())
                    text = query['settings']['caption'].replace('{}', current)
                    caption += f'{text}'
                case 'radio_buttons':
                    layout = value.layout()
                    for i in range(layout.count()): 
                        if layout.itemAt(i).widget().isChecked():
                            current = query['settings']['choices'][i]
                    text = query['settings']['caption'].replace('{}', current)
                    caption += f'{text}'
            
            if idx == count-1: caption += '.'
            else: caption += ', '
        self.caption_text.setText(caption)

    def _build_query_ui(self) -> None:
        queries_layout = self.find(QVBoxLayout, 'queries_layout')
        for i in reversed(range(queries_layout.count())): 
            queries_layout.itemAt(i).widget().setParent(None)

        self.query_widgets.clear()

        for query in self.queries:
            query_layout = QHBoxLayout()
            query_layout.addWidget(QLabel(text=query['title']))
            match query['type']:
                case 'checkbox':
                    widget = QCheckBox()
                    widget.released.connect(self._update_example_caption)
                case 'slider':
                    widget = QSlider(Qt.Orientation.Horizontal)
                    widget.setMinimum(query['settings']['range'][0])
                    widget.setMaximum(query['settings']['range'][1])
                    widget.valueChanged.connect(self._update_example_caption)
                case 'radio_buttons':
                    widget = QFrame()
                    buttons_layout = QHBoxLayout()
                    widget.setLayout(buttons_layout)

                    choices = query['settings']['choices']
                    for choice in choices:
                        button = QRadioButton(text=choice)
                        button.clicked.connect(self._update_example_caption)
                        buttons_layout.addWidget(button)
                    buttons_layout.itemAt(0).widget().setChecked(True)
            
            query_layout.addWidget(widget)
                
            frame = QFrame()
            frame.setLayout(query_layout)
            queries_layout.addWidget(frame)

            self.query_widgets[query['title']] = widget

        self._update_example_caption()

    ### METADATA ###
    
    def _build_metadata_file(self) -> bool:
        version = self.schema_version
        match version:
            case 1:
                base = {
                    'info': {
                        'schema_version': version,
                        'total_captions': 0,
                        'total_images': 0
                    },
                    'items': {},
                    'other': { 'landmarks': {}, 'choices': {} }
                }
            case _: raise ValueError(f'Schema version not valid: {version}')
        input_outdir = self.find(QLineEdit, 'output_directory').text()
        output_directory = Path(input_outdir)
        if input_outdir.strip() == '' or not output_directory.exists():
            self._warn(
                f'Unable to locate output directory at path: '
                f'{output_directory.as_posix()}')
            return False
        
        self.metadata_file = Path(output_directory, 'metadata.json')
        if self.metadata_file.exists():
            message = (
                f'Found existing metadata file at path: '
                f'{self.metadata_file.as_posix()}\n\n'
                f'Press "Ok" to load existing file, or "Discard" to overwrite '
                f' the existing file.')
            buttons = QMessageBox.StandardButton
            choice = QMessageBox.warning(
                None, 'Existing Metadata File', message, 
                buttons.Ok | buttons.Discard | buttons.Cancel)
            if choice == buttons.Discard:
                message = (
                    f'This will delete the existing metadata file at path: '
                    f'{self.metadata_file.as_posix()}\n\n'
                    f'Are you sure you would like to continue?')
                confirm = QMessageBox.warning(
                    None, 'Warning', message,
                    buttons.Ok | buttons.Cancel)
                if confirm == buttons.Ok: self.metadata_file.unlink()
                else: return False
            else: return choice == buttons.Ok
        try:
            with open(self.metadata_file, 'w') as file:
                file.write(json.dumps(base, indent=4))
        except Exception as e:
            self._warn(f'Unable to write metadata file. Reason: {e}')
            return False
        return True

    def _load_metadata(self) -> dict[str, Any]:
        with open(self.metadata_file, 'r') as file:
            metadata = json.loads(file.read())
        return metadata

    def _write_metadata(self) -> None:
        metadata = self._load_metadata()
        items = metadata['items']
        choices = metadata['other']['choices']
        file_tag = self.current_image.stem
                
        caption = self.caption_text.text()
        items[file_tag] = {
            'filepath': self.current_image.as_posix(),
            'captions': [caption]
        }
        items[file_tag]['captions'].extend(self._get_extra_captions())
        
        choices[file_tag] = {}
        for query in self.queries:
            match query['type']:
                case 'checkbox':
                    value = self.query_widgets[query['title']].isChecked()
                case 'slider':
                    value = self.query_widgets[query['title']].value()
                case 'radio_buttons':
                    layout = self.query_widgets[query['title']].layout()
                    for i in range(layout.count()): 
                        if layout.itemAt(i).widget().isChecked():
                            value = query['settings']['choices'][i]
            
            choices[file_tag][query['title']] = value
            
        landmark_data = self.image_display.get_landmark_data()
        metadata['other']['landmarks'][file_tag] = landmark_data

        with open(self.metadata_file, 'w') as file:
            file.write(json.dumps(metadata, indent=4))
            
    ### IMAGE METHODS ###
        
    def _set_image_button_state(self) -> None:
        prev_btn = self.find(QPushButton, 'previous_image')
        next_btn = self.find(QPushButton, 'next_image')

        prev_btn.setEnabled(self.current_index != 0)
        next_btn.setEnabled(self.current_index != len(self.image_files) - 1)

    def _load_image(self, previous: bool=False) -> None:
        if not previous:
            self.current_index = min(
                len(self.image_files) - 1, self.current_index + 1)
        else: self.current_index = max(0, self.current_index - 1)
        self.current_image = self.image_files[self.current_index]
        
        self._set_image_button_state()
        self.image_display.reset()
        
        self.find(QCheckBox, 'override_caption').setChecked(False)
        self._update_caption_override()
        self._destroy_extra_captions()
        
        self._build_query_ui()
        self._update_remaining()
        self.image_display.draw_image(self.current_image)
        self.image_loaded = True
        
    def _apply_caption(self) -> None:
        self._write_metadata()
        self.image_files.remove(self.current_image)
        self.current_index -= 1
        self._load_image()

    ### CALLBACKS ###

    def _exit_app(self) -> None:
        sys.exit(self.app.exec())

    def _get_images(self) -> bool:
        input_dir = self.find(QLineEdit, 'image_directory').text()
        image_directory = Path(input_dir)
        if input_dir.strip() == '' or not image_directory.exists():
            self._warn('Image Directory must point to a valid directory!')
            return False
        
        self.image_files = list(image_directory.rglob('*.jpg'))
        if len(self.image_files) == 0:
            self._warn('No image files found in Image Directory!')
            return False
        
        existing = self._load_metadata()['items']
        temp = []
        for file in self.image_files:
            file_tag = file.stem
            if file_tag in existing.keys():
                temp.append(file)
        [self.image_files.remove(i) for i in temp]
        
        self._update_remaining()
        return True

    def _get_config(self) -> bool:
        input_file = self.find(QLineEdit, 'config_file').text()
        config_file = Path(input_file)
        if input_file.strip() == '' or not config_file.exists():
            self._warn(
                f'Unable to locate config file at path: '
                f'{config_file.as_posix()}')
            return False
        try:
            with open(config_file, 'r') as file:
                self.config = json.loads(file.read())
        except Exception as e:
            self._warn(f'Unable to load config file. Reason: {e}')
            return False
        
        self.queries = self.config['queries']
        return True

    def _load_set(self) -> None:
        success = self._build_metadata_file()
        if not success: return
        
        success = self._get_images()
        if not success: return

        success = self._get_config()
        if not success: return

        self._set_ui_state(state='active')
        self._load_image()
        self.image_display.set_landmark_ids(
            red=self.config['landmarks']['red'],
            green=self.config['landmarks']['green'],
            blue=self.config['landmarks']['blue'])
        
    def _init_callbacks(self) -> None:
        self.find(QPushButton, 'exit_btn').clicked.connect(self._exit_app)
        self.find(QPushButton, 'load_set').clicked.connect(self._load_set)
        self.find(QPushButton, 'next_image').clicked.connect(self._load_image)
        self.find(QPushButton, 'previous_image').clicked.connect(
            lambda : self._load_image(previous=True))
        self.find(QPushButton, 'apply_caption').clicked.connect(self._apply_caption)
        self.find(QCheckBox, 'override_caption').clicked.connect(self._update_caption_override)
        self.find(QPushButton, 'add_extra_caption').clicked.connect(self._add_extra_caption)

    ### ENTRYPOINT ###

    def run(self) -> None:
        '''Entrypoint function for `Interface` class. Launches the GUI.'''
        self.app = QApplication(sys.argv)
        # self._set_stylesheet()

        self._init_mainwidget()
        self.find = self.mainwidget.findChild
        self.main_frame = self.find(QFrame, 'main_frame')
        self.config_frame = self.find(QFrame, 'config_frame')
        self.image_frame = self.find(QFrame, 'image_frame')
        self.caption_text = self.find(QLineEdit, 'caption_text')
        self.caption_text.setEnabled(False)
        
        
        
        self.image_display = InteractiveImageDisplay()
        self.image_frame.layout().addWidget(
            self.image_display, alignment=Qt.AlignmentFlag.AlignCenter)
        
        self.find(QLineEdit, 'image_directory').setText(
            '/media/zach/UE/ML/test_data/diffusion/temp_celeba_raw/celeba/test')
        
        self.find(QLineEdit, 'config_file').setText(
            '/media/zach/UE/ML/NectarGAN/nectargan/annotations/creator/celeba_config.json')
        
        self.find(QLineEdit, 'output_directory').setText(
            '/media/zach/UE/ML/NectarGAN/nectargan/annotations/creator')


        self._init_callbacks()
        self._set_ui_state(state='config')

        self.mainwidget.show()

        sys.exit(self.app.exec())

if __name__ == "__main__":
    interface = Interface()
    interface.run()

