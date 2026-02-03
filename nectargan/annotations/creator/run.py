import sys
import json
from pathlib import Path
from importlib.resources import files

from PySide6.QtWidgets import (
    QWidget, QPushButton, QApplication, QFileDialog, QLineEdit, QMessageBox, 
    QLabel, QVBoxLayout, QHBoxLayout, QFrame, QCheckBox, QSlider, QRadioButton)
from PySide6.QtUiTools import QUiLoader
from PySide6.QtCore import Qt, QFile, QObject
from PySide6.QtGui import QShortcut, QKeySequence, QPixmap

class Interface(QObject):    
    def __init__(self, schema_version: int=1) -> None:
        super().__init__()
        self.schema_version = schema_version
        self.current_index = -1
        self.current_image: Path = None 
        self.query_widgets: dict[str, QWidget] = {}

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

    def _set_stylesheet(self) -> None:
        path = files('nectargan.toolbox.resources').joinpath('stylesheet.qss')
        file = Path(path)
        if not file.exists():
            msg = f'Unable to locate stylesheet: {file.resolve().as_posix()}'
            raise FileNotFoundError(msg)
        with open(file.resolve().as_posix(), 'r') as file:
            stylesheet = file.read()
            self.app.setStyleSheet(stylesheet)

    ### UTILS ###

    def _warn(self, message: str) -> None:
        QMessageBox.warning(
            None, 'Warning', message, QMessageBox.StandardButton.Ok)

    def _update_example_caption(self) -> None:
        caption = ''
        for key, value in self.query_widgets.items():
            for x in self.queries:
                if x['title'] == key:
                    query = x
            
            match query['type']:
                case 'checkbox':
                    caption_key = 'caption_true' if value.isChecked() \
                        else 'caption_false'
                    caption += f'{query['settings'][caption_key]}, '
                case 'slider':
                    current = str(value.value())
                    text = query['settings']['caption'].replace('{}', current)
                    caption += f'{text}, '
                case 'radio_buttons':
                    layout = value.layout()
                    for i in range(layout.count()): 
                        if layout.itemAt(i).widget().isChecked():
                            current = query['settings']['choices'][i]
                    text = query['settings']['caption'].replace('{}', current)
                    caption += f'{text}, '
        self.find(QLabel, 'caption_text').setText(caption)

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
            
            
            query_layout.addWidget(widget)
                
            frame = QFrame()
            frame.setLayout(query_layout)
            queries_layout.addWidget(frame)

            self.query_widgets[query['title']] = widget

    def _write_metadata(self, caption: str) -> None:
        with open(self.metadata_file, 'r') as file:
            metadata = json.loads(file.read())
        items = metadata['items']
        
        file_tag = self.current_image.stem
        items[file_tag] = {
            'filepath': self.current_image.as_posix(),
            'captions': [caption]
        }

        with open(self.metadata_file, 'w') as file:
            file.write(json.dumps(metadata, indent=4))

    ### IMAGE METHODS ###

    def _load_image(self, previous: bool=False) -> None:
        if not previous:
            self.current_index = min(
                len(self.image_files) - 1, self.current_index + 1)
        else: self.current_index = max(0, self.current_index - 1)
        self.current_image = self.image_files[self.current_index]

        pixmap = QPixmap(self.current_image)
        image_label = self.find(QLabel, 'image_display')
        image_label.setPixmap(pixmap)
        image_label.setScaledContents(True)

        self._build_query_ui()

    def _previous_image(self) -> None:
        self._load_image(previous=True)

    def _next_image(self) -> None:
        caption = 'Test caption'
        self._write_metadata(caption=caption)
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

    def _build_metadata_file(self) -> bool:
        version = self.schema_version
        match version:
            case 1:
                base = {
                    "info": {
                        "schema_version": version,
                        "total_captions": 0,
                        "total_images": 0
                    },
                    "items": {},
                    "other": {}
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
            self._warn(
                f'Found existing metadata file at path: '
                f'{self.metadata_file.as_posix()}')
            return False
        try:
            with open(self.metadata_file, 'w') as file:
                file.write(json.dumps(base, indent=4))
        except Exception as e:
            self._warn(f'Unable to write metadata file. Reason: {e}')
            return False
        return True
    
    def _load_set(self) -> None:
        success = self._get_images()
        if not success: return

        success = self._get_config()
        if not success: return

        success = self._build_metadata_file()
        if not success: return
        
        self._load_image()

    def _init_callbacks(self) -> None:
        self.find(QPushButton, 'exit_btn').clicked.connect(self._exit_app)
        self.find(QPushButton, 'load_set').clicked.connect(self._load_set)
        self.find(QPushButton, 'next_image').clicked.connect(self._next_image)
        self.find(QPushButton, 'previous_image').clicked.connect(self._previous_image)

    ### ENTRYPOINT ###

    def run(self) -> None:
        '''Entrypoint function for `Interface` class. Launches the GUI.'''
        self.app = QApplication(sys.argv)
        # self._set_stylesheet()

        self._init_mainwidget()
        self.find = self.mainwidget.findChild


        self.find(QLineEdit, 'image_directory').setText(
            '/media/zach/UE/ML/test_data/diffusion/temp_celeba_raw/celeba/train')
        
        self.find(QLineEdit, 'config_file').setText(
            '/media/zach/UE/ML/NectarGAN/nectargan/annotations/creator/celeba_config.json')
        
        self.find(QLineEdit, 'output_directory').setText(
            '/media/zach/UE/ML/NectarGAN/nectargan/annotations/creator')


        self._init_callbacks()

        self.mainwidget.show()

        sys.exit(self.app.exec())

if __name__ == "__main__":
    interface = Interface()
    interface.run()

