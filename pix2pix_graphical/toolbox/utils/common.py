import pathlib
from os import PathLike
from importlib.resources import files

from PySide6.QtWidgets import QPushButton, QLineEdit, QFileDialog
from PySide6.QtGui import QIcon

def browse_directory(lineedit: QLineEdit) -> None:
    directory = pathlib.Path(QFileDialog.getExistingDirectory())
    if directory is None or directory.as_posix() == '.': return
    lineedit.setText(directory.as_posix())

def get_icon_file(filename: str) -> PathLike:
    try: 
        _root = 'pix2pix_graphical.toolbox.resources.icons'
        _path = files(_root).joinpath(filename)
        icon_path = pathlib.Path(_path)
        if not icon_path.exists():
            message = f'Unable to locate icon file: {icon_path.as_posix()}'
            raise FileNotFoundError(message)
        return icon_path
    except Exception as e:
        message = f'Unable to locate icon file: {icon_path.as_posix()}'
        raise RuntimeError(message) from e

def set_button_icon(filename: str, button: QPushButton) -> None:
    file = get_icon_file(filename)
    icon = QIcon(file.as_posix())
    button.setIcon(icon)