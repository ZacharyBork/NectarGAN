from pathlib import Path
from os import PathLike
from PIL import Image
from importlib.resources import files

from PySide6.QtWidgets import QPushButton, QLineEdit, QFileDialog
from PySide6.QtGui import QIcon

def browse_directory(lineedit: QLineEdit) -> None:
    directory = Path(QFileDialog.getExistingDirectory())
    if directory is None or directory.as_posix() == '.': return
    lineedit.setText(directory.as_posix())

def get_icon_file(filename: str) -> PathLike:
    try: 
        _root = 'pix2pix_graphical.toolbox.resources.icons'
        _path = files(_root).joinpath(filename)
        icon_path = Path(_path)
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

def concat_images(args: tuple[Path, Path, Path, str]) -> None:
        '''Concatenates two images and exports the result.
        
        This is used by `toolbox.workers.utility_worker.UtilityWorker` to pair
        images together for the creation of pix2pix datasets.

        Args:
            args : A tuple containing:
                Path : The input A file for the pairing operation.
                Path : The directory contining the B image files.
                Path : The directory to output the paired images to.
                str  : The pairing direction (i.e. 'AtoB, 'BtoA').
        '''
        fileA, dirB, outdir, direction = args
        fileB = Path(dirB, fileA.name)
        if not fileB.exists(): return

        try:
            imgA = Image.open(fileA).convert('RGB')
            imgB = Image.open(fileB).convert('RGB')

            if (imgA.height != imgB.height or imgA.width != imgB.width):
                message = (
                    f'Unable to pair images. '
                    f'Image A and image B must have the same resolution.')
                raise RuntimeError(message)

            new_width = imgA.width + imgB.width
            concat = Image.new('RGB', (new_width, imgA.height))
            if direction == 'AtoB': x, y = imgA, imgB
            else: x, y = imgB, imgA
            concat.paste(x, (0, 0))
            concat.paste(y, (x.width, 0))

            output_path = Path(outdir, fileA.name)
            concat.save(output_path)
            return None

        except Exception as e:
            message = f'Unable to concatenate image: {fileA.as_posix()}'
            raise RuntimeError(message) from e