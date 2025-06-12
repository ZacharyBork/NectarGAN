from pathlib import Path
from os import PathLike
from PIL import Image
from importlib.resources import files
from typing import Literal

import cv2
import numpy as np

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

def concat_images(args: tuple[Path, Path, Path, str, int]) -> None:
    '''Concatenates two images and exports the result.
    
    This is used by `toolbox.workers.utility_worker.UtilityWorker` to pair
    images together for the creation of pix2pix datasets.

    Args:
        args : A tuple containing:
            Path : The input A file for the pairing operation.
            Path : The directory contining the B image files.
            Path : The directory to output the paired images to.
            str  : The pairing direction (i.e. 'AtoB, 'BtoA').
            int  : Resolution (squared) to scale each image to before they
                are concatenated, or -1 for no scaling. Final output 
                resolution will be: [value * 2, value]. 
    '''
    fileA, dirB, outdir, direction, scale = args
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
        if not scale == -1:
            imgA = imgA.resize((scale, scale))
            imgB = imgB.resize((scale, scale))

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
        
def get_image_value(args) -> tuple[int | float, Path]:
    image, sort_type = args
    if not image.exists():
        raise ValueError(f'Unable to find image: {image.as_posix()}')
    _image = image
    image = image.as_posix()
    value = 0.0
    match sort_type:
        case 'Count (White Pixels)':
            image = cv2.imread(image)
            white_mask = np.all(image >= 250, axis=2)
            value = int(np.sum(white_mask))
        case 'Count (Black Pixels)':
            image = cv2.imread(image)
            white_mask = np.all(image <= 20, axis=2)
            value = int(np.sum(white_mask))
        case 'Mean Pixel Value':
            image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
            mean = cv2.mean(image)
            value = int(mean[0])
        case 'Contrast (Average Local)':
            image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
            kernel = np.ones((7, 7), np.uint8)
            max = cv2.dilate(image, kernel, iterations=1)
            max = max.astype(np.float64)
            min = cv2.erode(image, kernel, iterations=1)
            min = min.astype(np.float64)
            contrast = (max - min) / (max + min + 1e-8)
            value = float(np.mean(contrast))
        case 'Contrast (RMS)':
            image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
            mean = np.mean(image)
            value = float(np.std(image))
        case 'Contrast (Haziness)':
            image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
            image = image.astype(np.float32) / 255.0
            sharpness = cv2.Laplacian(image, cv2.CV_32F, ksize=3)
            haziness = np.abs(sharpness)
            value = float(np.mean(haziness))
    return (value, _image)