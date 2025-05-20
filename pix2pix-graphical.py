import sys
import pathlib
import PySide6.QtWidgets as QtWidgets
from PySide6.QtUiTools import QUiLoader
from PySide6.QtCore import QFile
from typing import Union

def update_spinbox_from_slider(spinbox: Union[QtWidgets.QSpinBox, QtWidgets.QDoubleSpinBox], slider: QtWidgets.QSlider, multiplier: float=1.0):
    spinbox.setValue(float(slider.value()) * multiplier)

def link_sliders_to_spinboxes(widget: QtWidgets.QWidget):
    sliders_to_link = [
        # (slider, spinbox, multiplier)
        ('lr_slider', 'learning_rate', 0.0001),
        ('num_epochs_slider', 'num_epochs', 1.0),
        ('batch_size_slider', 'batch_size', 1.0),
        ('worker_count_slider', 'worker_count', 1.0),
        ('beta1_slider', 'beta1', 0.01),
        ('l1_slider', 'l1', 1.0),
        ('sobel_slider', 'sobel', 1.0)
    ]

    for x, y, z in sliders_to_link:
        slider = widget.findChild(QtWidgets.QSlider, x)
        spinbox = widget.findChild(QtWidgets.QSpinBox, y)
        if spinbox == None: spinbox = widget.findChild(QtWidgets.QDoubleSpinBox, y)
        slider.valueChanged.connect(
            lambda value, s=spinbox, sl=slider, m=z: update_spinbox_from_slider(s, sl, multiplier=m))

def init_ui(widget: QtWidgets.QWidget):
    link_sliders_to_spinboxes(widget)

    coninue_train = widget.findChild(QtWidgets.QCheckBox, 'continue_train')
    load_epoch = widget.findChild(QtWidgets.QSpinBox, 'load_epoch')

    

def main():
    app = QtWidgets.QApplication(sys.argv)

    loader = QUiLoader()
    
    root = pathlib.Path(__file__).parent
    ui_file = pathlib.Path(root, 'ui', 'base.ui')
    file = QFile(ui_file.resolve().as_posix())
    file.open(QFile.ReadOnly)
    
    widget = loader.load(file)
    file.close()
    init_ui(widget)

    widget.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
