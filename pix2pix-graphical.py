import sys
import pathlib
import PySide6.QtWidgets as QtWidgets
from PySide6.QtUiTools import QUiLoader
from PySide6.QtCore import QFile

def update_spinbox_from_slider(spinbox: QtWidgets.QSpinBox, slider: QtWidgets.QSlider, multiplier: float=1.0):
    spinbox.setValue(float(slider.value()) * multiplier)

def connect_ui_elements(widget: QtWidgets.QWidget):
    # Connect learning rate slide to spinbox
    lr_slider = widget.findChild(QtWidgets.QSlider, 'lr_slider')
    learning_rate = widget.findChild(QtWidgets.QDoubleSpinBox, 'learning_rate')
    lr_slider.valueChanged.connect(lambda: update_spinbox_from_slider(learning_rate, lr_slider, multiplier=0.0001))

def main():
    app = QtWidgets.QApplication(sys.argv)

    loader = QUiLoader()
    
    root = pathlib.Path(__file__).parent
    ui_file = pathlib.Path(root, 'ui', 'base.ui')
    file = QFile(ui_file.resolve().as_posix())
    file.open(QFile.ReadOnly)
    
    widget = loader.load(file)
    file.close()
    connect_ui_elements(widget)

    widget.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
