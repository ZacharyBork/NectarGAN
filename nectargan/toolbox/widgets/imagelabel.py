from PySide6.QtGui import QPixmap
from PySide6.QtCore import Qt, QEvent
from PySide6.QtWidgets import QWidget, QLabel, QSizePolicy

class ImageLabel(QLabel):
    def __init__(
            self, 
            name: str,
            pixmap: QPixmap | None=None,
            parent: QWidget=None
        ) -> None:
        super().__init__(parent)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding)
        
        self._pixmap: QPixmap = pixmap
        if not pixmap is None: self._rescale()
        self._name: str = name
        
    def set_pixmap(self, pixmap: QPixmap) -> None:
        self._pixmap = pixmap
        self._rescale()

    def resizeEvent(self, event: QEvent) -> None:
        self._rescale()
        super().resizeEvent(event)

    def _rescale(self) -> None:
        if self._pixmap:
            scaled = self._pixmap.scaled(
                self.size(), 
                Qt.AspectRatioMode.KeepAspectRatio, 
                Qt.TransformationMode.SmoothTransformation)
            self.setPixmap(scaled)