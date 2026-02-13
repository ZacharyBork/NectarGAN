from pathlib import Path
from typing import Any, Literal

from PySide6.QtWidgets import (
    QSizePolicy, QWidget, QLabel, QFrame, QHBoxLayout, QVBoxLayout, 
    QPushButton, QRadioButton)
from PySide6.QtCore import Qt, QRect, QPoint, QEvent, QTimer
from PySide6.QtGui import QPainter, QPen, QColor, QPixmap

class ImageLabel(QLabel):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.start_pt = None
        self.end_pt = None
        self.drawing = False
        self.boxes: dict[str, list] = {}
        self.current_image: Path = None
        self.parent_widget = parent
        
        self.mode: Literal['Draw', 'Erase'] = 'Draw'
        self.current_color = 'red'
        self.colors = {
            'red':   QColor(255, 0, 0),
            'green': QColor(0, 255, 0),
            'blue':  QColor(0, 0, 255)}
        
    def _get_color(self) -> QColor:
        return self.colors[self.current_color]
        
    def mousePressEvent(self, event: QEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.start_pt = event.pos()
            self.end_pt = event.pos()
            self.drawing = True
            self.update()
    
    def mouseMoveEvent(self, event: QEvent) -> None:
        if self.drawing:
            self.end_pt = event.pos()
            self.update()
    
    def mouseReleaseEvent(self, event: QEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self.drawing:
            self.end_pt = event.pos()
            self.drawing = False
            
            pixmap = self.pixmap()
            if pixmap and not pixmap.isNull():
                bounds = self._get_pixmap_rect()
                
                x1 = (self.start_pt.x() - bounds.x()) / bounds.width()
                y1 = (self.start_pt.y() - bounds.y()) / bounds.height()
                x2 = (self.end_pt.x() - bounds.x()) / bounds.width()
                y2 = (self.end_pt.y() - bounds.y()) / bounds.height()
                
                x1, y1 = max(0, min(1, x1)), max(0, min(1, y1))
                x2, y2 = max(0, min(1, x2)), max(0, min(1, y2))
                
                bbox = (x1, y1, x2, y2)
                
                if self.mode == 'Draw':
                    try: self.boxes[self.current_color].append(bbox)
                    except KeyError: self.boxes[self.current_color] = [bbox]
                elif self.mode == 'Erase': self._erase_overlapping_boxes(bbox)
            
            self.update()
            
    def _get_pixmap_rect(self):
        pixmap = self.pixmap()
        if not pixmap or pixmap.isNull(): return QRect()
        
        w = pixmap.width()
        h = pixmap.height()
        x = (self.width() - w) // 2
        y = (self.height() - h) // 2
        
        return QRect(x, y, w, h)
    
    def _erase_overlapping_boxes(self, erase_box: tuple) -> None:
        ex1, ey1, ex2, ey2 = erase_box
        ex1, ex2 = min(ex1, ex2), max(ex1, ex2)
        ey1, ey2 = min(ey1, ey2), max(ey1, ey2)
        
        colors_to_remove = []
        for color_key, boxes in self.boxes.items():
            boxes_to_keep = []
            
            for box in boxes:
                bx1, by1, bx2, by2 = box
                bx1, bx2 = min(bx1, bx2), max(bx1, bx2)
                by1, by2 = min(by1, by2), max(by1, by2)
                
                corners = [(bx1, by1), (bx2, by1), (bx1, by2), (bx2, by2)]
                corner_inside = False
                for cx, cy in corners:
                    if ex1 <= cx <= ex2 and ey1 <= cy <= ey2:
                        corner_inside = True
                        break
                
                if not corner_inside: boxes_to_keep.append(box)
            
            if boxes_to_keep: self.boxes[color_key] = boxes_to_keep
            else: colors_to_remove.append(color_key)
    
        for color_key in colors_to_remove: del self.boxes[color_key]
    
    def paintEvent(self, event: QEvent) -> None:
        super().paintEvent(event)
        
        pixmap = self.pixmap()
        if not pixmap or pixmap.isNull(): return
        
        painter = QPainter(self)
        bounds = self._get_pixmap_rect()
        
        for key, value in self.boxes.items():
            for box in value:
                color = self.colors[key]
                pen = QPen(color, 2, Qt.PenStyle.SolidLine)
                painter.setPen(pen)
        
                x1, y1, x2, y2 = box
                p_x1 = bounds.x() + x1 * bounds.width()
                p_y1 = bounds.y() + y1 * bounds.height()
                p_x2 = bounds.x() + x2 * bounds.width()
                p_y2 = bounds.y() + y2 * bounds.height()
                
                box = QRect(
                    QPoint(int(p_x1), int(p_y1)), 
                    QPoint(int(p_x2), int(p_y2)))
                painter.drawRect(box.normalized())
            
        if self.drawing and self.start_pt and self.end_pt:
            if self.mode == 'Draw': color = self._get_color()
            else: color = QColor(255, 255, 255)
            pen = QPen(color, 2, Qt.PenStyle.SolidLine)
            painter.setPen(pen)
            current_box = QRect(self.start_pt, self.end_pt).normalized()
            painter.drawRect(current_box)
        
        painter.end()
            
    def draw_image(
            self, 
            image_path: Path, 
            minimum_size: int = 400
        ) -> None:
        pixmap = QPixmap(image_path)
        
        self.setMinimumSize(minimum_size, minimum_size)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding)
                
        pixmap = pixmap.scaled(
            self.parent_widget.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation)
        
        self.setPixmap(pixmap)
    
    def clear_boxes(self) -> None:
        self.boxes = {}
        self.start_pt = None
        self.end_pt = None
        self.drawing = False
        self.update()
        
class InteractiveImageDisplay(QFrame):
    def __init__(self, minimum_size: int = 400, *args, **kwargs) -> None:
        super(InteractiveImageDisplay, self).__init__(*args, **kwargs)
        self.minimum_size = minimum_size
        self.landmark_ids = { 'Red': 'red', 'Green': 'green', 'Blue': 'blue' }
        self.build_interface()
        
    def set_landmark_ids(self, red: str, green: str, blue: str) -> None:
        self.red_button.setText(red)
        self.green_button.setText(green)
        self.blue_button.setText(blue)
        self.landmark_ids = { red: 'red', green: 'green', blue: 'blue' }
        
    def _set_mode(self) -> None:
        match self.mode_button.text():
            case 'Draw': self.mode_button.setText('Erase')
            case 'Erase': self.mode_button.setText('Draw')
        self.image_label.mode = self.mode_button.text()
                
    def _set_color(self) -> None:
        if self.mode_button.text() == 'Erase': self._set_mode()
        for i in range(self.colors_layout.count()): 
            button: QRadioButton = self.colors_layout.itemAt(i).widget()
            if button.isChecked(): color = button.text()
        self.image_label.current_color = self.landmark_ids[color]
        
    def _build_button_frame(self) -> None:
        button_frame = QFrame()
        self.button_layout = QVBoxLayout()
        self.button_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        button_frame.setLayout(self.button_layout)
        
        self.mode_button = QPushButton(text='Draw')
        self.mode_button.clicked.connect(self._set_mode)
        self.button_layout.addWidget(self.mode_button)
        
        colors_frame = QFrame()
        self.colors_layout = QVBoxLayout()
        self.red_button = QRadioButton(text='Red')
        self.red_button.setStyleSheet('background-color: red;')
        self.red_button.setChecked(True)
        
        self.green_button = QRadioButton(text='Green')
        self.green_button.setStyleSheet('background-color: green;')
        
        self.blue_button = QRadioButton(text='Blue')
        self.blue_button.setStyleSheet('background-color: blue;')
        
        for x in [self.red_button, self.green_button, self.blue_button]:
            x.clicked.connect(self._set_color)
            self.colors_layout.addWidget(x)
        colors_frame.setLayout(self.colors_layout)
        self.button_layout.addWidget(colors_frame)
        
        self.clear_button = QPushButton(text='Clear')
        self.clear_button.clicked.connect(self.image_label.clear_boxes)
        self.button_layout.addWidget(self.clear_button)
        
        self.base_layout.addWidget(button_frame)
        
    def build_interface(self) -> None:
        self.setMinimumSize(self.minimum_size, self.minimum_size)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding)
        
        self.base_layout = QHBoxLayout()
        self.image_label = ImageLabel(parent=self)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.base_layout.addWidget(self.image_label)
        
        self._build_button_frame()
        
        self.setLayout(self.base_layout)
        
    def draw_image(self, image_path: Path) -> None: 
        draw = self.image_label.draw_image
        QTimer.singleShot(0, lambda : draw(image_path, self.minimum_size))
        
    def get_landmark_data(self) -> dict[str, Any]:
        boxes = self.image_label.boxes
        output = {}
        for key, value in self.landmark_ids.items():
            try: output[key] = boxes[value]
            except KeyError: pass
        return output
        
    def reset(self) -> None:
        if self.mode_button.text() == 'Erase': self._set_mode()
        self.red_button.setChecked(True)
        self.image_label.clear_boxes()



