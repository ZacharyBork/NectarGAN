from typing import Literal, Any

import pyqtgraph as pg
from PySide6.QtGui import QColor
from PySide6.QtCore import Qt, QEvent
from PySide6.QtWidgets import (  
    QWidget, QFrame, QVBoxLayout, QHBoxLayout, QCheckBox, QLabel)

class Graph(QWidget):
    def __init__(
            self, 
            window_title: str,
            bottom_label: str='',
            bg_color: tuple[float]=(50, 50, 50),
            fg_color: tuple[float]=(0, 0, 0),
            show_grid: tuple[bool]=(True, True),
            grid_alpha: float=0.3,
            xmax: float=10.0,
            ymax: float=1.0
        ) -> None:
        super().__init__()
        # Stores data for lines belonging to the graph
        self.lines: dict[str, dict[str, Any]] = {}
        
        self.xmax = xmax # Default X max
        self.ymax = ymax # Default Y max
        
        self.setWindowTitle(window_title)
        self._init_base_layout()
        self._build_layout()
        self._init_graph(
            bg_color, fg_color, show_grid, grid_alpha, bottom_label)
        self.reframe_graph()

        self._set_toggle_visibility(visible=False)
        self._init_hover_event()

    def _init_base_layout(self) -> None:
        self.base_layout = QHBoxLayout(self)
        self.base_layout.setContentsMargins(0, 0, 0, 0)
        
        self.container = QHBoxLayout()
        self.container.setSpacing(0)
        self.container.setContentsMargins(0, 0, 0, 0)
        
        self.base_frame = QFrame()
        self.base_frame.setLayout(self.container)
        self.base_layout.addWidget(self.base_frame)

    def _init_hover_event(self) -> None:
        self.base_frame.installEventFilter(self)

    def _set_toggle_visibility(self, visible: bool) -> None:
        self.toggle_box.setVisible(visible)

    def eventFilter(self, watched: QWidget, event: QEvent) -> bool:
        if watched == getattr(self, 'base_frame', None):
            if event.type() == QEvent.Type.Enter:
                self._set_toggle_visibility(visible=True)
            elif event.type() == QEvent.Type.Leave:
                self._set_toggle_visibility(visible=False)
        return super().eventFilter(watched, event)

    def _build_layout(self) -> None:
        self.graph_box = QFrame()
        self.graph_layout = QVBoxLayout()
        self.graph_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.graph_layout.setSpacing(4)
        self.graph_box.setLayout(self.graph_layout)
        self.graph_box.setContentsMargins(0, 0, 0, 0)
        
        self.toggle_box = QFrame()
        self.toggle_box.setFrameShape(QFrame.Shape.Panel)
        self.toggle_box.setFrameShadow(QFrame.Shadow.Sunken)
        self.toggle_layout = QVBoxLayout()
        self.toggle_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.toggle_layout.setSpacing(2)
        self.toggle_box.setLayout(self.toggle_layout)
        self.toggle_box.setContentsMargins(2, 2, 2, 2)

        self.container.addWidget(self.toggle_box)
        self.container.addWidget(self.graph_box)
   
    def _init_graph(
            self, 
            bg_color: tuple[float], 
            fg_color: tuple[float],
            show_grid: tuple[bool],
            grid_alpha: float,
            bottom_label: str
        ) -> None:
        pg.setConfigOption('background', bg_color)
        pg.setConfigOption('foreground', fg_color)
        self.graph = pg.PlotWidget()
        self.graph.showGrid(show_grid[0], show_grid[1], alpha=grid_alpha)
        self.graph_layout.addWidget(self.graph)
        if not bottom_label == '':
            _bottom_label = QLabel(bottom_label)
            _bottom_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
            _bottom_label.setStyleSheet('QLabel { font-size: 12px; }')
            self.graph_layout.addWidget(_bottom_label)

    def add_label(
            self, 
            label: str, 
            units: str='', 
            location: Literal['left', 'right', 'top', 'bottom']='left'
        ) -> None:
        try: self.graph.setLabel(location, label, units=units)
        except Exception as e:
            raise RuntimeError('Unable to set plot label.') from e
        
    def _add_line_vis_checkbox(self, name: str, color: tuple[int]) -> None:
        toggle = QCheckBox(name) # Build vis toggle checkbox
        text_color = QColor(color[0], color[1], color[2])
        toggle.setStyleSheet(f'QCheckBox {{ color: {text_color.name()}; }}')

        toggle.setChecked(True)  # Start checked, all lines visible
        toggle.clicked.connect(  # Connect to line vis function
            lambda value, i=name : self.set_line_visibility(i, value))
        self.toggle_layout.addWidget(toggle) # Add to toggles layout

        return toggle

    def add_line(
            self, 
            name: str, 
            width: int=1, 
            color: tuple[int]=(255, 255, 255)
        ) -> None:
        # Validate name
        if name in self.lines.keys():
            msg = (f'Invalid line name: {name}. '
                   f'A line with this name already exists')
            raise KeyError(msg)
        # Build plot
        pen = pg.mkPen(color=color, width=width, style=Qt.PenStyle.SolidLine)
        plot = self.graph.plot([], [], pen=pen)
        toggle = self._add_line_vis_checkbox(name=name, color=color)
        self.lines[name] = { # Store line data
            'values': [], 'steps':[], 'plot': plot,
            'visible': True, 'toggle': toggle }

    def set_line_visibility(
            self, 
            name: str, 
            visible: bool=True
        ) -> None:
        try: self.lines[name]['visible'] = visible        
        except Exception as e:
            raise KeyError(f'Invalid line name: {name}') from e
        self.update_graph()
        
    def set_exclusive_line_visibility(self, name: str) -> None:
        try: 
            for _name, line in self.lines.items():
                if name == _name: line['visible'] = True 
                else: line['visible'] = False 
        except Exception as e:
            raise KeyError(f'Invalid line name: {name}') from e
        self.update_graph()

    def update_plot(
            self, 
            name: str, 
            value: float | None=None,
            step: float | None=None
        ) -> None:
        if (value is None) != (step is None):
            raise ValueError('`value` and `step` must both be float or None.')

        line = self.lines[name]
        if not value is None: 
            line['values'].append(value)
            if value > self.ymax and line['visible']:
                self.ymax = value
                self.graph.setYRange(0.0, self.ymax)
        if not step is None: 
            line['steps'].append(step)
            if step > self.xmax and line['visible']:
                self.xmax = step
                self.graph.setXRange(1.0, self.xmax)
        
        if line['visible']: line['plot'].setData(line['steps'], line['values'])
        else: line['plot'].setData([], [])

    def update_graph(self) -> None:
        if not len(self.lines) == 0:
            for line in self.lines:
                self.update_plot(line, value=None, step=None)
        self.reframe_graph()

    def reset_graph(self) -> None:
        for line in self.lines.values(): 
            line['values'].clear()
            line['steps'].clear()
        self.update_graph()
        self.reframe_graph()

    def get_max_values(self) -> tuple[float]:
        _xmax = _ymax = 0.0
        for line in self.lines.values():
            if not line['visible']: continue
            if len(line['steps']) > 0:
                _xmax = max(_xmax, list(sorted(line['steps']))[-1])
            if len(line['values']) > 0:
                _ymax = max(_ymax, list(sorted(line['values']))[-1])
        return _xmax, _ymax

    def reframe_graph(self, min_x: float=10.0, min_y: float=1.0) -> None:
        _xmax, _ymax = self.get_max_values()

        self.xmax = max(min_x, _xmax)
        self.graph.setXRange(1.0, self.xmax)
        
        self.ymax = max(min_y, _ymax)        
        self.graph.setYRange(0.0, self.ymax)
        
