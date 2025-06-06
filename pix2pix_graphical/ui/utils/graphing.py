from typing import Literal, Any

from PySide6.QtWidgets import QWidget, QVBoxLayout
from PySide6.QtCore import Qt
import pyqtgraph as pg

class Graph(QWidget):
    def __init__(
            self, 
            window_title: str,
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
        self.steps: list[float]=[]
        
        self.xmax = xmax # Default X max
        self.ymax = ymax # Default Y max
        

        self.setWindowTitle(window_title)
        self._init_plot(bg_color, fg_color, show_grid, grid_alpha)
        self.reframe_graph()
   
    def _init_plot(
            self, 
            bg_color: tuple[float], 
            fg_color: tuple[float],
            show_grid: tuple[bool],
            grid_alpha: float
        ) -> None:
        layout = QVBoxLayout(self)
        pg.setConfigOption('background', bg_color)
        pg.setConfigOption('foreground', fg_color)
        self.graph = pg.PlotWidget()
        self.graph.showGrid(show_grid[0], show_grid[1], alpha=grid_alpha)
        layout.addWidget(self.graph)

    def add_label(
            self, 
            label: str, 
            units: str='', 
            location: Literal['left', 'right', 'top', 'bottom']='left'
        ) -> None:
        try: self.graph.setLabel(location, label, units=units)
        except Exception as e:
            raise RuntimeError('Unable to set plot label.') from e

    def add_line(
            self, 
            name: str, 
            width: int=1, 
            color: tuple[int]=(255, 255, 255)
        ) -> None:
        if name in self.lines.keys():
            msg = (f'Invalid line name: {name}. '
                   f'A line with this name already exists')
            raise KeyError(msg)
        pen = pg.mkPen(color=color, width=width, style=Qt.PenStyle.SolidLine)
        plot = self.graph.plot([], [], pen=pen)
        self.lines[name] = { 'values': [], 'plot': plot }

    def set_step(self, step: float) -> None:
        self.steps.append(step)
        if step > self.xmax:
            self.xmax = step
            self.graph.setXRange(1.0, self.xmax)

    def update_plot(self, name: str, value: str) -> None:
        line = self.lines[name]
        line['values'].append(value)
        if value > self.ymax:
            self.ymax = value
            self.graph.setYRange(0.0, self.ymax)
        line['plot'].setData(self.steps, line['values'])

    def reset_graph(self) -> None:
        self.steps = []
        for line in self.lines.values(): 
            line['values'] = []
            line['plot'].setData([], [])
        self.reframe_graph()

    def reframe_graph(self, min_x: float=10.0, min_y: float=1.0) -> None:
        if len(self.steps) == 0: self.xmax = min_x
        else: self.xmax = max(min_x, self.steps[-1])
        self.graph.setXRange(1.0, self.xmax)

        _ymax = 0.0
        for line in self.lines.values():
            if len(line['values']) > 0:
                _ymax = max(_ymax, list(sorted(line['values']))[-1])
        self.ymax = max(min_y, _ymax)
        self.graph.setYRange(0.0, self.ymax)
