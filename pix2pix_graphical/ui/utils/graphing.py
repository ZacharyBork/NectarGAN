from PySide6.QtWidgets import QWidget, QVBoxLayout
from PySide6.QtCore import Qt
import pyqtgraph as pg

class Graph(QWidget):
    def __init__(
            self, 
            title: str,
            left_label: tuple[str] | None=None, # (label, units)
            right_label: tuple[str] | None=None,
            bottom_label: tuple[str] | None=None,
            top_label: tuple[str] | None=None,
            line_color: tuple[int]=(255, 255, 255),
            line_width: int=1,
            show_grid:bool=True,
            grid_alpha: float=0.3 
        ) -> None:
        super().__init__()

        self.x_data, self.y_data = [], [] # Store X, Y graph data
        self.ymax = 1.0  # Default Y max
        self.xmax = 10.0 # Default X max

        self.setWindowTitle(title)
        self.init_plot()
        self.reframe_graph()
        if show_grid: self.plot_widget.showGrid(True, True, alpha=grid_alpha)
        self.build_labels(left_label, right_label, bottom_label, top_label)
        self.init_line(line_color, line_width)
        
    def init_plot(self) -> None:
        layout = QVBoxLayout(self)
        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')
        self.plot_widget = pg.PlotWidget()
        layout.addWidget(self.plot_widget)

    def build_labels(
            self,
            left_label: tuple[str] | None,
            right_label: tuple[str] | None,
            bottom_label: tuple[str] | None,
            top_label: tuple[str] | None,
        ) -> None:
        if not left_label is None:
            self.plot_widget.setLabel(
                'left', left_label[0], units=left_label[1])
        if not right_label is None:
            self.plot_widget.setLabel(
                'right', right_label[0], units=right_label[1])
        if not bottom_label is None:
            self.plot_widget.setLabel(
                'bottom', bottom_label[0], units=bottom_label[1])
        if not top_label is None:
            self.plot_widget.setLabel(
                'top', top_label[0], units=top_label[1])

    def init_line(self, line_color: tuple[int], line_width: int) -> None:
        pen = pg.mkPen(
            color=line_color, 
            width=line_width, 
            style=Qt.PenStyle.SolidLine)
        self.plot_data_item = self.plot_widget.plot([], [], pen=pen) 

    def update_graph(self, x: float, y: float) -> None:
        self.x_data.append(x)
        if x > self.xmax:
            self.x_max = x
            self.plot_widget.setXRange(0.0, self.x_max)
        self.y_data.append(y)
        if y > self.ymax:
            self.ymax = y
            self.plot_widget.setYRange(0.0, self.ymax)
        self.plot_data_item.setData(self.x_data, self.y_data)

    def reset_graph(self) -> None:
        self.x_data.clear()
        self.y_data.clear()
        self.plot_data_item.setData(self.x_data, self.y_data)
        self.reframe_graph()

    def reframe_graph(self, min_x: float=10.0, min_y: float=1.0) -> None:
        if len(self.x_data) == 0: self.xmax = min_x
        else: self.xmax = max(min_x, list(sorted(self.x_data))[-1])
        self.plot_widget.setXRange(1.0, self.xmax)
        
        if len(self.y_data) == 0: self.ymax = min_y
        else: self.ymax = max(min_y, list(sorted(self.y_data))[-1])
        self.plot_widget.setYRange(0.0, self.ymax)