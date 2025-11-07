# Highly configurable helper script for graphing loss log data.
# Run `python scripts/build_loss_graphs.py --help` from the repository
# root for info about configuring graph settings and functionality.

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np

import argparse

class GraphBuilder():
    def __init__(self) -> None:
        self.sample_frequency = 50 # Graph data sample frequency
        self.graph_width: float = 10.0
        self.graph_height: float = 6.0
        self.output_dpi: int = 300
        self.line_width: float = 1.0
        self.one_graph_per_loss = False
        self.use_log_scaling = False
        self.split_g_and_d = False
        self.graph_weights = False
        self.preview = False # If False, save graphs. If True, display graphs

        self.experiment_directory: Path = None
        self.log_data: dict[str, dict[str, Any]] = None

        # List of data to plot, after sampling has occured
        # list[(Loss Name, Loss Values, Loss Weights)]
        self.graph_data: list[tuple[str, list[float], list[float]]] = None

        self.fig = None

    def parse_arguments(self) -> argparse.Namespace:
        parser = argparse.ArgumentParser()
        parser.add_argument(
            '-e', '--experiment_directory', type=str, 
            help=(
                f'Path to the experiment directory you\'d like to graph the '
                f'losses for.'))
        parser.add_argument(
            '-f', '--sample_frequency', type=int, default=50,
            help=(
                f'The frequency at which to sample the value and weight list '
                f'for each loss. Lower values will create a more detailed '
                f'graph, but will take longer to calculate.'))
        parser.add_argument(
            '-W', '--graph_width', type=float, default=14.0,
            help=(f'The graph width (in inches).'))
        parser.add_argument(
            '-H', '--graph_height', type=float, default=7.0,
            help=(f'The graph height (in inches).'))
        parser.add_argument(
            '-dpi', '--output_dpi', type=int, default=300,
            help=(f'The DPI of the exported graph images.'))
        parser.add_argument(
            '-lw', '--line_width', type=float, default=1.0,
            help=(f'The thickness of the graphed lines.'))
        parser.add_argument(
            '-og', '--one_graph_per_loss', action='store_true',
            help=(
                f'If this flag is provided, the GraphBuilder will generate '
                f'a one graph per loss found in the log. If this flag is NOT '
                f'provided, it will instead generate one combined graph of '
                f'all of the losses.'))
        parser.add_argument(
            '-ls', '--use_log_scaling', action='store_true',
            help=(
                f'If this flag is provided, the rendered graphs will use '
                f'logorithmic scaling on their Y axis. If not provided, they '
                f'will instead use linear scaling.'))
        parser.add_argument(
            '-s', '--split_g_and_d', action='store_true',
            help=(
                f'If this flag is provided, the generator and discrimiator '
                f'losses will be split and rendered on separate graphs.\n'
                f'NOTE: This assumes the first letter of each loss name is '
                f'either "G" or "D" (i.e. "G_GAN", "D_real"). This does '
                f'nothing if the "-og" flag is provided.'))
        parser.add_argument(
            '-w', '--graph_weights', action='store_true',
            help=(
                f'If present, graph loss weight values. Otherwise, only '
                f'graph the actual loss values.'))
        parser.add_argument(
            '-p', '--preview', action='store_true',
            help=(
                f'The frequency at which to sample the value and weight list '
                f'for each loss. Lower values will create a more detailed '
                f'graph, but will take longer to calculate.'))
        
        return parser.parse_args()

    def init_from_args(self) -> None:
        args = self.parse_arguments()

        self.get_experiment_directory(args.experiment_directory)
        self.get_log_data()
        self.sample_frequency = args.sample_frequency
        self.graph_width = args.graph_width
        self.graph_height = args.graph_height
        self.output_dpi = args.output_dpi
        self.line_width = args.line_width
        self.one_graph_per_loss = args.one_graph_per_loss
        self.use_log_scaling = args.use_log_scaling
        self.split_g_and_d = args.split_g_and_d
        self.graph_weights = args.graph_weights
        self.preview = args.preview

    def get_experiment_directory(self, input_dir: str | None) -> None:
        if input_dir is None:
            raise TypeError(
                f'Experiment directory cannot be Nonetype. '
                f'Please pass a valid experiment directory with the "-e" flag.')
        experiment_directory = Path(input_dir)
        if not experiment_directory.exists():
            raise FileNotFoundError(
                f'Unable to locate experiment directory at path: '
                f'{experiment_directory.as_posix()}')
        self.experiment_directory = experiment_directory

    def get_log_data(self) -> None:
        loss_log = Path(self.experiment_directory, 'loss_log.json')
        if not loss_log.exists():
            raise FileNotFoundError(
                f'Unable to locate loss log at path: {loss_log.as_posix()}')
        with open(loss_log, 'r') as file:
            log_data = json.loads(file.read())
        self.log_data = log_data

    def sample_log(
            self, 
            loss_functions: dict[str, dict[str, float]]
        ) -> None:
        output = []
        for key, value in loss_functions.items():
            if len(value['loss']) == 0: continue
            loss_values = [
                i for j, i in enumerate(value['loss']) 
                if j%self.sample_frequency == 0]
            weight_values =  [
                i for j, i in enumerate(value['weights']) 
                if j%self.sample_frequency == 0]
            output.append((key, loss_values, weight_values))
        self.graph_data = output

    def add_supertitle(self) -> None:
        device = self.log_data['LOSSMANAGER_LOG']['device']
        dataroot = self.log_data['LOSSMANAGER_LOG']['dataroot']
        dataset_length = self.log_data['LOSSMANAGER_LOG']['dataset_length']
        experiment = self.log_data['LOSSMANAGER_LOG']['experiment']

        self.fig.suptitle(
            f'Experiment: {experiment} | '
            f'Device: {device} | '
            f'Dataset Length: {dataset_length} | '
            f'Dataroot: {dataroot}',
            fontsize=8)

    def show_or_export_graph(self, graph_name: str) -> None:
        self.add_supertitle()

        if self.preview: 
            plt.show(block=True)
            return
               
        output_directory = Path(self.experiment_directory, 'graphs')
        if not output_directory.exists():
            output_directory.mkdir()
        else: 
            raise FileExistsError(
                f'Graphs directory already exists at path: '
                f'{output_directory.as_posix()}\n\n'
                f'Please delete it to continue.')
        output_path = Path(output_directory, f'{graph_name}.png')
        self.fig.savefig(output_path, dpi=self.output_dpi, bbox_inches='tight')

    def _render_graph_core(
            self, 
            axes: Axes,
            loss_name: str,
            loss_values: list[float],
            loss_weights: list[float]
        ) -> None:
        axes.plot(
            loss_values, label=f'{loss_name} Loss', 
            linewidth=self.line_width)
        if self.graph_weights:
            axes.plot(
                loss_weights, label=f'{loss_name} Weight', 
                linewidth=self.line_width, alpha=0.7)
        axes.set_xlabel('Epoch')
        axes.set_ylabel('Value')
        if self.use_log_scaling: axes.set_yscale('log')
        axes.legend(loc='best')

    def render_single_graph(self) -> None:
        if self.split_g_and_d: 
            self.fig, (ax1, ax2) = plt.subplots(
                2, 1, figsize=(self.graph_width, self.graph_height), 
                sharex=True)
            ax1.set_title('Generator Loss', fontsize=12)
            ax2.set_title('Discriminator Loss', fontsize=12)
        else: 
            self.fig, ax1 = plt.subplots(
                figsize=(self.graph_width, self.graph_height))
            ax1.set_title('Loss (Combined)', fontsize=12)
        for loss in self.graph_data:
            loss_name, loss_values, loss_weights = loss
            if self.split_g_and_d:
                axes = ax1 if loss_name[0] == 'G' else ax2
            else: axes = ax1
            self._render_graph_core(axes, loss_name, loss_values, loss_weights)
        self.show_or_export_graph(graph_name='loss_combined')

    def render_individual_graphs(self) -> None:
        for loss in self.graph_data:
            self.fig, ax1 = plt.subplots(
                figsize=(self.graph_width, self.graph_height))
            loss_name, loss_values, loss_weights = loss
            self._render_graph_core(ax1, loss_name, loss_values, loss_weights)
            ax1.set_title(f'Loss: {loss_name}', fontsize=12)
            self.show_or_export_graph(graph_name=loss_name)

    def execute(self) -> None:
        self.init_from_args()
        self.sample_log(self.log_data['LOSSMANAGER_LOG']['loss_functions'])

        if self.one_graph_per_loss: self.render_individual_graphs()
        else: self.render_single_graph()

if __name__ == "__main__":
    builder = GraphBuilder()
    builder.execute()

