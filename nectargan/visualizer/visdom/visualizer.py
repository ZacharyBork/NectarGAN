from typing import Sequence
from collections.abc import Mapping
import threading
import queue

import torch
import numpy as np
from visdom import Visdom

class VisdomVisualizer():
    def __init__(
            self, 
            env: str='main',
            server: str='http://localhost',
            port: int=8097
        ) -> None:
        self.env = env
        self.server = server
        self.port = port
        self.is_threaded: bool = False # Overridden by `self.start_thread()`

        self.init_visdom()

    ### INITIALIZATION ###

    def init_visdom(self) -> None:
        '''Initializes Visdom and validates endpoint connection.

        Also assigns Visdom client to self.vis internally to send commands.
        
        Raises:
            ConnectionError : If unable to connect to Visdom server.
        '''
        vis = Visdom(
            server=self.server, 
            port=self.port,
            env=self.env)
        if not vis.check_connection():
            message = (
                f'Unable to connect to Visdom server. '
                f'Please ensure it is running.')
            raise ConnectionError(message)
        self.vis = vis

    def clear_env(self) -> None:
        '''Closes all windows in the VisdomVisualizer's environment.'''
        self.vis.close(win=None, env=self.env)
        
    ### HELPERS ###

    def _denorm_tensor(
            self, 
            tensor: torch.Tensor, 
            clamp: bool=True
        ) -> torch.Tensor: 
        '''Denormalizes a pix2pix output tensor [-1, 1] -> [0, 1] for viewing.
        
        The final upsampling convolutional layer in the pix2pix architecture 
        uses tanh nonlinearity. This outputs a tensor normalized [-1, 1]. This 
        function just takes those output tensors and renormalizes them to the 
        [0, 1] range which is what Visdom is expecting for images.

        Args:
            tensor : The torch.Tensor object to denormalize.
            clamp : Whether to clamp the denormed values to [0, 1].

        Returns:
            torch.Tensor: The input tensor renormalized to [0, 1].
        '''
        norm = (tensor + 1) * 0.5
        return torch.clamp(norm, 0.0, 1.0) if clamp else norm

    ### UPDATE (UNTHREADED) ###

    def update_images(
            self,
            x: torch.Tensor, 
            y: torch.Tensor, 
            z: torch.Tensor, 
            title: str, 
            image_size: int=300
        ) -> None:
        if self.is_threaded: self._store_images(x, y, z, title, image_size)
        else: self._update_images_core(x, y, z, title, image_size)

    def update_loss_graphs(
            self, 
            graph_step: float,
            losses_G: Mapping[str, float], 
            losses_D: Mapping[str, float]    
        ) -> None:
        if self.is_threaded:
            self._store_graph_data(graph_step, losses_G, losses_D)
        else: 
            self._update_loss_graphs_core(graph_step, losses_G, losses_D)

    def _update_images_core(
            self,
            x: torch.Tensor, 
            y: torch.Tensor, 
            z: torch.Tensor, 
            title: str, 
            image_size: int=300
        ) -> None:
        '''Updates the Visdom [x, y, z] image grid.
        
        Takes three torch.Tensor objects as input and normalizes them [0, 1], 
        then concatenates them [x, y, z], then updates Visdom with the 
        concatenated image.

        Args:
            x : The first input tensor (left image in grid).
            y : The second input tensor (middle image in grid).
            z : The third input tensor (right image in grid).
            title : The title of the concatenated image window.
            image_size : The width and height, in pixels, to render each image. 
        '''
        composite = torch.cat([
            self._denorm_tensor(x), 
            self._denorm_tensor(y), 
            self._denorm_tensor(z)], dim=3)
        self.vis.images(
            composite, win='comparison_grid', nrow=1, padding=2,
            opts=dict(title=title, width=image_size*3, height=image_size))

    def _update_graph(
            self, 
            values: Sequence[torch.Tensor], 
            steps: Sequence[float],
            window_internal_name: str, 
            window_title: str,
            xlabel: str, 
            ylabel: str, 
            legend: list[str]
        ) -> None:
        '''Updates a visdom.Visdom.line graph to add new values at steps.

        Args:
            values : The values to add to the graph.
            steps : The steps, 1 per value, to add each value at on the graph.
            window_internal_name : Internal name of the Visdom graph window.
            window_title : The human-readable title of the Visdom graph window.
            xlabel : The X axis label of the graph.
            ylabel : The Y axis label of the graph.
            legend : List of strings, one per value, to title each graph line.
        ''' 
        self.vis.line(
            Y=np.column_stack((values)),
            X=np.column_stack((steps)),
            win=window_internal_name, update='append',
            opts=dict(
                title=window_title, 
                xlabel=xlabel, 
                ylabel=ylabel,
                legend=legend))

    def _update_loss_graphs_core(
            self, 
            graph_step: float,
            losses_G: Mapping[str, float], 
            losses_D: Mapping[str, float]    
        ) -> None:
        '''Updates generator and discriminator loss graphs.

        Takes a dictionary of loss values and a current graph step, generated 
        by the Trainer class, and updates Visdom loss graphs with the new data.

        Args:
            losses : Loss values to graph.
            graph_step : Graph step value to add input losses to.
        '''
        graph_info = [
            # Losses, visdom window name, graph title
            (losses_G, 'loss_G', 'Generator Loss'),
            (losses_D, 'loss_D', 'Discriminator Loss')]
        
        for graph in graph_info:             # Loop through graphs
            legend = list(graph[0].keys())   # Get loss keys
            values = list(graph[0].values()) # Get loss values
            steps = [graph_step]*len(values) # Build steps list
            self._update_graph(              # Update loss graph
                values=values, steps=steps, 
                window_internal_name=graph[1], window_title=graph[2], 
                xlabel='Iterations', ylabel='Loss', legend=legend)
    
    ### STORE THREAD DATA ###

    def _store_images(
            self,
            x: torch.Tensor, 
            y: torch.Tensor, 
            z: torch.Tensor, 
            title: str, 
            image_size: int=300
        ) -> None:
        self._image_queue.put({
            'tensors': (x.detach().cpu(), y.detach().cpu(), z.detach().cpu()),
            'title': title,
            'image_size': image_size})
        
    def _store_graph_data(
            self, 
            graph_step: float,
            losses_G: Mapping[str, float], 
            losses_D: Mapping[str, float]    
        ) -> None:
        self._graph_queue.put({
            'graph_step': graph_step,
            'losses_G': losses_G,
            'losses_D': losses_D})

    ### UPDATE (THREADED) ###

    def start_thread(self) -> None:
        '''Starts a thread for updating the Visdom visualizer.'''
        self.is_threaded = True
        self._image_queue = queue.Queue()
        self._graph_queue = queue.Queue()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._update, daemon=True)
        self._thread.start()

    def _update(self) -> None:
        while not self._stop.is_set():
            try: image_data = self._image_queue.get(timeout=1)
            except queue.Empty: continue
            try: graph_data = self._graph_queue.get(timeout=1)
            except queue.Empty: continue
            if self._stop.is_set(): break
            self._update_images_core(
                x=image_data['tensors'][0], 
                y=image_data['tensors'][1], 
                z=image_data['tensors'][2], 
                title=image_data['title'], 
                image_size=image_data['image_size'])
            self._image_queue.task_done()
            if self._stop.is_set(): break
            self._update_loss_graphs_core(
                graph_step=graph_data['graph_step'], 
                losses_G=graph_data['losses_G'], 
                losses_D=graph_data['losses_D'])
            self._graph_queue.task_done()

    def stop_thread(self) -> None:
        '''Stops the visdom visualizer thread.'''
        self._stop.set()
        