import torch
import numpy as np
from visdom import Visdom

from typing import Sequence, Any
from collections.abc import Mapping

class VisdomVisualizer():
    def __init__(self, env: str='main', port: int=8097):
        self.env = env
        self.port = port
        self.init_visdom()

    def start_server(self):
        pass

    def init_visdom(self):
        '''Initializes Visdom and validates endpoint connection.

        Also assigns Visdom client to self.vis internally to send commands.
        
        Raises:
            ConnectionError : If unable to connect to Visdom server.
        '''
        vis = Visdom(env=self.env)
        if not vis.check_connection():
            raise ConnectionError(
                'Unable to connect to Visdom server. Please ensure it is running.')
        self.vis = vis

    def clear_env(self) -> None:
        '''Closes all windows in the VisdomVisualizer's environment.'''
        self.vis.close(win=None, env=self.env)
        
    def denorm_tensor(self, tensor: torch.Tensor, clamp: bool=True) -> torch.Tensor: 
        '''Denormalizes a pix2pix output tensor [-1, 1] -> [0, 1] for viewing.
        
        The final upsampling convolutional layer in the pix2pix architecture uses
        tanh nonlinearity. This outputs a tensor normalized [-1, 1]. This function 
        just takes those output tensors and renormalizes them to the [0, 1] range
        which is what Visdom is expecting for images.

        Args:
            tensor : The torch.Tensor object to denormalize.
            clamp : Whether to clamp the denormed values to [0, 1].

        Returns:
            torch.Tensor: The input tensor renormalized to [0, 1].
        '''
        norm = (tensor + 1) * 0.5
        return torch.clamp(norm, 0.0, 1.0) if clamp else norm

    def update_images(
            self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, 
            title: str, image_size: int=300) -> None:
        '''Updates the Visdom [x, y, z] image grid.
        
        Takes three torch.Tensor objects as input and normalizes them [0, 1], then
        concatenates them [x, y, z]. Creates a torchvision.utils.make_grid
        from them and updates Visdom with the concatenated image.

        Args:
            x : The first input tensor (left image in grid).
            y : The second input tensor (middle image in grid).
            z : The third input tensor (right image in grid).
            title : The title of the concatenated image window.
            image_size : The width and height, in pixels, to render each image. 
        '''
        composite = torch.cat([
            self.denorm_tensor(x), 
            self.denorm_tensor(y), 
            self.denorm_tensor(z)], dim=3)
        self.vis.images(
            composite.cpu(), win='comparison_grid', nrow=1, padding=2,
            opts=dict(title=title, width=image_size*3, height=image_size)
        )

    def update_graph(
        self, values: Sequence[float], steps: Sequence[float],
        window_internal_name: str, window_title: str,
        xlabel: str, ylabel: str, legend: list[str]) -> None:
        '''Updates a visdom.Visdom.line graph to add new values at steps.

        Args:
            values : The values to add to the graph.
            steps : The steps, 1 per value, to add each value at on the graph.
            window_internal_name : Internal name of the Visdom graph window.
            window_title : The human-readable title of the Visdom graph window.
            xlabel : The X axis label of the graph.
            ylabel : The Y axis label of the graph.
            legend : List of strings, one per value, to title each line on the graph.
        '''
        self.vis.line(
            Y=np.column_stack(values),
            X=np.column_stack(steps),
            win=window_internal_name, update='append',
            opts=dict(title=window_title, xlabel=xlabel, ylabel=ylabel,legend=legend))

    def update_loss_graphs(self, losses: Mapping[str, float], graph_step: float) -> None:
        '''Updates generator and discriminator loss graphs.

        Takes a dictionary of loss values and a current graph step, generated 
        by the Trainer class, and updates Visdom loss graphs with the new data.

        Args:
            losses : Loss values to graph.
            graph_step : Graph step value to add input losses to.
        '''
        self.update_graph( # Generator loss graph
            values=(losses['G_GAN'], losses['G_L1'], losses['G_SOBEL'], losses['G_LAP']),
            steps=(graph_step, graph_step, graph_step, graph_step),
            window_internal_name='loss_G', window_title='Generator Loss',
            xlabel='Iterations', ylabel='Loss', legend=['G_GAN', 'G_L1', 'G_SOBEL', 'G_LAP'])
        self.update_graph( # Discriminator loss graph
            values=(losses['D_real'], losses['D_fake']), steps=(graph_step, graph_step),
            window_internal_name='loss_D', window_title='Discriminator Loss',
            xlabel='Iterations', ylabel='Loss', legend=['D_real', 'D_fake'])
        