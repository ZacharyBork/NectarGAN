from collections.abc import Mapping
import queue

import torch
from nectargan.visualizer.visdom.visualizer import VisdomVisualizer

class DiffusionVisualizer(VisdomVisualizer):
    def __init__(
            self, 
            env = 'main', 
            server = 'http://localhost', 
            port = 8097
        ) -> None:
        super().__init__(env, server, port)

    ### UPDATE (UNTHREADED) ###

    def update_loss_graphs(
            self, 
            graph_step: float,
            losses_G: Mapping[str, float]
        ) -> None:
        if self.is_threaded:
            self._store_graph_data(graph_step, losses_G)
        else: 
            self._update_loss_graphs_core(graph_step, losses_G)

    def _update_images_core(
            self,
            x: torch.Tensor, 
            y: torch.Tensor, 
            z: torch.Tensor, 
            title: str, 
            image_size: int=200
        ) -> None:
        x_s, y_s, z_s = x.unbind(), y.unbind(), z.unbind()
        for i in range(x.shape[0]):
            composite = torch.cat([
                self._denorm_tensor(x_s[i]), 
                self._denorm_tensor(y_s[i]), 
                self._denorm_tensor(z_s[i])], dim=2)
            self.vis.images(
                composite, win=f'comparison_grid{i}', nrow=1, padding=2,
                opts=dict(title=title, width=image_size*3, height=image_size))
        
    def _update_loss_graphs_core(
            self, 
            graph_step: float,
            losses_G: Mapping[str, float]
        ) -> None:
        graph_info = (losses_G, 'loss_G', 'Generator Loss')
        
        legend = list(graph_info[0].keys())
        values = list(graph_info[0].values())
        steps = [graph_step]*len(values)
        self._update_graph(
            values=values, steps=steps, 
            window_internal_name=graph_info[1], window_title=graph_info[2], 
            xlabel='Iterations', ylabel='Loss', legend=legend)

    ### STORE THREAD DATA ###
        
    def _store_graph_data(
            self, 
            graph_step: float,
            losses_G: Mapping[str, float]
        ) -> None:
        self._graph_queue.put({
            'graph_step': graph_step,
            'losses_G': losses_G})

    ### UPDATE (THREADED) ###

    def _update(self) -> None:
        while not self._stop.is_set():
            try: image_data = self._image_queue.get(timeout=1)
            except queue.Empty: continue
            try: graph_data = self._graph_queue.get(timeout=1)
            except queue.Empty: continue
            if self._stop.is_set(): break
            self._update_loss_graphs_core(
                graph_step=graph_data['graph_step'], 
                losses_G=graph_data['losses_G'])
            self._graph_queue.task_done()
            if self._stop.is_set(): break
            self._update_images_core(
                x=image_data['tensors'][0], 
                y=image_data['tensors'][1], 
                z=image_data['tensors'][2], 
                title=image_data['title'], 
                image_size=image_data['image_size'])
            self._image_queue.task_done()


