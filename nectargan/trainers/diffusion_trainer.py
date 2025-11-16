import sys
import time
import pathlib
from os import PathLike
from typing import Any, Callable

import torch
import torch.nn as nn
from torchvision.utils import save_image

from nectargan.trainers.trainer import Trainer
from nectargan.config.config_manager import ConfigManager
from nectargan.dataset.unpaired_dataset import UnpairedDataset
from nectargan.models.diffusion.models.pixel import DiffusionModel
from nectargan.models.diffusion.models.latent import LatentDiffusionModel
from nectargan.visualizer.visdom.diffusion import DiffusionVisualizer

class DiffusionTrainer(Trainer):
    def __init__(
            self, 
            config: str | PathLike | ConfigManager | None=None, 
            diffusion_model: DiffusionModel=LatentDiffusionModel,
            diffusion_timesteps: int=1000,
            log_losses: bool=True
        ) -> None:
        super().__init__(config=config, quicksetup=True, log_losses=log_losses)
        self.diffusion_timesteps = diffusion_timesteps
        self.batch_times = []

        self.diffusion_model = diffusion_model(
            timesteps=self.diffusion_timesteps, device=self.device)
        self.train_loader = self.build_dataloader('train', is_train=True)
        self.register_losses()

        if self.config.train.load.continue_train:
            lr_initial = self.config.train.generator.learning_rate.initial
            model = self.diffusion_model
            self.load_checkpoint(
                'DAE', model.autoencoder, model.opt_dae, lr_initial)

    def build_dataloader(
            self, 
            loader_type: str,
            is_train: bool=True
        ) -> torch.utils.data.DataLoader:
        '''Initializes a dataloader of the given type from a UnpairedDataset.

        This function will grab the dataroot path from the config and use it to
        first create a UnpairedDataset instance of the given loader type, then 
        use that dataset to create and return a Torch Dataloader.

        Args:
            loader_type : Type of loader to create (e.g. 'train', 'val')

        Returns:
            torch.utils.data.DataLoader : Dataloader created from the dataset.
        '''
        dataset_path = pathlib.Path(
            self.config.dataloader.dataroot, loader_type).resolve()
        if not dataset_path.exists():
            message = f'Unable to locate dataset at: {dataset_path.as_posix()}'
            raise FileNotFoundError(message)
        dataset = UnpairedDataset(
            config=self.config, root_dir=dataset_path, is_train=is_train)
        return torch.utils.data.DataLoader(
            dataset, batch_size=self.config.dataloader.batch_size, 
            shuffle=True, num_workers=self.config.dataloader.num_workers)

    def register_losses(self) -> None:
        self.loss_manager.register_loss_fn(
            loss_name='G_MSE', loss_fn=nn.MSELoss().to(self.device), 
            loss_weight=1.0, tags=['G'])

    def update_display(
            self, 
            x: torch.Tensor, 
            y: torch.Tensor, 
            z: torch.Tensor,
            idx: int
        ) -> None:
        self.loss_manager.print_losses(self.current_epoch, idx)
        
        if self.config.visualizer.visdom.enable:
            self.vis.update_images(
                x=x, y=y, z=z, title='Input | Real | Predicted', 
                image_size=self.config.visualizer.visdom.image_size)
            
            losses_G = self.loss_manager.get_loss_values(query=['G'])

            graph_step = self.current_epoch + idx / len(self.train_loader) 
            self.vis.update_loss_graphs(graph_step, losses_G)

    def save_checkpoint(
            self, 
            capture: bool=False
        ) -> str | None:
        net = 'DAE'
        model = self.diffusion_model
        path = self.export_model_weights(model.autoencoder, model.opt_dae, net)
        output = f'Checkpoint Saved ({net}): {path}'
        if capture: return output
        else: return None

    def export_examples(self, x: torch.Tensor, idx: int) -> None:
        with torch.amp.autocast('cuda'):
            with torch.no_grad():
                output = self.diffusion_model.sample()
        normalized_output = torch.clamp((output + 1) * 0.5, 0.0, 1.0)
        filename = f'epoch{self.current_epoch}_{idx}.png'
        output_path = pathlib.Path(self.examples_dir, filename).resolve()
        save_image(
            normalized_output, output_path.as_posix(), 
            normalize=False, value_range=None)
        
    def init_visdom(self) -> None:
        '''Initializes Visdom visualization for diffusion training.'''
        vcon = self.config.visualizer
        if vcon.visdom.enable:
            self.vis = DiffusionVisualizer(
                env=vcon.visdom.env_name,
                server=vcon.visdom.server,
                port=vcon.visdom.port) 
            self.vis.clear_env()

    def backward(self, loss: torch.Tensor) -> None:
        self.diffusion_model.g_scaler.scale(loss).backward()
        self.diffusion_model.g_scaler.step(self.diffusion_model.opt_dae)
        self.diffusion_model.g_scaler.update()

    ##### TRAINING HOOKS #####

    def on_epoch_start(self, **kwargs: Any) -> None:
        self.batch_times.clear()

    def train_step(
            self,  
            x: torch.Tensor, 
            y: torch.Tensor, 
            idx: int,
            **kwargs: Any
        ) -> None:
        self.diffusion_model.opt_dae.zero_grad()

        batch_size = x.shape[0]
        timesteps = torch.randint(
            0, self.diffusion_timesteps, 
            (batch_size,), 
            device=self.device
        ).long()

        with torch.amp.autocast('cuda'):
            x_t, noise = self.diffusion_model.q_sample(x=x, t=timesteps)
            predicted = self.diffusion_model.autoencoder(x_t, timesteps)
            loss_G_MSE = self.loss_manager.compute_loss_xy(
                'G_MSE', predicted, noise, self.current_epoch)
            noise = self.diffusion_model.decode_from_latent(noise)
            predicted = self.diffusion_model.decode_from_latent(predicted)
        self.backward(loss_G_MSE)

        if idx % self.config.visualizer.visdom.update_frequency == 0:
            # TO-DO: Change visualizer triplet to:
            # (ground truth (x), noisy input (x_t), predicted clean (x_0_pred))
            self.update_display(x, noise, predicted, idx)
            
            avg_time = sum(self.batch_times) / max(1, len(self.batch_times))
            avg_time = round(avg_time, 3)
            print(f'Average batch time: {avg_time} seconds')
            self.batch_times.clear()
        if idx % 200 == 0: self.export_examples(x, idx)
            

    def on_epoch_end(self, **kwargs: Any) -> None:
        self.print_end_of_epoch()

        cfg = self.config
        lr = cfg.train.generator.learning_rate
        epoch_count = lr.epochs + lr.epochs_decay
        if self.current_epoch == epoch_count:
            self.save_checkpoint()
        elif (cfg.save.save_model and 
              self.current_epoch % cfg.save.model_save_rate == 0):
            self.save_checkpoint()

    ##### DIFFUSION TRAINING LOOP #####

    def _train_diffusion_core(
            self, 
            train_step_fn: Callable[[torch.Tensor, torch.Tensor, int], None],
            train_step_kwargs: dict[str, Any]
        ) -> None:
        for idx, x in enumerate(self.train_loader):
            start_time = time.time()
            x: torch.Tensor = x.to(self.device)
            y: torch.Tensor = torch.zeros_like(x).to(self.device)
            train_step_fn(x, y, idx, **train_step_kwargs)
            batch_time = time.time() - start_time
            self.batch_times.append(batch_time)

    def train_diffusion(
            self,
            epoch:int,
            on_epoch_start: Callable[[], None] | None=None,
            train_step: Callable[[torch.Tensor, torch.Tensor, int], None] | 
            None=None,
            on_epoch_end: Callable[[], None] | None=None,
            multithreaded: bool=True,
            callback_kwargs: dict[str, dict[str, Any]] = {}
        ) -> None:
        if self.config.train.load.continue_train:
            self.current_epoch = 1 + epoch + self.config.train.load.load_epoch
        else: self.current_epoch = epoch + 1

        start_fn = on_epoch_start or self.on_epoch_start
        train_fn = train_step or self.train_step
        end_fn = on_epoch_end or self.on_epoch_end

        start_time = time.perf_counter()
        
        start_fn(**callback_kwargs.get('on_epoch_start', {})) 
        if multithreaded and self.config.visualizer.visdom.enable: 
            try:
                self.vis.start_thread()
                self._train_diffusion_core(
                    train_fn, callback_kwargs.get('train_step', {}))
            except KeyboardInterrupt:
                sys.exit('Interrupt Recieved: Stopping training...')
            finally: self.vis.stop_thread()
        else: self._train_paired_core(
            train_fn, callback_kwargs.get('train_step', {}))
        
        end_fn(**callback_kwargs.get('on_epoch_end', {})) 

        end_time = time.perf_counter()
        self.last_epoch_time = end_time-start_time


