import sys
import time
import pathlib
from os import PathLike
from typing import Any, Callable

import torch
import torch.nn as nn
from torchvision.utils import save_image
from torch_ema import ExponentialMovingAverage

from nectargan.trainers import Trainer
from nectargan.config import ConfigManager, DiffusionConfig
from nectargan.dataset import DiffusionDataset
from nectargan.models import DiffusionModel, LatentDiffusionModel
from nectargan.visualizer import DiffusionVisualizer

class DiffusionTrainer(Trainer):
    def __init__(
            self, 
            config: str | PathLike | ConfigManager | None=None, 
            log_losses: bool=True
        ) -> None:
        super().__init__(config=config, quicksetup=True, log_losses=log_losses)
        
        self.batch_times = []
        self.config: DiffusionConfig = self.config
        self._get_model_info()

        self.build_dataloader('train', is_train=True)
        self._init_model()
        self.register_losses()

        self.ema = ExponentialMovingAverage(
            self.model.autoencoder.parameters(), decay=0.9999)
        for param in self.ema.shadow_params:
            param.data = param.data.to(self.device)

        if self.config.train.load.continue_train:
            self.load_checkpoint(
                'DAE', self.model.autoencoder, self.model.opt_dae, 
                self.config.model.common.dae.learning_rate.initial)
            self.load_checkpoint('EMA', self.ema)

    def _get_model_info(self) -> None:
        self.model_type = self.config.model.model_type
        self.diffusion_timesteps = self.config.model.common.timesteps
        match self.model_type:
            case 'pixel': 
                self.diffusion_model = DiffusionModel
                self.model_config = self.config.model.pixel
            case 'latent': 
                self.diffusion_model = LatentDiffusionModel
                self.model_config = self.config.model.latent
            case _:
                raise ValueError(
                    f'Invalid model_type: {self.model_type}')

    def build_dataloader(
            self, 
            loader_type: str,
            is_train: bool=True
        ) -> None:
        '''Initializes a dataloader of the given type from a DiffusionDataset.

        This function will grab the dataroot path from the config and use it to
        first create a DiffusionDataset instance of the given loader type, then 
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
        dataset = DiffusionDataset(
            config=self.config, root_dir=dataset_path, is_train=is_train)
        self.train_loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.config.dataloader.batch_size, 
            shuffle=True, num_workers=self.config.dataloader.num_workers,
            pin_memory=True, drop_last=True)

    def _init_model(self) -> None:
        self.model = self.diffusion_model(config=self.config)
        if self.model_type == 'latent':
            cache_cfg = self.model_config.precache
            if cache_cfg.enable:
                self.train_loader = self.model.cache_latents(
                    batch_size=cache_cfg.batch_size,
                    shard_size=cache_cfg.shard_size)
                self.model.read_from_cache = True

    def register_losses(self) -> None:
        print(f'Lambda MSE: {self.config.train.loss.lambda_mse}')
        self.loss_manager.register_loss_fn(
            loss_name='G_MSE', loss_fn=nn.MSELoss().to(self.device), 
            loss_weight=self.config.train.loss.lambda_mse, tags=['G'])

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
                x=x, y=y, z=z, title='x | x_t | pred x0', 
                image_size=self.config.visualizer.visdom.image_size)
            
            losses_G = self.loss_manager.get_loss_values(query=['G'])

            graph_step = self.current_epoch + idx / len(self.train_loader) 
            self.vis.update_loss_graphs(graph_step, losses_G)

    def save_checkpoint(
            self, 
            capture: bool=False
        ) -> str | None:
        net = 'DAE'
        model = self.model
        path = self.export_model_weights(model.autoencoder, model.opt_dae, net)
        output = f'Checkpoint Saved ({net}): {path}'
        path = self.export_model_weights(self.ema, None, 'EMA')
        output = f'Checkpoint Saved ({net}): {path}'

        if capture: return output
        else: return None

    def export_examples(self, idx: int) -> None:
        with self.ema.average_parameters():
            with torch.amp.autocast(self.device):
                with torch.no_grad():
                    output = self.model.sample()
        normalized_output = torch.clamp((output + 1) * 0.5, 0.0, 1.0)
        filename = f'epoch{self.current_epoch}_{idx}.png'
        output_path = pathlib.Path(self.examples_dir, filename).resolve()
        save_image(
            normalized_output, output_path.as_posix(), 
            normalize=False, value_range=None)
        
    def _build_latent_vis_tensors(
            self, 
            x: torch.Tensor,
            x_t: torch.Tensor,
            predicted: torch.Tensor,
            timesteps: torch.Tensor
        ) -> tuple[torch.Tensor]:
        with torch.amp.autocast(self.device):
            abar = self.model.noiseparams.alphas_cumprod[
                timesteps].view(-1,1,1,1)
            pred_x0 = torch.clamp(
                (x_t - (1.0 - abar).sqrt() * predicted) /  abar.sqrt(),
                -4.0, 4.0)
            x = self.model.decode(x)
            x_t = self.model.decode(x_t)
            predicted = self.model.decode(pred_x0)
        return x, x_t, predicted
        
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
        self.model.g_scaler.scale(loss).backward()
        self.model.g_scaler.step(self.model.opt_dae)
        self.model.g_scaler.update()
        
        self.ema.update()

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
        self.model.opt_dae.zero_grad()

        timesteps = torch.randint(
            0, self.diffusion_timesteps, 
            (self.config.dataloader.batch_size,), 
            device=self.device
        ).long()

        with torch.amp.autocast(self.device):
            x_t, noise = self.model.q_sample(x=x, t=timesteps)
            predicted = self.model.autoencoder(x_t, timesteps)
            loss_G_MSE = self.loss_manager.compute_loss_xy(
                'G_MSE', predicted, noise, self.current_epoch)
        self.backward(loss_G_MSE)

        vis = self.config.visualizer
        if idx % vis.console.print_frequency == 0:
            avg_time = sum(self.batch_times) / max(1, len(self.batch_times))
            avg_time = round(avg_time, 3)
            print(f'Average batch time: {avg_time} seconds')
            self.batch_times.clear()
        if idx % vis.visdom.update_frequency == 0:
            if isinstance(self.model, LatentDiffusionModel):
                x, x_t, predicted = self._build_latent_vis_tensors(
                    x, x_t, predicted, timesteps)
            self.update_display(x, x_t, predicted, idx)
        if idx % self.config.save.example_save_rate == 0: 
            self.export_examples(idx)    

    def on_epoch_end(self, **kwargs: Any) -> None:
        self.print_end_of_epoch()
        s = self.config.save
        lr = self.config.model.common.dae.learning_rate
        epoch_count = lr.epochs + lr.epochs_decay
        if self.current_epoch == epoch_count: self.save_checkpoint()
        elif (s.save_model and self.current_epoch % s.model_save_rate == 0):
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
            train_step_fn(x, None, idx, **train_step_kwargs)
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
        else: self._train_diffusion_core(
            train_fn, callback_kwargs.get('train_step', {}))
        
        end_fn(**callback_kwargs.get('on_epoch_end', {})) 

        end_time = time.perf_counter()
        self.last_epoch_time = end_time-start_time


