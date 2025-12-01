import sys
import time
import math
import random
import pathlib
from os import PathLike
from typing import Any, Callable
from contextlib import nullcontext, AbstractContextManager

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image
from torch_ema import ExponentialMovingAverage

from nectargan.trainers import Trainer
from nectargan.config import ConfigManager, DiffusionConfig
from nectargan.models import \
    DiffusionModel, LatentDiffusionModel, StableDiffusionModel
from nectargan.visualizer import DiffusionVisualizer

from nectargan.utils.meminfo import MemoryInfo_CUDA

class DiffusionTrainer(Trainer):
    def __init__(
            self, 
            config: str | PathLike | ConfigManager | None=None, 
            log_losses: bool=True
        ) -> None:
        super().__init__(config=config, quicksetup=True, log_losses=log_losses)
        self.config: DiffusionConfig = self.config
        self.CFG_M = self.config.model
        self.CFG_LR = self.config.model.common.dae.learning_rate

        self.meminfo = MemoryInfo_CUDA()
        self.global_step = 0
        self.total_steps = self.CFG_LR.steps_before_decay
        if self.CFG_LR.do_decay: self.total_steps += self.CFG_LR.decay_steps
        self._validate_fixed_captions()
        
        self.print_avg_loss = self.config.visualizer.console.average_loss
        self.accum_loss = 0.0

        self._init_model()
        self.register_losses()
        if self.CFG_M.use_ema: self._init_ema()
        self._load_checkpoints()

    ##### INIT #####        

    def _validate_fixed_captions(self) -> None:
        if self.config.model.captions.use_fixed_captions and \
           len(self.config.model.captions.fixed_captions) == 0:
            raise RuntimeError(
                'Length of fixed_captions in config file must be >1 to train '
                'with use_fixed_captions=true. Please add captions to config '
                'or set use_fixed_captions to false to continue.')

    def _init_model(self) -> None:
        self.model_type = self.config.model.model_type
        self.diffusion_timesteps = self.config.model.common.timesteps
        match self.model_type:
            case 'pixel' : model = DiffusionModel
            case 'latent': model = LatentDiffusionModel
            case 'stable': model = StableDiffusionModel
            case _: raise ValueError(f'Invalid model_type: {self.model_type}')
        self.model = model(config=self.config)

    def register_losses(self) -> None:
        print(f'Lambda MSE: {self.config.train.loss.lambda_mse}')
        self.loss_manager.register_loss_fn(
            loss_name='G_MSE', loss_fn=nn.MSELoss().to(self.device), 
            loss_weight=self.config.train.loss.lambda_mse, tags=['G'])

    def _init_ema(self) -> None:
        self.ema = ExponentialMovingAverage(
            self.model.autoencoder.parameters(), 
            decay=self.config.model.ema_decay)
        for param in self.ema.shadow_params:
            param.data = param.data.to(self.device)
        
    def _load_checkpoints(self) -> None:
        if self.config.train.load.continue_train:
            self.load_checkpoint(
                'DAE', self.model.autoencoder, self.model.opt_dae, 
                self.config.model.common.dae.learning_rate.base_rate)
            if self.CFG_M.use_ema: self.load_checkpoint('EMA', self.ema)

    ##### UTILS #####

    def get_dataset_length(self) -> int:
        return len(self.model.train_loader)
    
    def get_epoch_count(self) -> int:
        return math.ceil(self.total_steps / max(1, self.get_dataset_length()))

    ##### CONTEXTS #####

    def ema_context(self) -> AbstractContextManager[None]:
        return self.ema.average_parameters() \
            if self.CFG_M.use_ema else nullcontext()

    def autocast_context(self) -> AbstractContextManager[None]:
        return torch.amp.autocast(device_type=self.device) \
            if self.CFG_M.mixed_precision else nullcontext()

    ##### VISUALIZATION #####

    def update_display(
            self, 
            x: torch.Tensor, 
            y: torch.Tensor, 
            z: torch.Tensor
        ) -> None:
        if self.config.visualizer.visdom.enable:
            self.vis.update_images(
                x=x, y=y, z=z, title='x | x_t | pred x0', 
                image_size=self.config.visualizer.visdom.image_size)
            losses_G = self.loss_manager.get_loss_values(query=['G'])
            self.vis.update_loss_graphs(self.global_step, losses_G)

    def export_model_weights(
            self,
            mod: nn.Module, 
            opt: optim.Optimizer | None, 
            net: str,
        ) -> str | None: 
        checkpoint = { 'state_dict': mod.state_dict() }
        if not opt is None: checkpoint['optimizer'] = opt.state_dict()
        name = f'net{net}_step{str(self.global_step)}.pth.tar'
        output_path = pathlib.Path(self.experiment_dir, name)
        try: torch.save(checkpoint, output_path.as_posix())
        except Exception as e:
            message = 'Unable to save checkpoint file: {}'
            raise RuntimeError(message.format(output_path.as_posix())) from e
        return output_path.resolve().as_posix()

    def save_checkpoint(self, capture: bool=False) -> str | None:
        net = 'DAE'
        model = self.model
        path = self.export_model_weights(model.autoencoder, model.opt_dae, net)
        output = f'Checkpoint Saved ({net}): {path}'
        if self.CFG_M.use_ema:
            path = self.export_model_weights(self.ema, None, 'EMA')
            output = f'Checkpoint Saved (EMA): {path}'
        if capture: return output
        else: return None

    def export_examples(
            self, 
            context: torch.Tensor | None=None,
            cfg_scale: float=7.5
        ) -> None: 
        if self.config.model.captions.use_fixed_captions:
            captions = self.config.model.captions.fixed_captions 
            context, _ = self.model.text_encoder(captions)
        else: captions = self.model.captions
        for i in range(self.config.save.num_examples):
            caption = 'nullcaption'
            if not captions is None and not context is None:
                count = float(len(captions))
                if count > 0.0:
                    idx = int(math.floor(random.random() * count))
                    context = context[idx].unsqueeze(0).detach()
                    context = context.to(self.device)
                    caption = captions[idx]
            print(f'Running inference with caption:\n{caption}')
            with self.ema_context():
                with self.autocast_context(), torch.no_grad():
                    output = self.model.sample(
                        context=context, cfg_scale=cfg_scale)
            output = torch.clamp((output + 1) * 0.5, 0.0, 1.0)
            filename = f'step{self.global_step}_{i}.png'
            filepath = pathlib.Path(self.examples_dir, filename).resolve()
            save_image(output, filepath.as_posix())
        
    def _build_latent_vis_tensors(
            self, 
            x: torch.Tensor,
            x_t: torch.Tensor,
            predicted: torch.Tensor,
            timesteps: torch.Tensor
        ) -> tuple[torch.Tensor]:
        with self.autocast_context():
            params = self.model.noiseparams
            abar = params.alphas_cumprod[timesteps].view(-1,1,1,1)
            predicted = (x_t - (1.0 - abar).sqrt() * predicted) /  abar.sqrt()
            predicted = torch.clamp(predicted, -4.0, 4.0)
            tensors = [self.model.decode(t) for t in (x, x_t, predicted)]
        return tuple(tensors)
        
    def init_visdom(self) -> None:
        '''Initializes Visdom visualization for diffusion training.'''
        vcon = self.config.visualizer
        if vcon.visdom.enable:
            self.vis = DiffusionVisualizer(
                env=vcon.visdom.env_name,
                server=vcon.visdom.server,
                port=vcon.visdom.port) 
            self.vis.clear_env()

    def print_losses(
            self,
            iter: int,
            precision: int=2,
            capture: bool=False
        ) -> str | None:
        output = f'\n(total steps: {iter}) '
        if not self.print_avg_loss:
            losses = self.loss_manager.get_loss_values(precision=precision)
            output += 'Loss:'
            for loss in losses: output += f' {loss}: {losses[loss]}'
        else:
            divisor = max(1, self.config.visualizer.console.print_frequency)            
            average = round(self.accum_loss / divisor, 3)
            output += f'Average Loss: MSE: {average}'
            self.accum_loss = 0.0
        if not capture:
            print(output)
            return None
        else: return output

    def _display(
            self,
            x: torch.Tensor,
            x_t: torch.Tensor,
            predicted: torch.Tensor,
            timesteps: torch.Tensor
        ) -> None:
        vis = self.config.visualizer
        if self.global_step % vis.console.print_frequency == 0:
            avg_time = sum(self.model.batch_times) 
            avg_time /= max(1, len(self.model.batch_times))
            avg_time = round(avg_time, 3)
            self.print_losses(self.global_step)
            print(f'Average batch time: {avg_time} seconds')
            self.model.batch_times.clear()
        if self.global_step % vis.visdom.update_frequency == 0:
            if isinstance(self.model, LatentDiffusionModel) or \
               isinstance(self.model, StableDiffusionModel):
                x, x_t, predicted = self._build_latent_vis_tensors(
                    x, x_t, predicted, timesteps)
            self.update_display(x, x_t, predicted)

    ##### TRAINING METHODS #####

    def _build_timesteps(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        timesteps = torch.randint(
            0, self.config.model.common.timesteps, (B,),
            device=self.device).long()
        return timesteps
    
    def _ramp_up_lr(self) -> None:
        if self.global_step <= self.CFG_LR.ramp_up_steps:
            steps = max(1, self.CFG_LR.ramp_up_steps)
            lr = self.CFG_LR.base_rate * (self.global_step / steps)
            for param_group in self.model.opt_dae.param_groups:
                param_group['lr'] = lr

    def _decay_lr(self) -> None:
        if self.global_step > self.CFG_LR.steps_before_decay:
            steps = max(1, self.CFG_LR.decay_steps)
            current = self.global_step - self.CFG_LR.steps_before_decay
            lr = self.CFG_LR.base_rate * (1.0 - current / steps)
            for param_group in self.model.opt_dae.param_groups:
                param_group['lr'] = lr

    def backward(
            self, 
            loss: torch.Tensor, 
            max_norm: float=1.0,
            step: bool=True
        ) -> None:
        if self.CFG_M.accumulate_gradients: 
            loss /= float(self.CFG_M.gradient_accumulation_steps)
        if self.CFG_M.mixed_precision:
            self.model.g_scaler.scale(loss).backward()
            if step:
                self.model.g_scaler.unscale_(self.model.opt_dae)
                torch.nn.utils.clip_grad_norm_(
                    self.model.autoencoder.parameters(), max_norm)
                self.model.g_scaler.step(self.model.opt_dae)
                self.model.g_scaler.update()
        else:
            loss.backward()
            if step:
                torch.nn.utils.clip_grad_norm_(
                    self.model.autoencoder.parameters(), max_norm)
                self.model.opt_dae.step()
        if step and self.CFG_M.use_ema: self.ema.update()

    def _assert_finite(
            self, 
            x: torch.Tensor, 
            x_t: torch.Tensor, 
            noise: torch.Tensor
        ) -> None:
        assert torch.isfinite(x).all()
        assert torch.isfinite(x_t).all()
        assert torch.isfinite(noise).all()

    ##### TRAINING HOOKS #####

    def on_epoch_start(self, **kwargs: Any) -> None:
        self.model.batch_times.clear()

    def train_step(
            self,  
            x: torch.Tensor, 
            y: torch.Tensor | None=None, 
            idx: int=None,
            assert_finite: bool=False,
            **kwargs: Any
        ) -> None:
        self.global_step += 1
        if self.CFG_LR.ramp_up: self._ramp_up_lr()
        if not self.CFG_M.accumulate_gradients or \
           self.global_step % self.CFG_M.gradient_accumulation_steps == 0:
            self.model.opt_dae.zero_grad()
        
        timesteps = self._build_timesteps(x)
        with self.autocast_context():  
            x_t, noise = self.model.q_sample(x=x, t=timesteps)            
            if not self.CFG_M.mixed_precision:
                x_t = x_t.to(device=self.device, dtype=torch.float32)
                noise = noise.to(device=self.device, dtype=torch.float32)
                if y is not None and torch.is_floating_point(y):
                    y = y.to(self.device, dtype=torch.float32)
            predicted = self.model.autoencoder(x_t, timesteps, context=y)
            loss_G_MSE = self.loss_manager.compute_loss_xy(
                'G_MSE', predicted, noise, self.current_epoch)
        
        if assert_finite: self._assert_finite(x, x_t, noise)
        if self.print_avg_loss: self.accum_loss += loss_G_MSE.mean().item()
        step = True if not self.CFG_M.accumulate_gradients else \
            (self.global_step + 1) % self.CFG_M.gradient_accumulation_steps==0
        self.backward(loss_G_MSE, step)

        if self.global_step == self.total_steps:
            self.export_examples(context=y)
            self.save_checkpoint()
            print('Training complete!')
            exit(0)

        self._display(x, x_t, predicted, timesteps)
        if self.global_step % self.config.save.example_save_rate == 0: 
            self.export_examples(context=y)    
        if self.config.save.save_model and \
           self.global_step % self.config.save.model_save_rate == 0:
            self.save_checkpoint()
        
    def on_epoch_end(self, **kwargs: Any) -> None:
        pass

    ##### DIFFUSION TRAINING LOOP #####

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
                self.model._trainer_core(
                    train_fn, callback_kwargs.get('train_step', {}))
            except KeyboardInterrupt:
                sys.exit('Interrupt Recieved: Stopping training...')
            finally: self.vis.stop_thread()
        else: self.model._trainer_core(
            train_fn, callback_kwargs.get('train_step', {}))
        
        end_fn(**callback_kwargs.get('on_epoch_end', {})) 

        end_time = time.perf_counter()
        self.last_epoch_time = end_time-start_time


