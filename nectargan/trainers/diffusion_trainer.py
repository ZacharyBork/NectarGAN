import math
import random
import pathlib
from os import PathLike
from typing import Any
from contextlib import nullcontext, AbstractContextManager

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image
from torch_ema import ExponentialMovingAverage

from nectargan.trainers import Trainer
from nectargan.config import ConfigManager, DiffusionConfig
from nectargan.models.diffusion.data import AverageLossTracker
from nectargan.models import \
    PixelDiffusionModel, LatentDiffusionModel, StableDiffusionModel
from nectargan.visualizer import DiffusionVisualizer

from nectargan.utils.meminfo import MemoryInfo_CUDA

class DiffusionTrainer(Trainer):
    def __init__(
            self, 
            config: str | PathLike | ConfigManager, 
            log_losses: bool=True
        ) -> None:
        '''Init function for the DiffusionTrainer class.

        Args:
            config: Something representing a DiffusionConfig, either a str or
                os.Pathlike object pointing to a config JSON, or a Python dict
                representing the data from a config JSON, or a pre-defined
                ConfigManager instance.
            log_losses : If enabled (default) for a given `Trainer` instance, 
                losses run within the context of the `Trainer` will be cached
                and periodically dumped to the loss log JSON.
        '''
        assert config
        super().__init__(config=config, quicksetup=True, log_losses=log_losses)
        self.config: DiffusionConfig = self.config
        self.CFG_M = self.config.model
        self.CFG_LR = self.CFG_M.dae.learning_rate

        self.meminfo = MemoryInfo_CUDA()
        self.current_iteration = 0
        self.total_steps = self.CFG_LR.steps_before_decay
        if self.CFG_LR.do_decay: self.total_steps += self.CFG_LR.decay_steps
        self._validate_fixed_captions()
        
        self.print_avg_loss = self.config.visualizer.console.average_loss
        self.visdom_avg_loss = self.config.visualizer.visdom.average_loss
        self.loss_tracker = AverageLossTracker(config=self.config)

        self._init_model()
        self.register_losses()
        if self.CFG_M.use_ema: self._init_ema()
        self._load_checkpoints()

    ##### INIT #####        

    def _validate_fixed_captions(self) -> None:
        '''Ensures that fixed captions exist if enabled in the config.'''
        if self.config.captions.use_fixed_captions and \
           len(self.config.captions.fixed_captions) == 0:
            raise RuntimeError(
                'Length of fixed_captions in config file must be >1 to train '
                'with use_fixed_captions=true. Please add captions to config '
                'or set use_fixed_captions to false to continue.')

    def _init_model(self) -> None:
        '''Initialized a diffusion model by type from the input config.'''
        self.model_type = self.config.model.model_type
        self.diffusion_timesteps = self.config.model.noise_schedule.timesteps
        match self.model_type:
            case 'pixel' : model = PixelDiffusionModel
            case 'latent': model = LatentDiffusionModel
            case 'stable': model = StableDiffusionModel
            case _: raise ValueError(f'Invalid model_type: {self.model_type}')
        self.model = model(config=self.config)
        self.trainer_core = model._trainer_core
        self.model.opt_dae.zero_grad()

    def register_losses(self) -> None:
        '''Registers loss functions for training.'''
        print(f'Lambda MSE: {self.config.train.loss.lambda_mse}')
        self.loss_manager.register_loss_fn(
            loss_name='G_MSE', loss_fn=nn.MSELoss().to(self.device), 
            loss_weight=self.config.train.loss.lambda_mse, tags=['G'])

    def _init_ema(self) -> None:
        '''Initializes a PyTorch EMA module.'''
        self.ema = ExponentialMovingAverage(
            self.model.autoencoder.parameters(), 
            decay=self.config.model.ema_decay)
        for param in self.ema.shadow_params:
            param.data = param.data.to(self.device)
        
    def _load_checkpoints(self) -> None:
        '''Loads model checkpoint(s) to continue training.'''
        if self.config.train.load.continue_train:
            self.load_checkpoint(
                'DAE', self.model.autoencoder, self.model.opt_dae, 
                self.config.model.dae.learning_rate.base_rate)
            if self.CFG_M.use_ema: self.load_checkpoint('EMA', self.ema)

    ##### UTILS #####

    def get_dataset_length(self) -> int:
        '''Gets the length of the current dataset.'''
        return len(self.model.dataloader)
    
    def get_epoch_count(self) -> int:
        '''Calculates an epoch count based on dataset size and step count.'''
        return math.ceil(self.total_steps / max(1, self.get_dataset_length()))

    ##### CONTEXTS #####

    def ema_context(self) -> AbstractContextManager[None]:
        '''Returns EMA context if EMA is enabled, else returns nullcontext.
        
        Returns:
            AbstractContextManager : The EMA context to use, or a nullcontext
                if "use_ema" is disabled in the input config.
        '''
        return self.ema.average_parameters() \
            if self.CFG_M.use_ema else nullcontext()

    def autocast_context(self) -> AbstractContextManager[None]:
        '''Returns autocast context to current device if using mixed precision.

        Returns:
            AbstractContextManager : Either an autocast context based on the
                current PyTorch device, or a nullcontext if mixed precision is
                disabled in the config file.
        '''
        return torch.amp.autocast(device_type=self.device) \
            if self.CFG_M.mixed_precision else nullcontext()

    ##### VISUALIZATION #####

    def update_display(
            self, 
            x: torch.Tensor, 
            y: torch.Tensor, 
            z: torch.Tensor
        ) -> None:
        '''Updates Visdom during training with loss info and examples.
        
        Args:
            x : The first tensor of the training triplet to display. Usually
                the input image, either directly, or decoded from latent space
                if using latent pre-caching
            y : The second tensor to display, generally the noise tensor from
                the given timestep.
            z : The third tensor to display, generally the predicted clean
                image from the denoising autoencoder.
        '''
        if self.config.visualizer.visdom.enable:
            self.vis.update_images(
                x=x, y=y, z=z, title='x | x_t | pred x0', 
                image_size=self.config.visualizer.visdom.image_size)
            if not self.visdom_avg_loss:
                loss = self.loss_manager.get_loss_values(query=['G'])
            else: loss = { 'G_MSE': self.loss_tracker.get_average('visdom') }
            self.vis.update_loss_graphs(self.current_iteration, loss)

    def export_model_weights(
            self,
            mod: nn.Module, 
            opt: optim.Optimizer | None, 
            net: str,
        ) -> str | None: 
        '''Saves checkpoint files for each network in the model.
        
        Checkpoints will be saved to the current experiment directory as
        ".pth.tar" files, tagged with the name of the network and the current
        step at the time of export.

        Args:
            mod : The nn.Module to save a checkpoint file for.
            opt : The optimizer for the Module, if applicable.
            net : A human-readable tag for the Module being saved (i.e. "DAE",
                "EMA"). Will be used to name the checkpoint file.

        Raises:
            RuntimeError : If unable to save checkpoint file.
        '''
        checkpoint = { 'state_dict': mod.state_dict() }
        if not opt is None: checkpoint['optimizer'] = opt.state_dict()
        name = f'iter{str(self.current_iteration)}_net{net}.pth.tar'
        output_path = pathlib.Path(self.experiment_dir, name)
        try: torch.save(checkpoint, output_path.as_posix())
        except Exception as e:
            message = 'Unable to save checkpoint file: {}'
            raise RuntimeError(message.format(output_path.as_posix())) from e
        return output_path.resolve().as_posix()

    def save_checkpoint(self, capture: bool=False) -> str | None:
        '''Wrapper for DiffusionTrainer.export_model_weights().
        
        Saves a checkpoint file for the denoising autoencoder, and alse for the 
        EMA module if applicable.

        Args:
            capture : If False, this function will print a string to the
                console with the path to the saved checkpoint. If True, it will
                instead return the same string for you to process however you'd
                like.
        
        Returns:
            str | None : The log string which would have been printed, or None
                if capture is False. 
        '''
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
        '''Evals the DAE and exports the result to the experiment directory.
        
        Args:
            context : The context Tensor to pass to the diffusion model, or
                None if not using text conditioning.
            cfg_scale : The classifier free guidance scale to use during
                inference. SD uses 7.5. Sometimes it can be useful early on in
                training to use a lower cfg_scale, as it makes it easier to
                spot problems. Past a certain point in training, though, lower
                values will degrade model performance. 
        '''
        if self.config.captions.use_fixed_captions:
            captions = self.config.captions.fixed_captions 
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
            for scale in [1.0, cfg_scale]:
                with self.ema_context():
                    with self.autocast_context(), torch.no_grad():
                        output = self.model.sample(
                            context=context, cfg_scale=scale)
                output = torch.clamp((output + 1) * 0.5, 0.0, 1.0)
                scale_tag = str(scale).replace('.', '')
                name = f'step{self.current_iteration}_{i}_cfg{scale_tag}.png'
                filepath = pathlib.Path(self.examples_dir, name).resolve()
                save_image(output, filepath.as_posix())
        
    def _build_latent_vis_tensors(
            self, 
            x: torch.Tensor,
            x_t: torch.Tensor,
            predicted: torch.Tensor,
            timesteps: torch.Tensor
        ) -> tuple[torch.Tensor]:
        '''Decodes tensors from latent space for viewing in Visdom.

        Args:
            x : The input image tensor.
            x_t : The noise tensor from the given timestep.
            z : The predicted clean image tensor from the DAE.
            timesteps : The timesteps tensor for the current batch.

        Returns : The input tensors decoded from latent space as a tuple,
            ordered as: (x, x_t, predicted)
        '''
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
        '''Prints (or returns) loss values for the current iteration.
        
        Args:
            iter : The current iteration.
            precision : The rounding precision for the loss values.
            capture : If False, this function will print a string to the
                console with loss values. If True, it will instead return the 
                same string for you to process however you'd like.
        '''
        output = f'\n(total steps: {iter}) '
        if not self.print_avg_loss:
            losses = self.loss_manager.get_loss_values(precision=precision)
            output += 'Loss:'
            for loss in losses: output += f' {loss}: {losses[loss]}'
        else:
            average = self.loss_tracker.get_average('console')
            output += f'Average Loss: MSE: {average}'
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
        '''Updates Visdom and prints loss values to the console.

        This function reads the update frequencies for console and Visdom from
        the current config, and will only update each if it is appropriate to
        do so on the current iteration.
        
        Args:
            x : The input image tensor.
            x_t : The noise tensor from the given timestep.
            z : The predicted clean image tensor from the DAE.
            timesteps : The timesteps tensor for the current batch.
        '''
        vis = self.config.visualizer
        if self.current_iteration % vis.console.print_frequency == 0:
            avg_time = sum(self.model.batch_times) 
            avg_time /= max(1, len(self.model.batch_times))
            avg_time = round(avg_time, 3)
            self.print_losses(self.current_iteration)
            print(f'Average batch time: {avg_time} seconds')
            self.model.batch_times.clear()
        if self.current_iteration % vis.visdom.update_frequency == 0:
            if isinstance(self.model, LatentDiffusionModel) or \
               isinstance(self.model, StableDiffusionModel):
                x, x_t, predicted = self._build_latent_vis_tensors(
                    x, x_t, predicted, timesteps)
            self.update_display(x, x_t, predicted)

    ##### TRAINING METHODS #####

    def _build_timesteps(self, x: torch.Tensor) -> torch.Tensor:
        '''Builds a timestep tensor for the current batch.
        
        Args:
            x : The input image tensor, used to derive batch size for the newly
                created timesteps tensor.

        Returns:
            torch.Tensor : The timesteps tensor.
        '''
        B = x.shape[0]
        timesteps = torch.randint(
            0, self.config.model.noise_schedule.timesteps, (B,),
            device=self.device).long()
        return timesteps
    
    def _warm_up_lr(self) -> None:
        '''Warms up learning rate for the DAE's optimizer.
        
        The values used for the warm up are derived from the DAE learning rate
        settings in the input config.
        '''
        if self.current_iteration <= self.CFG_LR.warm_up_steps:
            steps = max(1, self.CFG_LR.warm_up_steps)
            lr = self.CFG_LR.base_rate * (self.current_iteration / steps)
            for param_group in self.model.opt_dae.param_groups:
                param_group['lr'] = lr

    def _decay_lr(self) -> None:
        '''Decays learning rate for the DAE's optimizer.
        
        The values used for the decay are derived from the DAE learning rate
        settings in the input config.
        '''
        if self.current_iteration > self.CFG_LR.steps_before_decay:
            steps = max(1, self.CFG_LR.decay_steps)
            current = self.current_iteration - self.CFG_LR.steps_before_decay
            lr = self.CFG_LR.base_rate * (1.0 - current / steps)
            for param_group in self.model.opt_dae.param_groups:
                param_group['lr'] = lr

    def backward(
            self, 
            loss: torch.Tensor, 
            max_norm: float=1.0,
            step: bool=True
        ) -> None:
        '''Backward step for the diffusion model.
        
        Handles the backward pass and, if applicable, gradient accumulation.
        The behavior of this method is heavily dependent on config settings.

        Gradient accumulation:
            If "accumulate_gradients" is enabled in the config file, the input
            "loss" is first divided by the gradient_accumulation_steps value 
            from the config.

        Full precision:
            - Computes gradients from input loss tensor.
            - If step=True:
                - Clips loss gradients to the value of "max_norm".
                - Steps DAE optimizer.

        Mixed precision:
            - Scales loss, then computs gradients.
            - If step=True:
                - Unscales loss gradients.
                - Clips gradients to the value of "max_norm".
                - Steps DAE optimizer with gradient scaler.
                - Updates gradient scaler.

        EMA update:
            If step=True and "use_ema" is enabled in the input config, this
            this method will also update the EMA module.

        Args:
            loss : The loss tensor for the current iteration.
            max_norm : The maximum value to allow when clipping the gradients.
                Helps to stabilize training, especially when training with
                mixed precision.
            step : Whether to step the model backward. See above.
        '''
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
        '''Asserts all input tensors are finite.
        
        Slows down training, only enable for debugging! Useful for tuning
        settings for mixed precision training.

        Args:
            x : The input image tensor.
            x_t : The timestep-embedded noise tensor for the given iteration.
            noise : The real noise tensor for the given iteration.
        '''
        assert torch.isfinite(x).all()
        assert torch.isfinite(x_t).all()
        assert torch.isfinite(noise).all()

    ##### TRAINING HOOKS #####

    def on_epoch_start(self, **kwargs: Any) -> None:
        '''Epoch start callback for DiffusionTrainer.
        
        In this Trainer class, this is just used to clear the "batch_times" 
        array used to calculate average batch times. 
        
        See "Trainer.on_epoch_start()" for more info.
        '''
        self.model.batch_times.clear()

    def train_step(
            self,  
            x: torch.Tensor, 
            y: torch.Tensor | None=None, 
            idx: int=None,
            assert_finite: bool=False,
            **kwargs: Any
        ) -> None:
        '''Train step callback for DiffusionTrainer.

        1.) Warms up/ decays learning rate (if enabled)
        2.) Runs model "q_sample()" method to generate base and
            timestep embedded noise.
        3.) Predicts noise at timestep from DAE.
        4.) Calculates loss from base noise and predicted noise.
        5.) Run backward pass (see "DiffusionTrainer.backward()")
        6.) Exports examples, saves checkpoints, displays results.
        
        Args:
            x : The input image (or latent) tensor for the current batch.
            y : The context tensor for the current batch, if applicable.
            idx : The index of the current batch from the training loop. Not
                used in this child class currently.
            assert_finite : If True, will run a check which asserts that the x,
                x_t, and noise tensors are finite before stepping the 
                optimizer. See "DiffusionTrainer._assert_finite()" for more
                information.
        '''
        self.current_iteration += 1
        if self.CFG_LR.warm_up: self._warm_up_lr()
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
        if self.print_avg_loss: 
            self.loss_tracker.append_loss_value(loss_G_MSE.mean().item())
        
        step = not self.CFG_M.accumulate_gradients or \
           self.current_iteration % self.CFG_M.gradient_accumulation_steps == 0
        self.backward(loss_G_MSE, step)
        if step: self.model.opt_dae.zero_grad()

        if self.current_iteration == self.total_steps:
            self.export_examples(context=y)
            self.save_checkpoint()
            print('Training complete!')
            exit(0)

        if x.shape[1] == 3:
            x = torch.cat([x, torch.full_like(x[:, :1, :, :], 0.0)], dim=1)
        self._display(x, x_t, predicted, timesteps)
        if self.current_iteration % self.config.save.example_save_rate == 0: 
            self.export_examples(context=y)    
        if self.config.save.save_model and \
           self.current_iteration % self.config.save.model_save_rate == 0:
            self.save_checkpoint()
        
    def on_epoch_end(self, **kwargs: Any) -> None:
        '''Epoch end callback for DiffusionTrainer.
        
        Does nothing here. See "Trainer.on_epoch_end()" for more info.
        '''
        pass

    ##### DIFFUSION TRAINING LOOP #####

    # def train_diffusion(
    #         self,
    #         epoch:int,
    #         on_epoch_start: Callable[[], None] | None=None,
    #         train_step: Callable[[torch.Tensor, torch.Tensor, int], None] | 
    #         None=None,
    #         on_epoch_end: Callable[[], None] | None=None,
    #         multithreaded: bool=True,
    #         callback_kwargs: dict[str, dict[str, Any]] = {}
    #     ) -> None:
    #     '''Diffusion model training method.

    #     See "Trainer.train_paired()". This will be changed in the future. Most
    #     likely, the base "Trainer" class will use a generalized "trainer_core"
    #     method which can be overridden directly by the child classes, as it is
    #     here with the various diffusion model classes.
    #     '''
    #     if self.config.train.load.continue_train:
    #         self.current_epoch = 1 + epoch + self.config.train.load.load_epoch
    #     else: self.current_epoch = epoch + 1

    #     start_fn = on_epoch_start or self.on_epoch_start
    #     train_fn = train_step or self.train_step
    #     end_fn = on_epoch_end or self.on_epoch_end

    #     start_time = time.perf_counter()
        
    #     start_fn(**callback_kwargs.get('on_epoch_start', {})) 
    #     if multithreaded and self.config.visualizer.visdom.enable: 
    #         try:
    #             self.vis.start_thread()
    #             self.model._trainer_core(
    #                 train_fn, callback_kwargs.get('train_step', {}))
    #         except KeyboardInterrupt:
    #             sys.exit('Interrupt Recieved: Stopping training...')
    #         finally: self.vis.stop_thread()
    #     else: self.model._trainer_core(
    #         train_fn, callback_kwargs.get('train_step', {}))
        
    #     end_fn(**callback_kwargs.get('on_epoch_end', {})) 

    #     end_time = time.perf_counter()
    #     self.last_epoch_time = end_time-start_time


