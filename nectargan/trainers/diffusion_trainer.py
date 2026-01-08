import math
import random
import pathlib
from os import PathLike
from typing import Any, Literal
from contextlib import nullcontext, AbstractContextManager

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image
from torch_ema import ExponentialMovingAverage

from nectargan.trainers import Trainer
from nectargan.config import ConfigManager, DiffusionConfig
from nectargan.models.diffusion.data import AverageLossTracker
from nectargan.models import PixelDiffusionModel, LatentDiffusionModel
from nectargan.visualizer import DiffusionVisualizer
from nectargan.losses import VGGPerceptual, Sobel, Laplacian

class DiffusionTrainer(Trainer[DiffusionConfig]):
    def __init__(
            self, 
            config: str | PathLike | ConfigManager, 
            log_losses: bool = True,
            testing: bool = False
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
        self.testing = testing
        
        super().__init__(
            config=config, quicksetup=not self.testing, log_losses=log_losses)
        self.CFG_M = self.config.model
        self.CFG_LR = self.CFG_M.unet.learning_rate
        self.current_iteration = 0
        
        self.cuda = self.device == 'cuda'
        if self.cuda:
            cudnn = self.config.common.cudnn
            torch.backends.cudnn.benchmark = cudnn.benchmark
            torch.backends.cudnn.deterministic = cudnn.deterministic
        
        self._init_model()
        self._build_timesteps()
        if not self.testing:
            self._get_step_counts()
            self._validate_fixed_captions()
            self.register_losses()
            if self.CFG_M.use_ema: self._init_ema()
            self._load_checkpoints()
        
    ##### OPTIMIZATION #####

    def _flush_cache(self) -> None:
        if self.cuda: torch.cuda.empty_cache()

    ##### INIT #####                

    def _get_step_counts(self) -> None:
        if self.config.train.load.continue_train:
            self.current_step = 1 + self.config.train.load.load_step
        else: self.current_step = 1
        self.total_steps = self.CFG_LR.steps_before_decay
        if self.CFG_LR.do_decay: self.total_steps += self.CFG_LR.decay_steps

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
            case _: raise ValueError(f'Invalid model_type: {self.model_type}')
        self.model = model(config=self.config, testing=self.testing)
        self.trainer_core = self.model.trainer_core
        self.model.opt_unet.zero_grad()

        compile_cfg = self.config.model.unet.compile
        if compile_cfg.enable:
            self.model.unet = torch.compile(
                self.model.unet, mode=compile_cfg.mode)
        self._flush_cache()

    def init_visdom(self) -> None:
        '''Initializes Visdom visualization for diffusion training.'''
        vcon = self.config.visualizer
        if vcon.visdom.enable:
            self.vis = DiffusionVisualizer(
                env=vcon.visdom.env_name,
                server=vcon.visdom.server,
                port=vcon.visdom.port) 
            self.vis.clear_env()
    
    def _init_ema(self) -> None:
        '''Initializes a PyTorch EMA module.'''
        self.ema = ExponentialMovingAverage(
            self.model.unet.parameters(), 
            decay=self.config.model.ema_decay)
        for param in self.ema.shadow_params:
            param.data = param.data.to(self.device)
        
    ##### UTILS #####

    def get_dataset_length(self) -> int:
        '''Gets the length of the current dataset.'''
        return len(self.model.dataloader)
    
    def get_epoch_count(self) -> int:
        '''Calculates an epoch count based on dataset size and step count.'''
        # This is sort of a workaround to the framework relying so heavily on
        # training being measured in epochs, rather than steps as it is here.
        # This likely won't be necessary at some point in the future, once
        # more of the framework components have been generalized.
        return math.ceil(self.total_steps / max(1, self.get_dataset_length()))
    
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

    ##### CHECKPOINTS #####

    def _load_checkpoints(self) -> None:
        '''Loads model checkpoint(s) to continue training.'''
        if self.config.train.load.continue_train:
            self.load_checkpoint(
                'UNet', self.model.unet, self.model.opt_unet, 
                self.config.model.unet.learning_rate.base_rate, unit='step',
                value=self.config.train.load.load_step)
            if self.CFG_M.use_ema: 
                self.load_checkpoint(
                    'EMA', self.ema, unit='step',
                    value=self.config.train.load.load_step)
            self._flush_cache()

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
            net : A human-readable tag for the Module being saved (i.e. "UNet",
                "EMA"). Will be used to name the checkpoint file.

        Raises:
            RuntimeError : If unable to save checkpoint file.
        '''
        checkpoint = { 'state_dict': mod.state_dict() }
        if not opt is None: checkpoint['optimizer'] = opt.state_dict()
        
        name = f'step{str(self.current_step)}_net{net}.pth.tar'
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
        path = self.export_model_weights(
            self.model.unet, self.model.opt_unet, 'UNet')
        output = f'Checkpoint Saved (UNet): {path}'
        if self.CFG_M.use_ema:
            path = self.export_model_weights(self.ema, None, 'EMA')
            output += f'\nCheckpoint Saved (EMA): {path}'
        if not capture: print(output)
        else: return output

    ##### LOSS #####

    def _init_loss_tracker(self) -> None:
        vis = self.config.visualizer
        self.print_avg_loss = vis.console.average_loss
        self.visdom_avg_loss = vis.visdom.average_loss
        self.loss_tracker = AverageLossTracker(
            update_freq_visdom=vis.visdom.update_frequency,
            update_freq_console=vis.console.print_frequency)

    def register_losses(self) -> None:
        '''Registers loss functions for training.'''
        cfg = self.config.train.loss
        cfg_p = cfg.pixel
        self.do_pixel_loss = (
            cfg_p.lambda_l1
          + cfg_p.lambda_vgg
          + cfg_p.lambda_sobel
          + cfg_p.lambda_laplacian
        ) != 0.0

        print(f'Lambda MSE: {cfg.lambda_mse}')
        self._init_loss_tracker()
        self.loss_manager.register_loss_fn(
            loss_name='G_MSE', loss_fn=nn.MSELoss().to(self.device), 
            loss_weight=cfg.lambda_mse, tags=['G'])
        self.loss_tracker.loss_values['G_MSE'] = []

        if self.do_pixel_loss:
            self.pixel_losses = [
                ('L1', cfg_p.lambda_l1, nn.L1Loss),
                ('VGG', cfg_p.lambda_vgg, VGGPerceptual),
                ('Sobel', cfg_p.lambda_sobel, Sobel),
                ('Laplacian', cfg_p.lambda_laplacian, Laplacian)]
            for x in self.pixel_losses:
                if x[1] > 0.0:
                    print(f'Lambda {x[0]}: {cfg_p.lambda_l1}')
                    self.loss_manager.register_loss_fn(
                        loss_name=f'G_{x[0]}', loss_fn=x[2]().to(self.device), 
                        loss_weight=x[1], tags=['G'])
                    self.loss_tracker.loss_values[f'G_{x[0]}'] = []
        
    def _compute_pixel_loss(
            self, 
            x: torch.Tensor, 
            x_t: torch.Tensor, 
            predicted: torch.Tensor
        ) -> torch.Tensor:
        max_batches = self.config.train.loss.pixel.max_batches
        batches = min(max_batches, max(1, x.shape[0]))
        pixel_loss = torch.zeros(1, device=self.device)
        sums = {}
        
        for batch in range(batches):
            pred_x0 = self._predict_x0(
                x_t[batch:batch+1], predicted[batch:batch+1])
            pred_x0 = self.model.decode(pred_x0)
            pixel_x = self.model.decode(x[batch:batch+1])   
            
            for loss in self.pixel_losses:
                if loss[1] > 0:
                    value += self.loss_manager.compute_loss_xy(
                        loss[0], pred_x0, pixel_x, self.current_step)
                    if loss[0] in sums.keys(): sums[loss[0]] += value.item()
                    else: sums[loss[0]] = value.item()
                    pixel_loss += value

        pixel_loss = pixel_loss / batches
        if self.print_avg_loss: 
            for key, loss in sums.items():
                self.loss_tracker.append_loss_value(key, loss / batches)
        return pixel_loss

    ##### EXAMPLES #####

    def _get_example_captions(self) -> list[str]:
        if self.config.captions.use_fixed_captions:
            return self.config.captions.fixed_captions 
        else: return self.model.captions

    def _get_example_context(
            self, 
            context: torch.Tensor,
            captions: list[str]
        ) -> torch.Tensor:
        caption = 'nullcaption'
        if not captions is None and not context is None:
            context, _ = self.model.text_encoder(captions)
            count = float(len(captions))
            if count > 0.0:
                idx = int(math.floor(random.random() * count))
                context = context[idx].unsqueeze(0).detach()
                context = context.to(self.device)
                caption = captions[idx]
        print(f'Running inference with caption:\n{caption}')

    def _save_inference_example(
            self,
            idx: int,
            output_directory: PathLike,
            batches: int = 1,
            latent_spatial_size: int | None = None,
            context: torch.Tensor | None = None,
            cfg_scale: float = 7.5,
            inference_steps: int = 100,
            mode: Literal['DDPM', 'DDIM'] = 'DDIM'
        ) -> None:
        with self.ema_context():
            with self.autocast_context(), torch.no_grad():
                self._flush_cache()
                output = self.model.sample(
                    batches=batches, latent_spatial_size=latent_spatial_size,
                    context=context, cfg_scale=cfg_scale, 
                    mode=mode, inference_steps=inference_steps)
                
                
        output = torch.clamp((output + 1) * 0.5, 0.0, 1.0)
        scale_tag = str(cfg_scale).replace('.', '-')
        for i in range(batches):
            name = (
                f'step{self.current_step}_idx{idx+1}_'
                f'b{i+1}_cfg{scale_tag}.png')
            filepath = pathlib.Path(output_directory, name).resolve()
            save_image(output[i], filepath.as_posix())

    def export_examples(
            self, 
            context: torch.Tensor | None=None
        ) -> None: 
        '''Evals UNet and exports resulting images to the experiment directory.
        
        Args:
            context : The context Tensor to pass to the diffusion model, or
                None if not using text conditioning.
        '''
        smp = self.config.model.sampling
        was_training = self.model.training
        self.model.eval()
        try: 
            captions = self._get_example_captions()
            for i in range(self.config.save.num_examples):
                if self.config.captions.use_captions:
                    context = self._get_example_context(context, captions)
                    cfg_scales = self.config.save.sample_cfg_scales
                else: 
                    context = None
                    cfg_scales = [1.0]
                for scale in cfg_scales:
                    self._save_inference_example(
                        idx=i, output_directory=self.examples_dir, 
                        context=context, cfg_scale=scale, 
                        inference_steps=smp.ddim_timesteps, mode=smp.function)
        finally: self.model.train(was_training)
        
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
        visdom = self.config.visualizer.visdom
        if not visdom.enable: return

        if visdom.display_images:
            M = visdom.max_images
            x, y, z = x[:M], y[:M], z[:M] 

            if self.config.model.model_type == 'latent':
                x, y, z = self._build_latent_vis_tensors(
                    x, y, z, self.timesteps)
                
            self.vis.update_images(
                x=x, y=y, z=z, title='x | x_t | pred x0', 
                image_size=visdom.image_size)
            
        if self.visdom_avg_loss:
            _get = lambda i: self.loss_tracker.get_average(i, 'visdom')
            loss = { 'G_MSE': _get('G_MSE') }
            if self.do_pixel_loss:
                for loss in self.pixel_losses:
                    if loss[1] == 0.0: continue
                    loss[loss[0]] = _get(loss[0])
        else: loss = self.loss_manager.get_loss_values(query=['G'])
        self.vis.update_loss_graphs(self.current_step, loss)

    def _predict_x0(
            self, 
            x_t: torch.Tensor, 
            predicted: torch.Tensor,
            range: float=4.0
        ) -> torch.Tensor:
        params = self.model.noiseparams
        abar = params.alphas_cumprod[self.timesteps].view(-1,1,1,1)
        pred_x0 = (x_t - (1.0 - abar).sqrt() * predicted) /  abar.sqrt()
        return torch.clamp(pred_x0, -range, range)

    def _build_latent_vis_tensors(
            self, 
            x: torch.Tensor,
            x_t: torch.Tensor,
            predicted: torch.Tensor
        ) -> tuple[torch.Tensor]:
        '''Decodes tensors from latent space for viewing in Visdom.

        Args:
            x : The input image tensor.
            x_t : The noise tensor from the given timestep.
            z : The predicted clean image tensor from the UNet.

        Returns : The input tensors decoded from latent space as a tuple,
            ordered as: (x, x_t, predicted)
        '''
        with self.autocast_context():
            pred_x0 = self._predict_x0(x_t, predicted)
            tensors = [self.model.decode(t) for t in (x, x_t, pred_x0)]
        return tuple(tensors)

    ##### CONSOLE #####

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
            average = self.loss_tracker.get_average('G_MSE', 'console')
            output += f'Average Loss: MSE: {average} '
            if self.do_pixel_loss:
                for loss in self.pixel_losses:
                    if loss[1] == 0.0: continue
                    average = self.loss_tracker.get_average(loss[0], 'console')
                    output += f'{loss[0]}: {average} '
        if not capture: print(output)
        else: return output

    ##### DISPLAY TRAINING DATA #####

    def _display(
            self,
            x: torch.Tensor,
            x_t: torch.Tensor,
            predicted: torch.Tensor
        ) -> None:
        '''Updates Visdom and prints loss values to the console.

        This function reads the update frequencies for console and Visdom from
        the current config, and will only update each if it is appropriate to
        do so on the current iteration.
        
        Args:
            x : The input image tensor.
            x_t : The noise tensor from the given timestep.
            z : The predicted clean image tensor from the UNet.
        '''
        vis = self.config.visualizer
        if self.current_step % vis.console.print_frequency == 0:
            avg_time = sum(self.model.batch_times) 
            avg_time /= max(1, len(self.model.batch_times))
            avg_time = round(avg_time, 3)
            self.print_losses(self.current_step)
            print(f'Average batch time: {avg_time} seconds')
            self.model.batch_times.clear()
        if self.current_step % vis.visdom.update_frequency == 0:
            self.update_display(x, x_t, predicted)

    ##### TRAINING METHODS #####

    def _build_timesteps(self) -> None:
        '''Builds a timestep tensor for the current batch.'''
        B = self.config.dataloader.batch_size
        self.timesteps = torch.randint(
            0, self.config.model.noise_schedule.timesteps, (B,),
            device=self.device).long()
    
    def _warm_up_lr(self) -> None:
        '''Warms up learning rate for the UNet's optimizer.
        
        The values used for the warm up are derived from the UNet learning rate
        settings in the input config.
        '''
        if self.current_step <= self.CFG_LR.warm_up_steps:
            steps = max(1, self.CFG_LR.warm_up_steps)
            lr = self.CFG_LR.base_rate * (self.current_step / steps)
            for param_group in self.model.opt_unet.param_groups:
                param_group['lr'] = lr

    def _decay_lr(self) -> None:
        '''Decays learning rate for the UNet's optimizer.
        
        The values used for the decay are derived from the UNet learning rate
        settings in the input config.
        '''
        if self.current_step > self.CFG_LR.steps_before_decay:
            steps = max(1, self.CFG_LR.decay_steps)
            current = self.current_step - self.CFG_LR.steps_before_decay
            lr = self.CFG_LR.base_rate * (1.0 - current / steps)
            for param_group in self.model.opt_unet.param_groups:
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
            "loss" is first divided by the "gradient_accumulation_iterations"
            value from the config.

        Full precision:
            - Computes gradients from input loss tensor.
            - If step=True:
                - Clips loss gradients to the value of "max_norm".
                - Steps UNet optimizer.

        Mixed precision:
            - Scales loss, then computs gradients.
            - If step=True:
                - Unscales loss gradients.
                - Clips gradients to the value of "max_norm".
                - Steps UNet optimizer with gradient scaler.
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
            loss /= float(self.CFG_M.gradient_accumulation_iterations)
        if self.CFG_M.mixed_precision:
            self.model.g_scaler.scale(loss).backward()
            if step:
                self.model.g_scaler.unscale_(self.model.opt_unet)
                torch.nn.utils.clip_grad_norm_(
                    self.model.unet.parameters(), max_norm)
                self.model.g_scaler.step(self.model.opt_unet)
                self.model.g_scaler.update()
        else:
            loss.backward()
            if step:
                torch.nn.utils.clip_grad_norm_(
                    self.model.unet.parameters(), max_norm)
                self.model.opt_unet.step()
        if step and self.CFG_M.use_ema: self.ema.update()

    def _step(
            self,
            x: torch.Tensor,
            y: torch.Tensor | None,
            x_t: torch.Tensor,
            predicted: torch.Tensor
        ) -> None:
        self.model.opt_unet.zero_grad()
        self.current_step += 1

        if self.current_step == self.total_steps:
            self.export_examples(context=y)
            self.save_checkpoint()
            print('Training complete!')
            exit(0) # Should make this a more graceful exit eventually

        cfg = self.config.save
        if x.shape[1] == 3:
            x = torch.cat([x, torch.full_like(x[:, :1, :, :], 0.0)], dim=1)
        self._display(x, x_t, predicted)
        if self.current_step % cfg.example_save_rate == 0: 
            self.export_examples(context=y)    
        if cfg.save_model and self.current_step % cfg.model_save_rate == 0:
            self.save_checkpoint()

    ##### TRAINING HOOKS #####

    def on_epoch_start(self, **kwargs: Any) -> None:
        '''Epoch start callback for DiffusionTrainer.
        
        See "Trainer.on_epoch_start()" for more info.
        '''
        pass

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
        3.) Predicts noise at timestep from UNet.
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
        self._flush_cache()

        self.current_iteration += 1
        step = not self.CFG_M.accumulate_gradients or self.current_iteration %\
           self.CFG_M.gradient_accumulation_iterations == 0
                        
        if self.CFG_LR.warm_up: self._warm_up_lr()
        if self.CFG_LR.do_decay: self._decay_lr()
        self._build_timesteps()

        with self.autocast_context(): 
            x_t, noise = self.model.q_sample(x=x, t=self.timesteps)            
            if not self.CFG_M.mixed_precision:
                x_t = x_t.to(device=self.device, dtype=torch.float32)
                noise = noise.to(device=self.device, dtype=torch.float32)
                if y is not None and torch.is_floating_point(y):
                    y = y.to(self.device, dtype=torch.float32)
            
            predicted = self.model.unet(x_t, self.timesteps, context=y)
            loss = self.loss_manager.compute_loss_xy(
                'G_MSE', predicted, noise, self.current_step)
            
            if self.print_avg_loss: 
                self.loss_tracker.append_loss_value(
                    'G_MSE', loss.mean().item())

            if self.do_pixel_loss and self.current_step % \
               self.config.train.loss.pixel.frequency == 1:
                loss += self._compute_pixel_loss(x, x_t, predicted)

        if assert_finite: self._assert_finite(x, x_t, noise)
        self.backward(loss, step)
        if step: self._step(x, y, x_t, predicted)
        
    def on_epoch_end(self, **kwargs: Any) -> None:
        '''Epoch end callback for DiffusionTrainer.
        
        Does nothing here. See "Trainer.on_epoch_end()" for more info.
        '''
        pass


