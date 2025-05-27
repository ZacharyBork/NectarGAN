import time
import json
import copy
import warnings
import pathlib
from os import PathLike
from typing import Any, Literal

import torch
import torch.nn as nn

from pix2pix_graphical.config.config_data import Config
from pix2pix_graphical.losses.lm_data import LMHistory, LMLoss
import pix2pix_graphical.losses.lm_specs as specs
class LossManager():
    def __init__(
        self, 
        config: Config,
        experiment_dir: PathLike,
        spec: Literal['pix2pix'] | None=None,
        subspec: str='basic',
        enable_logging: bool=True,
        history_buffer_size: int=50000
    ) -> None:
        '''Init function for the LossManager class.
        
        Args:
            config: A Config object, created either via a Trainer, or via 
                pix2pix_graphical.config.config_manager.ConfigManager().
            experiment_dir: System path to experiment output dir.
            spec: Specification with which to init the LossManager losses, or 
                None (default) to create an empty LossManager to register 
                losses with later.
            subspec: Modifier to specs, allowing you to easily add additional 
                functionality to init specifications. The pix2pix spec, for
                example, implements a 'basic' subspec for standard losses, and
                an 'extended' subspec for added structural loss functions.
            enable_logging: If True, this tool will export a JSON loss log to
                the experiment_dir. Loss values and weights will be stored over
                time, and can be dumped to the loss log with:
                    - LossManager.update_loss_log()
            history_buffer_size: The max length of each list (losses, weights)
                for any LMHistory object managed by the LossManager. This is a
                sort of fallback, the stored losses will be dumped to the loss
                log every history_buffer_size number of batches to free up the
                memory if LossManager.update_loss_log() is not called before
                the buffer is filled.  
        '''
        self.config = config
        self.device = self.config.common.device
        self.experiment_dir = pathlib.Path(experiment_dir)
        self.enable_logging = enable_logging
        self.buffer_size = history_buffer_size

        # Stores registered losses with str keys for easy lookup
        self.loss_fns: dict[str, LMLoss] = {}

        self.make_dummy_shape()
        self.init_from_spec(spec, subspec)

    def export_base_log(self) -> None:
        '''Creates a JSON file in the experiment_dir to save loss logs to.

        Note:
            If this funtion is called from a LossManager with a
            self.enable_logging of False, it will print a warning and return
            immediately.

        Raises:
            RuntimeError : In unable to save loss log.
        '''
        if not self.enable_logging: 
            warnings.warn(
                'Call to LossManager.export_base_log() with '
                '`enable_logging=False`. Bypassing.')
            return
        log_path = pathlib.Path(self.experiment_dir, 'loss_log.json')
        blank = { # Dummy log to define log JSON structure
            'LOSSMANAGER_LOG':{
                'device': f'{self.config.common.device}',
                'dataroot': f'{self.config.common.dataroot}',
                'experiment': f'{self.experiment_dir.name}',
                'loss_functions': {}}}
        
        try: # Try to save log file, raise any exceptions
            with open(log_path.as_posix(), 'w') as file:
                file.write(json.dumps(blank, indent=4))
        except Exception as e:
            message = f'Unable to save loss log to: {log_path.as_posix()}'
            raise RuntimeError(message) from e
        self.loss_log = log_path # Store log filepath as pathlib.Path

    def make_dummy_shape(self) -> None:
        '''Creates a dummy tensor of the correct model input shape.
        '''
        channels = self.config.common.input_nc  # Get input channels
        size = self.config.dataloader.crop_size  # Get input W, H
        self.dummy = torch.zeros((1, channels, size, size)).to(self.device)

    def init_from_spec(
        self,
        spec: Literal['core', 'pix2pix'],
        subspec: Literal['basic', 'extended'] = 'basic'
    ) -> None:
        '''Initialize loss manager losses from a given specification.

        This is a helper function to allow you to quickly initialize basic loss
        schemas from common specifications. If spec='core' (default if you init
        the LossManager without passing it an init_spec argument), it will just
        create an empty loss structure which you can then register your own 
        losses to.

        Args:
            spec : Specification used to initialize the LossManager losses.

        Raises:
            RuntimeError : If unable to init from spec.
            ValueError : If init spec is invalid.
        '''
        match spec:
            case None: self.loss_fns = {}
            case 'pix2pix': 
                try:
                    _spec = specs.pix2pix(self.config, self.dummy, subspec)
                    self.loss_fns = _spec
                    #self.register_pix2pix_losses(subspec=subspec)
                except Exception as e:
                    message = 'Unable to init from spec.'
                    raise RuntimeError(message) from e
            case _: raise ValueError(f'Invalid LossManager init_spec: {spec}')

    def _strip_loss_history(self, loss: LMLoss) -> LMLoss:
        '''Removes the loss history lists from a given loss.

        This is an internal function that is used by:

            - LossManager.get_registered_losses()

        Since get_registered_losses returns the raw LMLoss items, and
        those items store their loss history via an LMHistory, if you
        wanted to print the results of get_registered_losses, you would get a 
        bunch of unusably long lists of unformatted floats polluting the info
        that's actually useful. This function just strips the loss history from 
        the input LMLoss to avoid that and reduce overhead when 
        passing the LMLoss around.

        Args:
            loss : The loss to remove the history from.
        '''
        loss.history = LMHistory([], [])
        return loss

    def get_registered_losses(
        self, query: list[str] | None = None, strip=True
    ) -> dict[str, LMLoss]:
        '''Returns a dictionary of losses registered with the LossManager.

        This function returns a dict of all of the raw LMLosss that 
        are currently registered with the LossManager instance, and which have 
        a tag that matches any single item in the input query. The output 
        dictionary keys correspond to the internal names of the loss object, 
        set when registering a new loss with the LossManager. 

        If query=None (default), this function will return a dict of every 
        LMLoss that is currently registered.

        Args:
            query : List of tags to search for when requesting losses. Only
                registered losses which have a tag matching the query will be
                returned.

        Returns:
            dict : The requested losses.
        '''
        output = {}
        for loss_name, loss_container in self.loss_fns.items():
            tags = getattr(loss_container, 'tags', [])
            if query is None or any(tag in tags for tag in query):
                dupe = copy.copy(loss_container)
                if strip: output[loss_name] = self._strip_loss_history(dupe)
                else: output[loss_name] = dupe
        return output

    def get_loss_values(
        self,
        query: list[str] | None = None,
        precision: int = 2
    ) -> dict[str, float]:
        '''Returns all of the most recent loss values (Tensor.mean) as a dict.

        Note: This function uses the last value stored in LossManager.history 
        for each loss. As such, this function should generally be called AFTER 
        all of the registered loss functions have been run for the batch. 
        Calling it before running some or all of the loss funtions could lead 
        to unexpected results.

        Args:
            query : List of tags to search for when requesting losses. Only
                the values of registered losses which have a tag matching the 
                query will be returned.
            precision : Rounding precision of the returned loss values.
        Returns:
            dict : The requested loss values, mapped by their loss name.
        '''
        fns = self.get_registered_losses(query, strip=False)
        return {name: round(fns[name].history.losses[-1], precision)
            for name in fns}

    def get_loss_tensors(
        self,
        query: list[str] | None = None
    ) -> dict[str, torch.Tensor]:
        '''Returns the queried LMLoss's last_loss_map tensors.

        Note: This function uses the last value stored in self.last_loss_map. As 
        such, this function should generally be called AFTER all of the 
        registered loss functions have been run for the batch. Calling it 
        before running some or all of the loss funtions could lead to 
        unexpected results.

        Args:
            query : List of tags to search for. Only the tensors of registered 
                losses which have a tag matching the query will be returned.
        Returns:
            dict : The requested loss tensors, mapped by their loss name.
        '''
        fns = self.get_registered_losses(query)
        return {name: container.last_loss_map 
            for name, container in fns.items()}

    def print_losses(
        self,
        epoch: int,
        iter: int,
        precision: int = 2,
        capture: bool = False
    ) -> str | None:
        '''Prints (or returns) a string of all the most recent loss values.

        Note: This function uses the last value stored in LossManager.history 
        for each loss. As such, this function should generally be called AFTER 
        all of the registered loss functions have been run for the batch. 
        Calling it before running some or all of the loss funtions could lead 
        to unexpected results.

        By default, this function will print a string of all registered losses 
        and their most recent values, tagged with epoch and iter, formatted as:

        "(epoch: {e}, iters: {i}) Loss: {L_1_N}: {L_1_V} {L_2_N}: {L_2_V} ..."

        Key:
            e : input epoch
            i : input iter
            L_X_N : Loss X name
            L_X_V : Loss X value

        If capture=True, however, the function will return the above formatted 
        string rather than printing it. 

        If you would prefer to get the loss values as a dict, please see: 
            - LossManager.get_loss_values()

        If you are trying to access the registered LMLoss items, see: 
            - LossManager.get_registered_losses()

        Args:
            epoch : The current epoch value. training_loop_iter+1 in the 
                default train script.
            iter : The current iteration.
            precision : The rounding precision of the loss values as they are 
                added to the output string.
            capture : If true, function will return loss values string instead 
                of printing.
        '''
        losses = self.get_loss_values(precision=precision)
        output = f'(epoch: {epoch}, iters: {iter}) Loss:'
        for loss in losses:
            output += f' {loss}: {losses[loss]}'
        if not capture:
            print(output)
            return None
        else:
            return output

    def register_loss_fn(
            self,
            loss_name: str,
            loss_fn: nn.Module,
            loss_weight: float = 1.0,
            tags: list[str] = []) -> None:
        '''Registers a loss function with the loss manager.

        A number of LossManager functions rely on their loss_name being unique, 
        so losses can be easily added, removed, grouped, or shifted around, 
        while ensuring that we maintain an immutable lookup value internally. 
        As such, this funtion will raise a ValueError if the requested 
        loss_name is already assignedto a registered loss, including one 
        created by LossManager.init_from_spec(). You can check what losses are
        currently registered via: 

            - LossManager.get_registered_losses().

        Args:
            loss_name : Name of the loss function (e.g. 'G_GAN', 'D_real').
                This name will be the key that is used to interact with the 
                loss function once it is registered with the LossManager.
            loss_fn : The loss function to register with the LossManager. The 
                loss function can be basically any type of nn.Module as long as
                it has a forward function that returns a torch.Tensor.
            loss_weight : Lambda value for the loss function.
            tags : Tags for the new loss value, used to query the loss objects
                in various forms.

        Raises:
            ValueError : Requested loss_name is already in use.
        '''
        if not loss_name in self.loss_fns:
            self.loss_fns[loss_name] = LMLoss(
                name=loss_name,
                function=loss_fn.to(self.device), 
                loss_weight=loss_weight,
                last_loss_map=self.dummy.clone(), 
                history=LMHistory([], []),
                tags=tags)
        else:
            raise ValueError(f'Loss function name already in use: {loss_name}')

    def _dump_to_log(self, name: str) -> None:
        '''Dumps loss values from LMLoss object to loss log.

        Note:
            If this funtion is called from a LossManager with a
            self.enable_logging of False, it will print a warning and return
            immediately.

        Args:
            name : The name of the loss function to update the logs of.
        
        Raises:
            RuntimeError : If unable to open log file.
        '''
        if not self.enable_logging: 
            warnings.warn(
                'Call to LossManager._dump_to_log() with '
                '`enable_logging=False`. Bypassing.')
            return
        log_path = self.loss_log
        try: # Try to get loss_log.json
            with open(log_path.as_posix(), 'r') as file:
                log_data = json.loads(file.read())
        except Exception as e:
            message = 'Unable to open loss log file: {}'
            raise RuntimeError(message.format(log_path.as_posix())) from e
        
        # Find the requested loss value in the loss log
        root, history = 'LOSSMANAGER_LOG', 'loss_functions'
        history_logs = log_data.setdefault(root, {}).setdefault(history, {})
        log_entry = history_logs.setdefault(name, {'loss': [], 'weights': []})
        
        # Update the loss log entries
        loss_fn = self.loss_fns[name]
        log_entry['loss'].extend(loss_fn.history.losses)
        log_entry['weights'].extend(loss_fn.history.weights)

        # Dump the log back to loss_log.json
        with open(log_path.as_posix(), 'w') as file:
            json.dump(log_data, file)

        # Clear the LMLoss's LMHistory
        self.loss_fns[name].history = LMHistory(losses=[], weights=[])

    def _add_loss_to_history(
            self, 
            loss_fn: LMLoss,
            loss_value: torch.Tensor
        ) -> None:
        '''Adds loss value and weight to history.
        '''
        # Always append loss value info
        loss_fn.history.losses.append(loss_value.mean().item())
        if not loss_fn.loss_weight is None:
            # But loss function might not have associated weight
            loss_fn.history.weights.append(loss_fn.loss_weight)

    def _append_loss_history(
            self, 
            name: str, 
            value: torch.Tensor,
            silent: bool=True
        ) -> None:
        '''Internal function to append values to loss histories.

        This function will append the value and weight of the loss queried with 
        `name` to the LMLoss's history. Before appending new loss values, this 
        funtion first does a safety check to make sure the current LMLoss's 
        history isn't >= max buffer size. If it is, it will first dump the 
        buffer to the loss log.

        Args:
            name : Name of the loss object which houses the LMHistory you want 
                to dump the output values of.
            value : The torch.Tensor result of the corresponding loss function.

        Raises:
            ValueError : If silent=False and the requested LMloss's history is 
                greater than self.buffer_size.
        '''
        loss_fn = self.loss_fns[name] # Lookup requested loss function
        if len(loss_fn.history.losses) >= self.buffer_size:
            if silent: # Dump to log if history >= buffer size
                self._dump_to_log(name) 
            else:
                message = f'Loss log full! Limit: {self.buffer_size}'
                raise RuntimeError(message)
        self._add_loss_to_history(loss_fn, value)

    def _swap_loss_values(
            self, 
            name: str, 
            value: torch.Tensor,
        ) -> None:
        '''Internal function to swap most recent value in loss histories.

        This function will clear the LMLoss's history, then add the latest 
        value and weight to their respective history lists. This keeps it 
        simple with the datclasses and lookup functions, all the histories are 
        list based, it's just that if logging is turned off, the lists only 
        save the latest entry.

        Args:
            name : Name of the loss object which houses the LMHistory you want 
                to dump the output values of.
            value : The torch.Tensor result of the corresponding loss function.
        '''
        loss_fn = self.loss_fns[name] # Lookup requested loss function
        loss_fn.history = LMHistory([], []) # Clear old loss value
        self._add_loss_to_history(loss_fn, value) # Add new loss values


    def update_loss_log(self, silent: bool=True) -> None:
        '''Exports loss history to loss log.

        This is a public wrapper for LossManager._dump_to_log(). It loops
        through each loss registered with the LossManager instance and dumps
        their loss history to loss_log.json. It then clears the lists used to
        buffer loss history to free up the memory.

        This should be generally be called once per epoch, after the training
        loop, to dump the loss values for the epoch and free the history for 
        the next epoch. If this is not called, though, the loss history will be 
        automatically dumped to the loss log if the corresponding loss function
        is run and len(history)>=self.buffer_size.

        Note:
            If this funtion is called from a LossManager with a
            self.enable_logging of False, it will print a warning and return
            immediately.

        Args:
            silent: If True (default), this function will not print anything to
                the console. If False, it will print execution time.
        '''
        if not self.enable_logging: 
            warnings.warn(
                'Call to LossManager.update_loss_log() with '
                '`enable_logging=False`. Bypassing.')
            return
        dump_history = lambda : [self._dump_to_log(i) for i in self.loss_fns]
        match silent:
            case True: dump_history()
            case False:
                start = time.perf_counter()
                print(f'LossManager: Updating loss logs')
                dump_history()
                end = time.perf_counter()
                print(f'LossManager: Time taken: {end - start:.3f} seconds')

    def compute_loss_xy(
        self,
        loss_name: str,
        x: torch.Tensor | Any,
        y: torch.Tensor | Any
    ) -> torch.Tensor:
        '''Runs a loss function and returns the result.

        Also detaches a version of the loss result tensor and moves it to CPU, 
        then stores it in self.losses_{"G"|"D"}[loss_name][1]. Additionally, 
        this function also computes the mean values of the loss tensor and adds 
        it to the associated self.history entry.

        Args:
            loss_name : The name used when registering the losses, or the 
                registered name of an inbuilt loss function if LossManager was 
                initialized from spec. A list of the losses currently 
                registered with the LossManagercan be accessed with: 

                    - LossManager.get_registered_losses().

            x : First input for the loss function, usually a torch.Tensor.
            y : Second input for the loss function, usually a torch.Tensor.

        Returns:
            torch.Tensor : Computed loss value.

        Raises:
            KeyError : If the loss name does not match any registered loss.
        '''
        try:
            loss_entry = self.loss_fns[loss_name]
        except KeyError as e:
            message = ('Invalid loss type. Valid types are: {}')
            raise KeyError(message.format(list(self.loss_fns))) from e

        # First, compute loss value
        loss_value = loss_entry.function(x, y)
        if not loss_entry.loss_weight is None:
            loss_value *= loss_entry.loss_weight
        # Then update the corresponding last_loss_map tensor
        self.loss_fns[loss_name].last_loss_map = loss_value.clone()
        self.loss_fns[loss_name].last_loss_map.detach().cpu()
        # Then update loss history
        if self.enable_logging:
            self._append_loss_history(loss_name, loss_value)
        else: self._swap_loss_values(loss_name, loss_value)

        return loss_value  # Return original loss value tensor

    def compute_loss_xyz(self) -> torch.Tensor:
        '''Equivelent to compute_loss_xy but for 3 input loss function.
        '''
        raise NotImplementedError('Not yet implemented.')
