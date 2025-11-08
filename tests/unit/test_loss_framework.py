from pathlib import Path

import torch

from nectargan.config.config_manager import ConfigManager
from nectargan.losses.loss_manager import LossManager

def test_loss_manager():
    '''Tests the loss manager class.

    Builds a loss manager from the default config, registers a loss with it and
    checks that it was successful. Then evaluates the loss 10,000 times and 
    performs various sanity checks on the output.
    '''
    root = Path(__file__).parent.parent.parent.resolve()
    config_file = Path(root, 'tests/tmp/config.json')
    assert config_file.exists()

    config_manager = ConfigManager(config_file)
    loss_manager = LossManager(
        config=config_manager.data,
        experiment_dir=root,
        enable_logging=False)
    loss_manager.register_loss_fn(
        loss_name='test_loss', 
        loss_fn=torch.nn.L1Loss().to('cpu'), 
        loss_weight=1.0)
    assert len(loss_manager.get_registered_losses().keys()) > 0
    
    for i in range(10000):
        loss_manager.compute_loss_xy(
            loss_name='test_loss',
            x=torch.randn((1, 3, 64, 64)),
            y=torch.randn((1, 3, 64, 64)),
            epoch=i)
        
        # Validate loss mean value
        value = loss_manager.get_loss_values()['test_loss']
        assert value \
           and isinstance(value, float) \
           and value != float('nan') \
           and value != float('inf')

        # Validate raw loss tensor
        tensor = loss_manager.get_loss_tensors()['test_loss']
        assert tensor \
           and isinstance(tensor, torch.Tensor) \
           and not torch.isnan(tensor).any() \
           and not torch.isinf(tensor).any()
