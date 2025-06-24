# NectarGAN - Getting Started (CLI)
#### This section will walk you through the process of using NectarGAN from the command line.

> [!NOTE]
> Up to this point, my focus has primarily been on the Toolbox interface. As such, the CLI options currently are rather limited. Expanding CLI support is a high-priority task so this is likely to change significantly in the near future.

## Training
Pix2pix training can be performed via CLI in NectarGAN with the following command:
```bash
nectargan-train-paired -f /path/to/config.json -lss extended+vgg --log_losses
```
There are currently only three command line arguments, all of which are shown here. Following is a brief explanation of each:
### `-f` `--config_file`
**Type:** `string`<br> 
**Description:** The system path to the config file to use for training. If this flag is not set, the default config (`/nectargan/config/default.config`) will be used.
### `-lss` `--loss_subspec`
**Type:** `string`<br>
**Description:** What loss subspec to use when initializing the objective function (see [`pix2pix_objective`](/nectargan/losses/pix2pix_objective.py) for more info).
### `-log` `--log_losses`
**Type:** `action`, `store_true`<br>
**Description:** If this flag is present, the [`LossManager`](/docs/api/losses/lossmanager.md) will be set up to cache loss values and occasionally dump the cache to the loss log.

**So, in short, all the actual training data is still pulled from the config file (see [here](/docs/api/config.md)), as with any other type of training in NectarGAN.** This means you could duplicate the default config, set it up however you'd like, then just point the train script to it via the `-f` flag when running the `nectargan-train-paired`. 

It also keeps the actual CLI simple which I generally prefer, as now, only things which aren't present in the config, like `loss_subspec` (since it's just a drop-in objective function, adding it's parameters to the config is not necessary), or things which you might want to override need to have CLI arguments.

## Testing


## Utils