# NectarGAN - Getting Started (Diffusion)
> [`Getting Started - Home`](../getting_started.md)

**NectarGAN includes prebuilt pixel and latent-space diffusion models.** Currently, these models can only be interacted with using CLI, or by via the API with your own custom training script. They will be added to NectarGAN Toolbox at some point in the future.

**_PLEASE NOTE:_ These models are in beta currently. You may find bugs or things which do not work correctly, and the functionality of these models is likely to change in the future. If you do come across something which is throwing exceptions or which you feel is not working as intended, please feel free to open a GitHub issue detailing your problem with steps to reproduce.**

**Additionally, the documentation for these models may be incomplete or outdated in some areas. Much of the documentation for the API components related to the diffusion models does not exist yet, but will come in a future release. _ALL API CODE HAS THOROUGH INLINE DOCUMENTATION,_ if you would like to know how a given API component works, please refer to the docstrings for the given component.**

**If you come across any issues with the documentation, please feel free to open an issue, or if you are comfortable doing so, a PR with your updates. Help is always appreciated!**

## Install

Please see the NectarGAN [getting started guide](https://github.com/ZacharyBork/NectarGAN/blob/main/docs/getting_started.md) for steps on how to install NectarGAN and configure your environment.

Using the NectarGAN diffusion models also requires some additional dependencies. To install all the required diffusion dependencies, after working through the getting started guide, please navigate to the NectarGAN root directory in your terminal and run this command:

```bash
python -m pip install ".[diffusion]"
```

If you intend to use the dataset streaming functionality (**BETA**), please also run:
```bash
python -m pip install ".[dataset_streaming]"
```

## The Models

### Pixel Diffusion Model

A simple pixel-space diffusion model, based largely on the [Denoising Diffusion Probabilistic Models (DDPM) (Ho et al., 2020)](https://arxiv.org/pdf/2006.11239).

### Latent Diffusion Model

A lightweight latent-space diffusion model. The architecture is based heavily on [Stable Diffusion](https://arxiv.org/pdf/2112.10752), but it is not as compute intensive. It uses the same self-attention pattern as SD, and has a deeper (configurable) middle layer, but does not incorporate the additional residual blocks on the shallower layers of the encoder/decoder path. It instead just uses a single identity connection inside of each of the conv blocks. This allows the model to be trained to convergence relatively quickly on lower end hardware (a single 2080Ti, in my case), but limits the model to largely single-domain use cases.

Eventually, I will add the ability to add back the additional residual blocks, likely in a configurable fashion so that you can decide where exactly to allocate the blocks to best suit your use case. Right now, though, I am a little compute limited, and likely would not be able to personally validate a model as heavy as Stable Diffusion.

This model also allows the use of [DDIM sampling](https://arxiv.org/pdf/2010.02502), in addition to DDPM.

## Using the models
Training and testing of diffusion models can be performed from the command line. 

### Training
When training diffusion models, all training settings will be pulled from the Diffusion config (outlined below). To begin a training session, first navigate to the NectarGAN root directory in your console, then run:

```bash
python -m nectargan.start.training.diffusion
```

### Testing
Testing of trained diffusion models can also be run from the command line using the following command:

```bash
python -m nectargan.start.testing.diffusion -e "/path/to/directory/of/model/to/test" -l 100
```
This command has a number of arguments which can be used to alter the testing behavior. These are:

| Argument | Description |
| --- | --- |
| `-e`, `--experiment_directory` | **REQUIRED.** The directory of the experiment to load for testing. |
| `-l`, `--load_step` | **REQUIRED.** The checkpoint step number to load for testing. |
| `-f`, `--config_file` | The system path to config file to use for testing. If not provided, the script will instead look for the most recent config file located in the gived experiment directory. |
| `-i`, `--test_iterations` | The number of test iterations to run. |
| `-b`, `--batches` | The number of batches to use in each iteration. |
| `-s`, `--latent_spatial_size` | The spatial size of the latent-space noise tensors used as input. |
| `-m`, `--inference_mode` | What inference mode to use (i.e. `DDIM`, `DDPM`) |
| `-is`, `--inference_steps` | The number of denoising steps to use for inference. |
| `-c`, `--caption` | The caption to use for text conditioning during inference, if applicable. |
| `-cfg`, `--cfg_scale` | The classifier free guidance scale to use for inference if using text conditioning. |
| `-ema`, `--sample_ema` | Whether to sample directly from the model's weights, or from the EMA weights. Requires an EMA checkpoint for the given step to be present in the experiment directory. |

## The Diffusion Config
The diffusion models use a different configuration file than the GAN models. The file can be found at:

[`nectargan/config/defaults/diffusion.json`](https://github.com/ZacharyBork/NectarGAN/blob/main/nectargan/config/defaults/diffusion.json)

Many of the options are mirrored from the default [GAN config](../api/config.md), but there are some new things to make note of. 

1. **All values in the diffusion config are based on optimizer steps, not epoch,** like they are with the GAN config.
2. Currently, augementations do not work. That will change in the future, but for now, the augmentations section of the config file can be ignored.

Now, we will briefly go over each config section to see what the settings do.

### `common`

| Settings | Description |
| --- | --- |
| `device` | What PyTorch device to use. Options are [`cuda`, `cpu`]. |
| `gpu_ids` | Currently this does nothing. Once multi-GPU is supported, this will allow you to define the GPUs to use for training and testing. |
| `cudnn` | This is a dict-like object containing a few settings related to CUDNN optimization for GPU training. `benchmark` will enable CUDNN benchmarking to reduce VRAM overhead. CUDNN will automatically benchmark various configuration settings to find the ones which will run best on the host system. `deterministic` will enable determinitic mode for CUDNN, allowing for more reproducable runs at the small compute cost. |
| `output_directory` | The root output directory to use. New subdirectories will be created in this root directory for each experiment / version. |
| `experiment_name` | The name for the experiment. This will be used to name the output subdirectory. |
| `experiment_version` | The version of the current experiment to run / load. |

### `train`
The `train` config section is broken down in to a couple smaller subsections. There are:

#### `load`

| Settings | Description |
| --- | --- |
| `continue_train` | If true, enables loading of checkpoint files to continue training. |
| `load_step` | The checkpoint step to load. |

#### `loss`

Generally, diffusion models only use MSE for UNet-based noise prediction. `lambda_mse` defines the weighting factor for the MSE loss. NectarGAN also allows you to apply pixel-space loss even when training latent-space models. This comes at a significant computational overhead, however, and likely will not improve the model's performance, but they can be fun to experiment with. The settings for these losses are defined in the `pixel` subcategory:

| Settings | Description |
| --- | --- |
| `frequency` | The frequency (in steps) to apply pixel-space losses. Since these are quite expensive, you may not want to apply them every step. A value of 10 will apply them every 10th step, a value of 1 will apply them every step. |
| `max_batches` | In order to not inflate VRAM overhead for the pixel space losses in the latent model, the batches are split and each batch is evaluated separately. This lets you define a maximimum number of batches to eval the losses for. |
| `lambda_l1` | The weighting factor for L1 (MAE) pixel-space loss. |
| `lambda_vgg` | The weighting factor for VGG perceptual (VGG19/L1) pixel-space loss. |
| `lambda_sobel` | The weighting factor for Sobel pixel-space loss. |
| `lambda_laplacian` | The weighting factor for Laplacian pixel-space loss. |

### `model`
The `model` section is broken up in to categories. At the top are some core options related to training behavior. These are:
| Settings | Description |
| --- | --- |
| `model_type` | What type of model to use as the base. Options are `pixel` for a pixel-space diffusion model, or `latent` for a latent-space diffusion model. |
| `input_size` | The resolution (^2) of the input images used for training (Note: this value is the pixel-space size of the inputs. Latent size is handled elsewhere). 
| `mixed_precision` | Whether to use mixed precision during training. This can save a significant amount of memory overhead at the cost of training stability. |
| `use_ema` | Whether to apply an exponential moving average to the DAE model. Can help to stabilize training and improve model generalization. |
| `ema_decay` | The momentum factor for the EMA. Higher values will cause the EMA to update more slowly. |
| `accumulate_gradients` | If `false`, the gradients will be backpropegated each batch. If `true`, they will be accumulated for a number of batches equal to `gradient_accumulation_iterations`, then backpropagated. This allows you to simulate larger batch sizes on GPUs with smaller amounts of VRAM. Your effective batch size will be `batch_size` * `gradient_accumulation_iterations`. |
| `gradient_accumulation_iterations` | The number of steps over which to accumulate the gradients. See above. |

#### Next we have settings related to the denoising schedule. These are:

| Settings | Description |
| --- | --- |
| `timesteps` | The number of denoising timesteps to use during training (and sampling, if using DDPM). |
| `schedule_type` | What type of noise schedule to use. Options are [`linear`, `cosine`]. |
| `cosine_offset` | How far to offset the cosine sampling if using a cosine noise schedule. |

#### Then we have settings related to the reverse diffusion algorithm:

| Settings | Description |
| --- | --- |
| `function` | What sampling algorithm to use. Options are [`DDPM`, `DDIM`] (Note: `DDIM` is only available for the latent-space diffusion model). |
| `ddim_timesteps` | How many timesteps to use for `DDIM` denoising. If using `DDPM`, the noise_schedule `timesteps` value will be used instead. |
| `ddim_recompute_epsilon` | Whether to recompute the epsilon each step when sampling with `DDIM`. |

#### Next are settings related to the multi-layer perceptron used for timestep embedding:

| Settings | Description |
| --- | --- |
| `time_embedding_dimension` | The feature width of the MLP input layer. Higher values allow for more accurate timestep sampling at a small performance cost. A general rule is `time_embedding_dim = features * 4`. |
| `hidden_dimension` | The feature width of the MLP's hidden layer. |
| `output_dimension` | The feature width of the MLP's output layer. Should be equal to `time_embedding_dimension`. |


#### And finally, we have settings related to the UNet used for noise prediction. These are broken down in to subsections:

##### Base UNet settings

| Settings | Description |
| --- | --- |
| `in_channels` | The number of input channels for the first encoder layer. Generally 3 for pixel-space, and 4 for latent-space. |
| `features` | The output feature count of the first encoder layer. |
| `n_downs` | The number of downsampling layers in the encoder path. |
| `middle_layer_depth` | The number of residual blocks to add to the middle layer. If `self_attention` is enabled, an attention block will be added after each residual block, save for the final one. |
| `self_attention` | If true, self attention blocks will be added to the deepest layers on the encoder and decoder path, and also to the middle layer. |
| `enable_checkpointing` | Enabled checkpointing of residual and attention blocks in the UNet to reduce VRAM overhead. |

##### UNet Optimizer Settings

| Settings | Description |
| --- | --- |
| `optimizer_type` | What type of optimizer to use for the UNet. Options are [`Adam`, `AdamW`] |
| `betas` | The betas for the UNet optimizer. |
| `fused` | Whether to fuse the optimizer. This can help save a bit of VRAM when training with CUDA. |

##### Compile settings

| Settings | Description |
| --- | --- |
| `enable` | If True, enables pre-compilation of the UNet. This can significantly reduce memory overhead and speed up training. (Note: with pre-compilation enabled, the first training update will appear to take much longer. This only applies to the first update, however.) |
| `mode` | What compilation mode to use. See [here](https://docs.pytorch.org/docs/stable/generated/torch.compile.html) for options. (Note: if using gradient acculumation, some modes, such as `reduce-overhead`, may not work currently). |

##### Learning rate settings

| Settings | Description |
| --- | --- |
| `base_rate` | The learning rate to use after warmup, and before decay. |
| `warm_up` | If true, learning rate will be warmed up from 0.0 to `base_rate` over `warm_up_steps` number of optimizer steps. |
| `warm_up_steps` | The number of steps over which to warm up the learning rate. When the current step equals `warm_up_steps`, the model will have reached its full learning rate, defined by `base_rate`. |

**Next we have some settings related to learning rate decay.** These are:

| Settings | Description |
| --- | --- |
| `enable` | Whether to enable LR decay for the UNet optimizer. |
| `schedule_type` | What type of decay schedule to use. Options are [`Linear`, `CosineAnnealing`, `CosineAnnealingWarmRestarts`] |
| `steps_before_decay` | The number of optimizer steps (after the warmup period has completed, if applicable) to hold the LR at its `base_rate` before beginning the decay schedule. |
| `decay_steps` | The number of steps over which to decay the LR from its `base_rate` down to the `minimum_lr` |
| `minimum_lr` | The desired LR after the decay has been fully completed. |

**And then we have a couple settings which are specifically related to warm restarts. These settings only apply when `schedule_type` is set to `CosineAnnealingWarmRestarts`.**

| Settings | Description |
| --- | --- |
| `steps_before_first_restart` | The number of optimizer steps to perform before the first warm restart. |
| `restart_steps_multiplier` | The value to multiply the number of restart steps by after each restart. |

### `dataloader`

| Settings | Description |
| --- | --- |
| `dataroot` | The system path to the directory containing the images to use for training. |
| `batch_size` | The batch size to use during training. |
| `num_workers` | The number of workers to allocate to the training dataloader. |
| `crop_type` | What crop type to use. Options are: `center` for a center crop, `random` for a random crop from the images original size to the `input_size` specified in the `model` category, or `null` for no cropping. |
| `drop_last` | Whether the dataloader should drop the last batch to maintain a constant batch size. |
| `pin_memory` | Whether to enable memory pinning on the dataloader. Can help save a bit of VRAM at training time. |
| `shuffle` | Whether to enable shuffling on the dataloader. |

#### `streaming`
Next we have some settings related to streaming dataloaders (i.e. dataloaders which pull data at runtime, like LAION)


> [!NOTE]
> This feature is in early beta. You may find bugs when using it, and the behavior of it is likely to change in the future. Use at your own risk!

| Settings | Description |
| --- | --- |
| `enable` | Whether to enable dataset streaming. |
| `dataset` | What streaming dataset class to use. Currently the only option is `LAION`. |
| `set` | What dataset to stream. A list of available datasets can be found at the top of [this file](https://github.com/ZacharyBork/NectarGAN/blob/main/nectargan/dataset/streaming_datasets/laion_dataset.py). NOTE: The only one I have personally tested is `aesthetics/square`. Others may or may not work! |
| `subset` | What subset to stream. See `set` for more details. |
| `max_samples` | The maximum number of samples to pull from the dataset. |
| `cache_directory` | The system path to the directory to cache pulled data to, or `null` to disable caching. |
| `require_login` | Whether to require Hugging Face login for the dataset. In the future, this will be automatic. For now though, you must manually enable this option if your chosen set requires you to log in, otherwise the script will throw an exception.  |

#### `augmentations`
***Augmentations are currently disabled for the diffusion datasets. The will be re-enabled at some point in the near future.***

### `latents`
Here you will find options related to the latent space tensors used for training the latent diffusion model. These are:

| Settings | Description |
| --- | --- |
| `latent_size_divisor` | The spatial size divisor for the input images. Generally `8`, so 512x512 inputs become 64x64 latents. |
| `override_latent_size` | Enable to override the size divisor with the spatial size specified by `latent_size`. |
| `latent_size` | Allows you to specify an exact spatial size for the latent-space tensors used for training. |
| `vae` | What VAE model to use. The default is `stabilityai/sd-vae-ft-ema`, but you may specify any model from `diffusers.AutoencoderKL`. |
| `vae_device` | What PyTorch device to use for the VAE. |
| `vae_dtype` | What dtype to use for the VAE. |
| `scaling_factor` | What scaling factor to use for the VAE outputs. |

#### `cache`
Next we have some options related to loading latent tensor caches, which is the preferred way to train currently with NectarGAN diffusion models.

| Settings | Description |
| --- | --- |
| `read_from_cache` | If `true`, latent space tensors will be loaded from a cache during training, rather than directly encoding from input images at training time. |
| `cache_directory` | The system path the the latent cache directory. |
| `loader_type` | What loader type to use when reading the cache. Options are `CacheLoader` to use the traditional cache loader which reads single caches using the associated metadata file, or `CacheShardHotLoader` to use a `dumb` loader which will load any shards dumped in to the given `cache_directory` without requiring metadata (allowing you to mix and match caches by just copying their individual shard files in to the `cache_directory`). |

### `captions`
Next we have settings related to caption loading for text conditional diffusion training.

| Settings | Description |
| --- | --- |
| `use_captions` | Whether to enable text conditioning during training. |
| `metadata_file` | The system path to the caption metadata file created with one of the provided caption loaders, or from your own metadata file which follows the metadata guidelines (discussed below). |
| `encoder_model` | What text encoder model to use. Default is `openai/clip-vit-large-patch14`, but you may use any model from `CLIPTokenizer.from_pretrained`. |
| `max_length` | The maximum caption length to allow. |
| `freeze_encoder` | Whether to freeze the encoder model during training. |
| `use_fixed_captions` | Whether to use the fixed captions defined by `fixed_captions` when generating example images during training. |
| `fixed_captions` | A list containing the captions you would like to use for example image generation during training. A single fixed caption will be selected randomly each time an example image is generated. |

### `save`
Here you will find settings related to checkpoint and example image saving during training.

| Settings | Description |
| --- | --- |
| `save_model` | Whether to save model checkpoints during training. |
| `model_save_rate` | The rate (in steps) at which to save checkpoints during training. |
| `auto_increment_version` | Whether to auto-increment the experiment version. |
| `save_examples` | Whether to save example inference images during training. |
| `example_save_rate` | The rate (in steps) at which to save example images. |
| `num_examples` | The number of example images to save at each increment of `example_save_rate`. NOTE: This is the number of examples to save for each CFG scale specified in `sample_cfg_scales`. |
| `sample_cfg_scales` | **Only used if text conditioning is enabled.** A list of floats specifying the CFG scales to use when generating example images. High CFG scales can make it difficult to identify early training issues, so using a low (1.0-2.0) and a high (~7.5) scale can be useful, especially early on in training, to ensure the the model is progressing well. |

### `visualizer`
And lastly, we have settings related to data visualization during training. This includes both console logging of loss values, and also loss and training image visualization in Visdom.

#### `visdom`

| Settings | Description |
| --- | --- |
| `enable` | Whether to enable Visdom visualization during training. |
| `average_loss` | Whether to average the losses displayed in Visdom over any given update period, rather than sampling the loss stochastically at the time which the update occurs. |
| `env_name` | The environment name to use in Visdom |
| `server` | What server to send the Visdom updates to. |
| `port` | The port over which to send the Visdom updates. |
| `display_images` | Whether to display example images in Visdom during training. NOTE: **This adds 3 VAE decodes per example set when using the latent diffusion model! THIS IS VERY SLOW!** It is generally best to only enable this during early training, to ensure that the model is learning correctly and that the VAE is able to correctly decode your training images. |
| `image_size` | The size to display the images in Visdom. |
| `max_images` | The maximum number of image sets to display. See `display_images` for the reasoning behind this option. |
| `update_frequency` | The frequency (in steps) at which to send updates to Visdom. |

#### `console`

| Settings | Description |
| --- | --- |
| `average_loss` | Whether to average the losses displayed in the console over the given update period, rather than sampling the loss stochastically at the time which the update occurs. |
| `print_frequency` | The frequency (in steps) at which to send updates to the console during training. |

## Latent Tensor Pre-Caching
NectarGAN includes a system to pre-cache latent-space tensors for local datasets. This eliminates the need for VAE encoding at runtime and can significantly speed up training of latent diffusion models. The pre-caching system is fully compatible with text conditional training. To begin, first navigate to the NectarGAN root directory in your terminal, the run the command:

```bash
python -m nectargan.latent.build_latent_cache -r "/path/to/dataset/directory"
```

When run, this command takes all the images in the directory specified with the `-r` flag, encodes them to latent space, caches them as shard files, and exports teh shards to a new output directory created as a subdirectory of the parent dataroot directory. This command includes a number of arguments which can be used to alter the functionality of caching operation. These are:

| Argument | Description |
| --- | --- |
| `-r`, `--dataroot` | **REQUIRED.** The system path to the directory containing the image files you would like to cache. |
| `-d`, `--device` | The PyTorch device to use when encoding the latent-space tensors. |
| `-t`, `--dtype` | The dtype to use when encoding the latent-space tensors. |
| `-m`, `--model` | The VAE model to use for encoding. The default is `stabilityai/sd-vae-ft-ema`, but you may use any model from `diffusers.AutoencoderKL`. |
| `-s`, `--latent_spatial_size` | The desired spatial size (^2) of the latent tensors. |
| `-bs`, `--batch_size` | The batch size to use when encoding the tensors to latent space. |
| `-ss`, `--shard_size` | The number of batches to save per shard. The total number of latent tensors per shard will be `--batch_size * --shard_size`. |
| `-w`, `--num_workers` | The number of workers to allocate to the dataloader used to load input images for VAE encoding. |
| `--store_file_names` | If this flag is present, the file names of the original image files will be stored in the cache manifest. **This is required if you intend to use text conditioning, as this is how the caption dataloader looks up the correct captions for each tensor!**  |
| `--validate_cache` | If this flag is present, after the caching operation is fully completed, each shard will then be loaded and each tensor will be checked for NaNs, infs., etc. |

**The caching mechanism has two modes it can run in,** as determined by the `-bs/--batch_size` argument. If `--batch_size` is greater than 1, the cache manager will expect that the given dataset images have a consistent aspect ratio (i.e. every dataset image has the same aspect ratio). This allows you to encode in batches to speed up the caching. If `--batch_size` is set to 1, however, the cache manager allows you to cache datasets whos images have inconsistent aspect ratios. This allows you to encode and cache all data in the dataset without losing anything, meaning you can then random crop the latent tensors at runtime for added variation.

***PLEASE NOTE: Currently, the manifest is only written out at the very end of the caching operation! This will be changed in the future to write progressively as shards are cached out. For now though, you must wait for the entire caching operation to complete if you intend to use the cache manifest!***

## Captions & Contexts

It is possible to train text-conditioned diffusion models with NectarGAN. Currently, this is limited to the latent diffusion model, and can be enabled from the config in the `captions` section. To train with text conditioning, you must first generate a NectarGAN compatible metadata file for your captions. 

As no two datasets in the wild really use the same caption format, NectarGAN instead has its own standardized caption metadata format, stored as a `.json` file. The current schema follows this pattern:

```json
{
    "info": {
        "schema_version": 1,
        "total_captions": 6,
        "total_images": 2
    },
    "items": {
        "file_name_1": {
            "filepath": "/path/to/image/file_1",
            "captions": [
                "Caption number 1.",
                "Caption number 2.",
                "Caption number 3."
            ]
        },
        "file_name_2": {
            "filepath": "/path/to/image/file_2",
            "captions": [
                "Caption number 1.",
                "Caption number 2.",
                "Caption number 3."
            ]
        }
    },
    "other": {}
}
```
**Let's break this down real quick,** then we will touch on how to generate this format for your own datasets. 

#### First, we have a section called `info`
In this section, we store the schema version (for eventual backward compatibility, should the metadata format ever change in the future). Currently, the only schema version is `1`.

Next, we store the total number of captions. This is the sum of all captions for every image in the dataset.

And lastly, we store the total number of images in the dataset.

#### Next, we have a section called `items`

Items is a dict-like object where the keys are the file names (**without file suffixes**), and the values are also dict-like objects. These values house two things:

1. The system path to the image file.
2. A list of strings containing all of the captions for the given image file.

#### And lastly, we have a section called `other`

In the native NectarGAN datasets, this is not used. It is included in the event you would like to write your own dataloader and wish to include some sort of additional metadata along with your captions.

### Now, let's have a quick look at how to generate this metadata file

NectarGAN includes a few classes to generate captions for commonly used datasets. These can be found [here](https://github.com/ZacharyBork/NectarGAN/blob/main/nectargan/dataset/metadata/caption_loader.py). Included are a class which can be used to generate a metadata file for the [COCO2017](https://cocodataset.org/#home) captions, and for the [CUB200](https://www.vision.caltech.edu/datasets/cub_200_2011/) captions. These are both relatively simple. They just load up the annotations file for the dataset, and based on how the original annotations are laid out, convert them to follow the NectarGAN standard. These can be used as a guide to write a script which can generate the metadata file for your own dataset, and the default classes will likely be expanded in the future. If there is a caption set you want which is not listed, please feel free to submit a GitHub issue with your caption request, and I will try to get a default class added for the given dataset!






