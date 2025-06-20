# NectarGAN API - Dataset
#### The NectarGAN API currently provides a dataset helper class and a robust Albumentations transform wrapper geared towards paired image translation tasks. There are plans to expand this for unpaired training in the near future.

## The PairedDataset class
Reference: [`nectargan.dataset.paired_dataset`](/nectargan/dataset/paired_dataset.py)

This is a simple helper class for loading and processing dataset images for paired adversarial training. As with most components of the NectarGAN API, the `PairedDataset` class takes a `Config` at init (see [here](/docs/api/config.md) for more info). It also takes an `os.Pathlike` object for its `root_dir` argument. This `PathLike` should point to the root directory of your dataset, the directory which contains your `train`, `val`, and, optionally, `test` subdirectories housing your dataset image files. 

> [!NOTE]
> If you are using a [`Trainer` subclass](/docs/api/trainers/trainer.md), the base `Trainer` class provides the [`build_dataloader()`](/nectargan/trainers/trainer.py#L217) wrapper for convenience. This function will use the values in the `Trainer`'s config to build a `PairedDataset`, and from it, a `torch.utils.data.Dataloader`. All you need to tell it is what type of loader (i.e. `test`, `train`, `val`) to create so it knows which dataset subdirectory to load the images from.

### Creating a PairedDataset instance
**A `PairedDataset` can be instantiated as follows:**
```python
from nectargan.config.config_manager import ConfigManager
from nectargan.dataset.paired_dataset import PairedDataset

config_manager = ConfigManager('path/to/config.json') # First we make a ConfigManager from a config file.
dataset = PairedDataset(                              # Then we create a PairedDataset instance,
    config=config_manager.data,                       # passing it the Config object from the ConfigManager
    root_dir='/path/to/dataset/root/directory')       # and the path to our dataset root directory.
```
**Then is can be passed directory to a `torch.utils.data.DataLoader`, again using our config values, like this:**
```python
from torch.utils.data import DataLoader
batch_size=config_manager.data.dataloader.batch_size
num_workers=config_manager.data.dataloader.num_workers

loader = DataLoader(
    dataset, 
    batch_size=batch_size, 
    shuffle=True, # Or False if you don't want shuffling
    num_workers=num_workers)
```

### Dataset Image Loading
Reference: [`nectargan.dataset.paired_dataset`](/nectargan/dataset/paired_dataset.py#L39)

The following are the exact steps which are taken when a `PairedDataset` loads an image:

1. It first loads the image from the dataset based on the `index` argument as a `PIL.Image`.
2. Then it resizes the image based on the dataloader load size in the config that the `PairedDataset` was passed when it was initialized.
3. Then it converts the `PIL.Image` to a `numpy.ndarray`.
4. After that, it checks for the dataloader direction in the config file. Based on that value, it crops the images and assigns the `input_image` and `target_image` (or raises an exception if the provided direction is invalid. Valid directions are `AtoB` and `BtoA`).
5. It uses the `Augmentations` class (described below) to apply augmentations based on the augmentation settings in the provided config.
6. Returns the two images as a tuple of `torch.Tensors` (i.e. (`input_image`, `target_image`))

## The Augmentations class
Reference: [`nectargan.dataset.augmentations`](/nectargan/dataset/augmentations.py)

> [!NOTE]
> This class used to be called `Transformer` until I decided that probably wasn't a great name. Some things in the class's code still reflect this old name though.

The NectarGAN API manages train-time dataset augmentations via a helper class called `Augmentations`. This class is not meant to be interacted with directly. It is instead used by the `PairedDataset` class to apply augmentations to images at run time based on the [dataset augmentations settings](/nectargan/config/default.json#L20) in the config file. The `Augmentations` class is very simple, consisting only of a few private functions to build an Albumentations `Compose` for transformations applied to both images, another for transformations applied only to the input images, and another still for those transforms which are only applied to the target image (although that's not particularly common in image-to-image tasks so all that one actually does in its standard implementation is normalizes the input and converts it to a tensor).

The only public function is [`apply_transforms()`](/nectargan/dataset/augmentations.py#L135). It takes two `numpy.ndarrays` as input (the input and target image from the dataset file, as passed to it via the `PairedDataset`'s [`__getitem__`](/nectargan/dataset/paired_dataset.py#L39) method). It takes those two `ndarrays`, applies the relevant transforms to each, and returns them as a tuple of `torch.Tensors`: (`input`, `target`).

**Following are a list of currently supported transforms, broken down by category. Note that while they are not listed, both the input and the target image are also normalized (-1, 1) and converted to tensors.**

### Augmentations (Input)
- Colorjitter
- Gaussian Noise
- Motion Blur
- Random Gamma
- Image Compression
### Augmentations (Both)
- Horizontal Flip
- Vertical Flip
- 90° Stepped Rotation
- Elastic Transform
- Optical Distortion
- Coarse Dropout
---