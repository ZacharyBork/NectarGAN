# NectarGAN API - Config
> [*`NectarGAN API - Home`*](../api.md)
#### The NectarGAN API is entirely driven by JSON configuration files. All data related to training, testing and testing models and visualizing results (with the exception of some Toolbox wrappers) is managed by these config files.

#### Configuration files in NectarGAN are managed by two core components, the [`ConfigManager`](https://github.com/ZacharyBork/NectarGAN/blob/main/nectargan/config/config_manager.py) class, and the [`Config`](https://github.com/ZacharyBork/NectarGAN/blob/main/nectargan/config/config_data.py#L148) dataclass, both of which will be explained in this document.

## The Config File
The default config file can be found at: [`nectargan/config/default.json`](https://github.com/ZacharyBork/NectarGAN/blob/main/nectargan/config/default.json) It is broken down into some core sections:
| Section | Description |
| :---: | --- |
`common` | Basic values like output directory, experiment name, and target device.
`dataloader` | Values related to loading and augmenting dataset files.
`train` | Values related to loading checkpoints, learning rate schedules, and loss weighting.
`save` | Values related to checkpoint and example image saving.
`visualizer` | Values related to progress visualization<br>(Currently only for Visdom, the Toolbox has its own system for managing these settings.) 
## The ConfigManager
In the NectarGAN API, config files management is done via the [`ConfigManager`](https://github.com/ZacharyBork/NectarGAN/blob/main/nectargan/config/config_manager.py). The main job of the `ConfigManager` is to take a JSON config file and parse the values into a much more usable set of [dataclasses](https://github.com/ZacharyBork/NectarGAN/blob/main/nectargan/config/config_data.py).
### Using the ConfigManager
Generally speaking, you likely will not be using the `ConfigManager` directly. At least not if you are building a trainer/tester that inherits from the core [`Trainer`](https://github.com/ZacharyBork/NectarGAN/blob/main/nectargan/trainers/trainer.py) class, as it creates it's own `ConfigManager` as one of the first steps in its `__init__()`. However, there are some cases where you might want to, such as writing an entirely new base class or even using a pre-built `ConfigManager` to initialize a `Trainer` via [`Trainer.init_config()`](https://github.com/ZacharyBork/NectarGAN/blob/main/nectargan/trainers/trainer.py#L65), so we will go over the process here.

First, let's have a quick look at the [`ConfigManager.__init__()`](https://github.com/ZacharyBork/NectarGAN/blob/main/nectargan/config/config_manager.py#L18) function. We can see that it only takes one argument: `input_config`. And that can be a few things. Here is a brief walkthrough of what each expects, and the behaviour of the config manager when it is initialized with each:

| Type | Behaviour |
| :---: | --- |
`str` | This is expecting a string with the system path to a `.json` config file. It will attempt to parse the config file and assign the values, or return various exception types if it fails depending on the reason for the failure.
`PathLike` | The is expecting an `os.PathLike` object containing the system path to a `.json` config file. It behaves exactly as the same as `str` does.
`dict[str,Any]` | This is expecting a pre-parsed config file (i.e. `with open('config.json', 'r') as f: data = json.load(f)`).<br><br>It will first run a key check on the input vs. the default config, and if the input passes, it will simply assign the values directly to the `ConfigManager`'s `raw` config data, then parse them into the dataclasses. This one is a little strange because it technically bypasses the JSON stuff altogether, which is actually why it exists, to allow the Toolbox to just dump [`Config`](https://github.com/ZacharyBork/NectarGAN/blob/main/nectargan/config/config_data.py#L148)-like data directly into a `ConfigManager`, rather than needing an intermediate `.json` export step.
`None` | If `input_config` is `None` (default), the `ConfigManager` will attempt to load the [default config file](https://github.com/ZacharyBork/NectarGAN/blob/main/nectargan/config/default.json) located at: `nectargan/config/default.json`. If it is unable to do so, it will raise an exception. 

#### So then, with these options in mind, here are some of the ways in which you could initialize a `ConfigManager`:
```python
import json
from pathlib import Path
from nectargan.config.config_manager import ConfigManager

filepath = '/path/to/my/config_file.json'

config_manager = ConfigManager()                          # Init from default config file
config_manager = ConfigManager(config_file=None)          # Also from default config file

config_manager = ConfigManager(filepath)                  # From a string
config_manager = ConfigManager(Path(filepath))            # or a pathlib.Path
config_manager = ConfigManager(Path(filepath).as_posix()) # or a string from a pathlib.Path

with open('/path/to/my/config_file.json', 'r') as file:   # Or with JSON-like data
    config_data = json.load(file)                         # Generally, don't do this though. Give the ConfigManger
config_manager = ConfigManager(config_data)               # the filepath and let the it parse it directly instead
```
## Config Dataclasses
The `ConfigManager` relies on a dataclass called [`Config`](https://github.com/ZacharyBork/NectarGAN/blob/main/nectargan/config/config_data.py#L147), and a number of other dataclasses which all get packed into it.

Let's have a quick look over the file that houses all of the dataclasses related to the `ConfigManager`: [`nectargan.config.config_data`](https://github.com/ZacharyBork/NectarGAN/blob/main/nectargan/config/config_data.py)

Looking over the dataclasses and their member variables, you may notice that the structure and all of the names and datatypes exactly match the structure and values of the config file. Each dict-type object in the config json has it's own dataclass, all of which are, eventually anyway, loaded into the main `Config` dataclass. All the `ConfigManager` does is load the config file, loop through the `Config` dataclass recursively, find the corresponding entry in the config file's data, and assigns the value, turning config lookups from this:
```python
device = config['common']['device']
```
To this:
```python
device = config_manager.data.common.device
```
Which is a little nicer to work with in my opinion.
## Exporting a Config Copy
**The `ConfigManager` has a simple helper function called[`export_config()`](https://github.com/ZacharyBork/NectarGAN/blob/main/nectargan/config/config_manager.py#L135).** When you call this function and pass it a directory filepath for the `output_directory` argument, the `ConfigManager` will export a copy of it's current config data as a `.json` file to the provided directory. Before exporting the file, the `ConfigManager` will first scan the input directory looking for previous config files, and it it finds any, as would be the case if the user in continuing training on an existing model, the file will be versioned as such: (`train1_config.json`, `train2_config.json`) 
## Adding a Value to the Config File
### Adding a value to an existing group
Adding a value to an existing group is very simple. Let's say we wanted to add a floating point value called 'my_cool_value' to the `common` section of the config file. Fist, we would open our config file and add the value:
```json
{
    "config": {
        "common": {
            "my_cool_value": 42.0, <--- Add your value
            "device": "cuda",
            ...
```
Then, we would go to our config data file, and we would find the related dataclass for the `common` section, [`ConfigCommon`](https://github.com/ZacharyBork/NectarGAN/blob/main/nectargan/config/config_data.py#L4), and we would add our new value with the corresponding type:
```python
@dataclass
class ConfigCommon:
    my_cool_value: float
    device: str
```
> [!IMPORTANT]
> ***The variable name must be all lowercase and the name and data type must match exctly with what is in the config file.***

**And that's it,** now when you initialize a `ConfigManager` with your updated config file, the value of `my_cool_variable` will be accessable with:
```python
config_manager = ConfigManager('/path/to/your/updated/config.json')
value = config_manager.data.common.my_cool_variable
```
**Basically any datatype allowed by JSON is viable to add. Have a quick look through the [`config_data`](https://github.com/ZacharyBork/NectarGAN/blob/main/nectargan/config/config_data.py) file to see how various datatypes are implemented.**
### Adding a new config group
Adding a new group is very similar to the above steps. First, we add our group to the config file, with a variable:
```json
{
    "config": {
        "my_cool_group": {         <--- First add your group,
            "my_cool_value": 42.0, <--- then add your value
        },
        "common": {
            "device": "cuda",
            ...
```
Then, in [`config_data`](https://github.com/ZacharyBork/NectarGAN/blob/main/nectargan/config/config_data.py), we first create a new dataclass and add our value to it:
```python
@dataclass
class ConfigMyCoolGroup:
    my_cool_value: float
```
> [!NOTE]
> Dataclass names are not strict. You can follow the established pattern, or name them whatever you want. **Variable names must be all lowercase though!**

Then, we just need to add out new dataclass to the main [`Config`](https://github.com/ZacharyBork/NectarGAN/blob/main/nectargan/config/config_data.py#L147) dataclass like this:
```python
@dataclass
class Config:
    my_cool_group: ConfigMyCoolGroup # "my_cool_group" here must be lowercase
    common: ConfigCommon
    dataloader: ConfigDataloader
    train: ConfigTrain
```
**And that's it,** now the value of `my_cool_variable` can be accessed as follows:
```python
config_manager = ConfigManager('/path/to/your/updated/config.json')
value = config_manager.data.my_cool_group.my_cool_variable
```
---