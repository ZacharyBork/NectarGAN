# NectarGAN API - ONNX Tools
#### The NectarGAN API includes a simple utility class for converting trained models to ONNX, and a script for testing the resulting model on real test images.

> [!NOTE]
> The `ONNXConverter` is primarily meant to be used with the Toolbox UI. It is fully functional as a standalone class, it's just a little rough to use. It and the ONNX tester script will likely be rewritten in the future to make them a little easier to use outside of the Toolbox.

## What is ONNX?
Reference: https://onnx.ai/

ONNX (or *Open Neural Network eXchange*) is an inference-only, open source, platform agnostic interchange format for pre-trained neural networks. Models from most any widely used machine learning framework can be converted to `.onnx` files, which can then be run in anything that supports the [ONNX Runtime](https://onnxruntime.ai/). It is a popular solution for sharing and deploying trained models.

## The ONNXConverter Class
Reference: [`nectargan.onnx.converter`](/nectargan/onnx/converter.py)

The `ONNXConverter` class provides a simple wrapper around `torch.onnx.export`, allowing you to more easily convert your models trained with NectarGAN to the `.onnx` format. 

### Creating an ONNXConverter Instance
Let's start by briefly going over the class's [`__init__`](/nectargan/onnx/converter.py#L13) function to see how we can create an `ONNXConverter` instance in our own scripts. We will start with the input arguments. Looking at the function, we can see there are only a few of them, and many are optional. Following is a list of all of the possible input arguements, with a short description of what each one is and how it is used:
| Argument | Description |
| :---: | --- |
`experiment_dir` | **Required.** This is an `os.PathLike` object representing the system path to the experiment directory which houses the checkpoint `.pth` you would like to convert to `.onnx`.
`config` | **Required.** This can be any number of things representing a NectarGAN config (see [here](/docs/api/config.md) for more info).
`load_epoch` | **Optional.** If this is not `None`, it should be an integer representing the epoch value of the checkpoint you would like to load for conversion. However, if it is `None` (default), the `load_epoch` value will instead be taken from the input `config`'s `["config"]["train"]["load"]["load_epoch"]`.
`in_channels` | **Optional.** If this is not `None`, it should be an integer representing the number of input channels for the generator (i.e. 1 for mono, 3 for RGB). ***Currently, the only number of `in_channels` supported by the API is 3, although there are plans to expand that in the future.*** However, if it is `None` (default), the `in_channels` value will instead be taken from the input `config`'s `["config"]["dataloader"]["load"]["input_nc"]`.
`crop_size` | **Optional.** If this is not `None`, it should be an integer value representing the desired input tensor width and height of the converted model. However, if it is `None` (default), the `crop_size` value will instead be taken from the input `config`'s `["config"]["dataloader"]["load"]["crop_size"]`.
`device` | **Optional.** If this is not `None`, it should be a string representing the desired target device of the converted model (i.e. 'cpu', 'cuda'). However, if it is `None` (default), the `device` value will instead be taken from the input `config`'s `["config"]["common"]["device"]`.

**Alright, pretty simple. The only things we are required to pass it are the path to an experiment directory, and a config. _The rest of the arguments are just convenience overrides._** If they are not set, the values will instead just be taken from the input `config`. So with that in mind then, let's have a quick look at how we can create an `ONNXConverter` instance:
```python
from nectargan.onnx.converter import ONNXConverter

converter = ONNXConverter(
    experiment_dir='/path/to/experiment/directory', # Path to experiment directory with model to convert
    config='/path/to/config_file.json',             # Path to .json config file for model settings
    load_epoch=200,                                 # Load the epoch200 checkpoint for conversion
    in_channels=3,                                  # Model input channels, 3 = RGB
    crop_size=256,                                  # Target input tensor shape: [1, 3, 256, 256]
    device='cpu')                                   # Target cpu (or 'cuda') for converted model
```

> [!NOTE]
> In it's current state at least, the `ONNXConverter` is pretty deeply tied in to the pipeline conventions of the NectarGAN API. It expect that the input model for conversion is using the directory structure and naming conventions which are standard to training with the API.
>
> **In practical terms, this means that the `ONNXConverter` is expecting that the directory which is passed for `experiment_dir` contains within it a `.pth.tar` file for the given `load_epoch` called `epoch{load_epoch}_netG.pth.tar`.** You can convert any other mode, as long as it is a `.pth.tar`, by just dropping it in any directory, calling it `epoch1_netG.pth.tar`, and then instantiating the `ONNXConverter` with the path the that directory and a `load_epoch` value of 1.
>
> **Please also note that this is taken a step further in the Toolbox's `ONNXConverter` wrapper**, in that it also requires the `train{x}_config.json` file, which is automatically exported with every train when run from the Toolbox, to be present in the `experiment_dir`. This is because in the Toolbox, everything related to conversion is set in the interface save for the generator architecture settings which are required to remain static across training sessions regardless, so those are just grabbed automatically from the latest config in the `experiment_dir` instead.
### Converting a Model
Great, now we have our `ONNXConverter` instance ready to go. Let's now take a look at how we can use it to convert the model we pointed it to. Looking over the `ONNXConverter` class, we can see that it actually only has one public function that is unique to itself, and not inherited from the `Trainer` class: [`convert_model`](/nectargan/onnx/converter.py#L55)

**Let's again start by having a quick look over the input arguments for `convert_model`:**
| Argument | Description |
| :---: | --- |
`export_params` | If `True` (default), model weights will be exported.
`opset_version` | What [ONNX opset version](https://onnxruntime.ai/docs/reference/compatibility.html) to use for the converted model.
`do_constant_folding` | Fold constants for model optimization (This is deprecated in later versions but included for compatibility. Note, if `suppress_onnx_warnings=True` (default), the compatibility warning related to this will be silenced from printing to the console).
`input_names` | Names to assign to the graph inputs.
`output_names` | Names to assign to the graph outputs.
`suppress_onnx_warnings` | If `True`, the [normalization training mode warning](https://github.com/pytorch/pytorch/issues/75252) and a warning related to constant folding incompatibility will be silenced from printing to the console. 
> [!NOTE]
> Save for `suppress_onnx_warnings`, all of the input arguments for `convert_model` are just direct passthrough arguments for [`torch.onnx.export()`](https://docs.pytorch.org/docs/stable/onnx_torchscript.html#torch.onnx.export).

**And now let's have a look at how we can use the `convert_model` function to convert our trained model.** Using our `converter` which we created above, we simply need to do this:
```python
converter.convert_model(
    export_params=True,              # Generally we want to export our model weights
    opset_version=11,                # Or whatever version you want. Higher version = more features
    do_constant_folding=True,        # This actually won't do anything if opset_version>10
    input_names=['my_input_name'],   # Whatever input names you want, or none for default: ['input']
    output_names=['my_output_name'], # Whatever output names you want, or none for default: ['output']
    suppress_onnx_warnings=True)     # Or False if you want it to print the warnings
```
**And that's it. When run, this will export the converted model to the given `experiment_dir`.** The converted file will the called `epoch{load_epoch}.onnx` (this can be changed by altering the `onnx_filename` variable in [`ONNXConverter._build_output_path()`](/nectargan/onnx/converter.py#L51))

## ONNX Model Test Script
Reference: [`nectargan.onnx.tester`](/nectargan/onnx/tester.py)

There is also a very simple test script your converted models. It will take the path to the `.onnx` model, and the path to a test image you would like to run the model inference on. The script will load the model with `onnxruntime`, run the model's inference on the given test image, and display the result via `matplotlib`. This is just a super simple validator, mostly provided as a convenience. You can also use the [ONNX tester](/docs/toolbox/utilities.md) in the Toolbox to test your converted models, though. It is a bit more full-featured and allows you to test the model on more than a single image at a time.

---