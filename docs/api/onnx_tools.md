# NectarGAN API - ONNX Tools
#### The NectarGAN API includes a simple utility class for converting trained models to ONNX, and a script for testing the resulting model on real test images.

> [!NOTE]
> The `ONNXConverter` is primarily meant to be used with the Toolbox UI. It is fully functional as a standalone class, it's just a little rough to use. It and the ONNX tester script will likely be rewritten in the future to make it a little easier to use outside of the Toolbox.

## What is ONNX?
Reference: https://onnx.ai/

ONNX (or *Open Neural Network eXchange*) is an inference-only, open source, platform agnostic interchange format for pre-trained neural networks. Models from most any widely used machine learning framework can be converted to `.onnx` files, which can then be run in anything that supports the [ONNX Runtime](https://onnxruntime.ai/). It is a popular solution for sharing and deploying trained models.

## The ONNXConverter Class
Reference: [`nectargan.onnx.converter`](/nectargan/onnx/converter.py)

The `ONNXConverter` class provides a simple wrapper around `torch.onnx.export`, allowing you to more easily convert your models trained with NectarGAN to the `.onnx` format.

### Using the ONNXConverter

## ONNX Model Test Script
Reference: [`nectargan.onnx.tester`](/nectargan/onnx/tester.py)