# NectarGAN API - Visdom
> [*`NectarGAN API - Home`*](/docs/api.md)
#### The NectarGAN API includes a utility class for managing runtime training data visualization with Visdom, useful for testing configs quickly and/or headlessly, and deugging training-related features without needing to launch the [Toolbox](/docs/toolbox.md) UI.

## What is Visdom?
Reference: [`ai.meta.com/tools/visdom`](https://ai.meta.com/tools/visdom/) [`github.com/fossasia/visdom`](https://github.com/fossasia/visdom)

Visdom is a commonly used, open-source solution for data visualization. It provides simple tools for graphing data in 2D and 3D, rendering images, and displaying text via a locally hosted web server.

*See [here](https://github.com/fossasia/visdom?tab=readme-ov-file#setup) for a guide on how to set up a local Visdom server, which is required in order to use the NectarGAN `VisdomVisualizer`.*

## VisdomVisualizer
Reference: [`nectargan.visualizer.visdom.visualizer`](/nectargan/visualizer/visdom/visualizer.py)

### Creating a VisdomVisualizer instance
This class is relatively simple. Looking at its `__init__` function, we can see it only takes two arguments:
1. `env`: A string which is used to define the Visdom Environment name.
2. `port`: An integer represent the port number over which the client should send updates to the server.

A `VisdomVisualizer` instance can be created as follows:
```python
from nectargan.visualizer.visdom.visualizer import VisdomVisualizer

vis = VisdomVisualizer(
    env='my_env_name', # Can be anything. Or ignored for default ('main')
    port=8097          # 8097 is Visdom's default port number
) 
```
If you are reusing the same environment for multiple tests (as is done in the base `Trainer` class), it is generall best to also clear any old data after initialization. This can be done like so:
```python
vis.clear_env()
```
### Multithreading
**The `VisdomVisualizer` has two "modes" it can run it.** The first will run the visualization in the same thread as the training, the second will spin up a new thread specifically for managing Visdom updates. Which mode it is running in significantly alters the core functionality of update functions so we will briefly discuss how they function first.

**By default, the `VisdomVisualizer` will run in the first mode.** To run visualization in a separate thread, after you initialize the `VisdomVisualizer`, you must call this function:
```python
vis.start_thread()
```
**This will do a few things:**
1. Change the value of the `VisdomVisualizer`'s member variable `is_threaded` to `True`. This flag is checked by the update functions to decide what exactly should be done with the data when it is passed to push to the server.
2. Initialize a `queue.Queue` for storing example images, and another for storing loss graph values.
3. Register and start a thread for Visdom updates.

**In this mode, whenever data is passed to the `VisdomVisualizer` via one of its update functions, rather than being immediately pushed to the Visdom server, it is instead stored in the associated `Queue`.** It will then be picked up during the next [`_update`](/nectargan/visualizer/visdom/visualizer.py#L218) cycle (every two seconds by default) and pushed to the server.

**Then, whenever we are finished with the visualizer thread (at the end of the training loop, for example), _we must also call_:**
```python
vis.stop_thread()
```
**This will ensure that the thread is shut down safely.** Please refer to [`Trainer.train_paired()`](/nectargan/trainers/trainer.py#L353) for an safe example implementation which also accounts for keyboard interrupts.

## Graphing Loss Data
**Loss data can be graphed at runtime by calling [`VisdomVisualizer.update_loss_graphs()`](/nectargan/visualizer/visdom/visualizer.py#L81). Currently, this is primarily intended to be used in conjunction with the [`LossManager`](/docs/api/losses/lossmanager.md), as the data format it is expecting is exactly that which is returned by [`LossManager.get_loss_values()`](/nectargan/losses/loss_manager.py#L211).**

### Input Arguments:
| Argument | Description |
| :---: | --- |
`graph_step` | A `float` value defining where along the X axis to graph the provided data.
`losses_G` | The generator losses, as returned with `LossManager.get_loss_values(query=['G'])` using the default [Pix2pix objective loss spec](/nectargan/losses/pix2pix_objective.py).
`losses_D` | The discriminator losses, as returned with `LossManager.get_loss_values(query=['D'])` using the default [Pix2pix objective loss spec](/nectargan/losses/pix2pix_objective.py).

### Single-Threaded
If run from a single thread, this function calls [`VisdomVisualizer._update_loss_graphs_core()`](/nectargan/visualizer/visdom/visualizer.py#L152), directly updating the graphs for generator and discriminator loss.

### Multi-Threaded
If run in a separate thread, this function instead calls [`VisdomVisualizer._store_graph_data`](/nectargan/visualizer/visdom/visualizer.py#L196), storing all of the input data in the `_graph_queue` to be picked up during the next `_update`.

### Example
See: [`nectargan.trainers.pix2pix_trainer.Pix2pixTrainer.update_display()`](/nectargan/trainers/pix2pix_trainer.py#L192)

## Visualizing `[x, y, y_fake]` Tensors
**Example outputs can be visualized during training via [`VisdomVisualizer.update_images()`](/nectargan/visualizer/visdom/visualizer.py#L70).**

### Input Arguments:
| Argument | Description |
| :---: | --- |
`x` | A `torch.Tensor`. The first image to display.
`y` | A `torch.Tensor`. The second image to display.
`z` | A `torch.Tensor`. The third image to display.
`title` | The title for the image display window.
`image_size` | The resolution (^2) to display **each** image at in the Visdom dashboard.

### Single-Threaded
If run from a single thread, this function directly updates the `[x, y, y_fake]` image display in the Visdom dashboard via [`VisdomVisualizer._update_images_core()`](/nectargan/visualizer/visdom/visualizer.py#L92).

### Multi-Threaded
If run from a separate thread, this function will instead store the input data in the `_image_queue` via [`VisdomVisualizer._store_images()`](/nectargan/visualizer/visdom/visualizer.py#L183) to be picked up during the next `_update`.

### Example
See: [`nectargan.trainers.pix2pix_trainer.Pix2pixTrainer.update_display()`](/nectargan/trainers/pix2pix_trainer.py#L182)

## Utility Functions
**The `VisdomVisualizer` has one notable utility function: [`VisdomVisualizer._denorm_tensor()`](/nectargan/visualizer/visdom/visualizer.py#L46)**

This function is called once for each of the three tensors (`x`, `y`, `z`) in [`VisdomVisualizer._update_images_core()`](/nectargan/visualizer/visdom/visualizer.py#L113). It takes the input tensor which, from Pix2pix at least since the last upsampling layer uses Tahn activation, is normalized (-1, 1), and renormalizes it in the (0, 1) range for viewing.

---