# NectarGAN Toolbox - Settings
> [*`NectarGAN Toolbox - Home`*](../toolbox.md)
#### This panel contains general interface settings. There is not much here currently, but more will likely be added in the future.

> [!IMPORTANT]
> Any value in this panel can be changed during training. For some, this has no effect but for settings like `Training Update Rate`, this will change the rate at which the UI is updated live during training. One small thing to note is that for the live loss graphs, this will also change the rate at which values are added which can look a little strange so if you are relying on the live loss graphs, set this value prior to training and then leave it unchanged for the duration of the training session.

| Setting | Description |
| :---: | --- |
| Always on Top | If checked, the Toolbox window will always remain on top of other windows. |
| Review Graph Sample Rate | When loading loss log data from the review panel, it is generally not advisable to load every single datapoint from the log, as that can take an unreasonable long time, especially for logs from longer training sessions. This value allows you to set a sample rate to be used when loading that data, and the graphing backend will skip this number of datapoints from the log in between each datapoint that does get added to the graph. Setting this to `1` will tell the Toolbox to load every datapoint, but expect the UI to lock while it does this. In the future, I may look in to offloading this to another thread with some progress feedback. |
| Training Update Rate | Using Toolbox, model training is run in a separate thread from the main program. Every so often, the worker in that thread will send a signal back to the main thread with some data (loss data, timing information, example image tensors, etc.). This value lets you set the frequency (**in iterations**) that the signal is sent. This will chance the rate at which the example images are updates, and which losses are printed to the console and added to their respective graphs. Higher values can speed up training at the cost of reduced real-time feedback. |

---
*See Also:*
| [Toolbox - Home](../toolbox.md) | [Experiment](experiment.md) | [Dataset](dataset.md) | [Training](training.md) | [Testing](testing.md) | [Review](review.md) | [Utilities](utilities.md) |