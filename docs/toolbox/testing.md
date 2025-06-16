# NectarGAN Toolbox - Test
#### Here you will find settings related to testing previously trained models.

| Setting | Description |
| :---: | --- |
| Experiment | The system path to the experiment directory for the experiment you would like to test. |
| Load Epoch | The checkpoint epoch you would like to load for testing. |
| Test Iterations | The number of images from the test dataset to run the model on. |
| Override Dataset | If enabled, this will allow you to input the path to a dataset to use from testing, rather than using the one defined on the `Dataset` tab. |
| Override Dataset Path | If override dataset is enabled, this path will be used for the test dataset rather than the one in the `Dataset` tab. |
| Test Version | Each time a test is run on an experiment, it will be added to this dropdown menu so that you can switch between them. |
| Image Scale | Scale multiplier for the displayed images. |
| Reset Scale | Resets images back to their initial scale. |
| Sort By | Sorts the test results by various metrics. |
| Sort Direction | Changes the direction of the test result sorting (i.e. Ascending, Descending).  |

---
*See Also:*
| [Toolbox - Home](/docs/toolbox.md) | [Experiment](/docs/toolbox/experiment.md) | [Dataset](/docs/toolbox/dataset.md) | [Training](/docs/toolbox/training.md) | [Review](/docs/toolbox/review.md) | [Utilities](/docs/toolbox/utilities.md) | [Settings](/docs/toolbox/settings.md) |