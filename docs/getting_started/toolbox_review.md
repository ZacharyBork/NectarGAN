# NectarGAN - Getting Started (Toolbox Review)
##### Reference: [`NectarGAN Toolbox Documentation`](/docs/toolbox.md)
> [!NOTE]
> This file is the third part in the Toolbox quickstart guide. It is recommended to read the first two before this one.
>
> Click [here](/docs/getting_started/toolbox_training.md) for part one.<br>
> Click [here](/docs/getting_started/toolbox_testing.md) for part two.

> [!IMPORTANT]
> The review panel is still relatively early in development. Some features may be a little rough around the edges, and you may experience stalls when loading large experiment outputs. It is functional in its current state, but it will likely be overhauled at some point in the future.

## The Review Panel
##### Reference: [`Toolbox Review Panel`](/docs/toolbox/review.md)
**The toolbox review panel can be accessed by clicking on the teal `Review` button on the left-hand bar.** When we do, we are presented with a screen which looks like this:
![_toolbox_review_](/docs/resources/images/toolbox/toolbox_review.PNG)
**Here, we can load the example images and graph the loss log data from a previous train.**

We will start by inputting the experiment directory for the model we trained in [part 1](/docs/getting_started/toolbox_training.md). After that, we can click the `Load Experiment` button to load all the data from our training session. After just a second, we should see the training data loaded up like so:
![_toolbox_review_example_](/docs/resources/images/toolbox/toolbox_review_example.PNG)

**On the left side of the results panel, we can see that it has loaded up all of the [`A_real`, `B_fake`, `B_real`] example image sets.**

**Then, on the right, it has also gone through the loss log and graphed all of the loss data for each loss in the loss log,** with the red line being the mean loss value, and the green line being the weight of the loss (more useful if you are scheduling loss weights, here it is just a flat line for each loss).
> [!NOTE]
> Graphing every single entry for every single loss can be a heavy task, especially for longer trains and/or larger datasets. For this reason, **by default, the review panel will only graph every 50<sup>th</sup> value.** This can be changed in the main settings panel, accessed via the red `Settings` button on the left-hand bar.

**We also see that, in the review settings panel, below where we input our experiment directory path, it has loaded the train config file from the experiment directory for viewing.** If there are multiple train configs in the experiement directory, all of them will be loaded and you can switch between them with the dropdown meny in the top right of the `Train Config` display.

## Conclusion

**And that's it really, those are the core components of the NectarGAN Toolbox. Thank you for reading, and I hope you enjoy working with it :)**

If you have suggestions or find any issues, please feel free to let me know!

For further reading, please see the [NectarGAN Toolbox Documentation](/docs/toolbox.md).

---