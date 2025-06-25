# NectarGAN - Getting Started (Toolbox | Testing)
##### Reference: [`NectarGAN Toolbox Documentation`](/docs/toolbox.md)
> [!NOTE]
> This is part two of the three-part Toolbox quickstart guide.
>
> Click [*here*](/docs/getting_started/toolbox_training.md) for part one.<br>
> Click [*here*](/docs/getting_started/toolbox_review.md) for part three.

## The Testing Panel
> [!IMPORTANT]
> **The `Testing` panel will use the generator architecture and dataset settings defined in their repective panels (i.e. `Experiment` and `Dataset`).**
> 
> If these UI settings have changed since you trained the model, you can load the settings used for training by clicking the `Load Setting from Config File` button on the bottom of the `Experiment` panel, and selecting the config file from the given experiment directory, which is exported automatically at the beginning of each train.

##### Reference: [`Toolbox Testing Panel`](/docs/toolbox/testing.md)
### Testing a Model

**The `Testing` panel can be accessed by clicking on the green `Testing` button on the left-hand bar.** After doing so, you will be presented with a screen which looks like this:
![_toolbox_testing_](/docs/resources/images/toolbox/toolbox_testing.PNG)
**Here, you can load your generator checkpoints which were exported during training and evaluate them on real test images.**

**To test our model, we will:**
1. **Input the path to our experiment directory** from the training session we ran in the previous chapter (the experiment directory is the root directory containing the `.pth.tar` checkpoint files, `examples` directory, etc.).
2. **Click on the `Most Recent` button** to have the toolbox grab the most recent generator checkpoint in the experiment directory. Or, alternatively, manually inputting the epoch number of a checkpoint we would like to test.
3. **Increase the value of `Test Iterations` to 30.** This will tell the `Tester` that we would like it to randomly select 30 images from the `test` directory.
4. **Click `Begin Test`.** 

### Test Results
**After a couple seconds, you should be presented with a screen which looks like this:**
![_toolbox_testing_example_](/docs/resources/images/toolbox/toolbox_testing_example.PNG)
> [!NOTE]
> Your images may be larger. Image scale can be controlled with the slider on the bottom of the results panel.

**Here we can see that, for each testing iteration, we are displaying the example input and ground truth images, as well as the generators fake image.**

**We also have some loss values.** For each test iteration, before converting the images for display, we also take the tensors and run a few loss functions on them. You can sort the test results by these metrics, as well as a few others, using the dropdown menus on the bottom right of the results panel.

**Now, let's hit `Begin Test` one more time.** When we do, we see that the model has been run on another random selection of images from the `test` set:
![_toolbox_testing_example_02_](/docs/resources/images/toolbox/toolbox_testing_example_02.PNG)
**Now, if we click on the dropdown menu called `Test Version`, located in the bottom left of the results panel, we will see that we have two items, `test_01` and `test_02`.** This allows you to switch back and forth between the tests to compare. Note that sorting settings will also be preserved when switching between them
### The `test` Directory
Now, let's quickly have a look at our experiment output directory for the experiment we are testing. In doing so, we notice a new subdirectory called `test`. Diving inside this new directory, we can see that, for each test we ran, asubdirectory was created inside. 

Then, in these test subdirectories, the [`A_real`, `B_fake`, `B_real`] image sets for each test iteration were exported, and a `log.json` was created which contains all of the data related to that test, including:
- Experiment name.
- Experiment version.
- Path to the experiment directory.
- Test version.
- Path to the test output directory.

And then, for each test iteration:
- Iteration number
- Path to the input image which was used for that iteration.
- Paths to the three output images from that iteration.
- Loss values tagged by name.

**This is how the test data is loaded into the Toolbox interface. It also makes it extremely easy to track all data related to model testing and integrate it quickly into automated workflows.**

**And that's really all there is to testing models with the NectarGAN Toolbox.** 

#### Next, let's move on to [reviewing training results using the `Review` panel](/docs/getting_started/toolbox_review.md).

---