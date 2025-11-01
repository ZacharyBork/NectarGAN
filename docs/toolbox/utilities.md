# NectarGAN Toolbox - Utilities
> [*`NectarGAN Toolbox - Home`*](../toolbox.md)
#### Here you will find some general utitilies related to ONNX conversion and testing, and various dataset processing tools.

## Convert to ONNX
This utility allows you to convert your trained model to ONNX format to allow for easier deployment into pipelines which support the ONNX runtime. 

> [!NOTE]
> The Pix2pix model uses instance normalization, but PyTorch considers instanceNorm as training so it will print a warning to the console during conversion. This is expected and does not affect the model's inference ability or quality. As such, it is silenced in the Toolbox implementation, along with another irrelevant opset-based constant folding warning. If you are using the ONNXConverter class to convert your model, you can choose to disable the silencing of these warnings with: `ONNXConverter.convert_model(suppress_onnx_warnings: bool=True)`.
>
> See [here](https://github.com/pytorch/pytorch/issues/75252) for more information on the supressed warning related to instance normalization.

| Setting | Description |
| :---: | --- |
| Experiment | The system path to the experiment directory of the model you would like to convert. |
| Load Epoch | What epoch checkpoint to load for conversion. |
| In Channels | Number of input channels for the model. Currently, the only supported value is 3 but this will be expanded in the future |
| Width/Height | The desired width and height of the converted model's model's input tensor. |
| Target Device | What device the ONNX model should target (i.e. CPU, CUDA). |
| Opset Version | What ONNX opset version to use. See [here](https://onnx.ai/onnx/api/) for more information.  |
| Export Params | Decides whether model weights should be exported. |
| Fold Constants | Decides whether constants should be folded during conversion for optimization. |
| Convert Model | Runs the ONNX conversion process and exported the converted model. |
## Test ONNX Model
**This utility allows you to test the inference capabilities of your converted ONNX model.** By feeding it a path to a model and hitting `Test Model`, Toolbox will load the ONNX model in inference mode and run it on every image in the provided `Test Images` directory, displaying the (input, output) results in the interface.
| Setting | Description |
| :---: | --- |
| Model | The system path to the `.onnx` file to run inference testing on. |
| Test Images | The system path to a directory of test images. ***These should not be paired training-style images. They should just be model input images, on their own, with no ground truth example.***  |
| Test Model | Begins the model testing process. This generally only takes a couple seconds for ~50 test images, though it will depend on the input width and height of your specific model. |
### Pair Images
**This is an extemely fast pairing utility for (A, B) dataset images.** It will take two input directories, one with the input images and the other with the ground truth images, a pairing direction, and, optionally, a target resolution. Then Toolbox will send the task to a worker in another thread which pools the processes to very rapidly scale, if enabled, and pair the images into Pix2pix ready training data. This was very fun to build.

> [!NOTE]
> This utility expects that the two input directories contain nothing but the image files to be used in the pairing operation, and that the A input images have the same file name as their B input counterpart.

| Setting | Description |
| :---: | --- |
| Input A | The system path to the directory containing the first set of images to pair. |
| Input B | The system path to the directory containing the second set of images to pair. |
| Output | The system path to the directory that you would like the utility to output the paried images to. |
| Direction | What direction to pair the images. As the names suggest, `AtoB` will put the input A images on the left and the input B images on the right. `BtoA` will do the opposite. This is sort of arbitrary if you're training with NectarGAN since you can select the direction at train time, but some may find it useful for their pipelines so I've decided to include it. |
| Scale Images | If this checkbox is ticked, the images will first be scaled to the resolution defined by the accompanying dropdown list before being paired. ***Scaling is applied to an image copy in memory at script runtime and will not affect your original image files.*** |
| Image Scale | If the `Scale Images` checkbox is enabled, this dropdown allows you to select a resolution to scale each image to before pairing. The final resolution of each paired output image will be (`Image Scale`*2, `Image Scale`). |
## Image Sorting Utilities
**These three utilities, `Sort Images`, `Remove Sorting Tags`, and `Copy Sort`, allow you to sort image files non-destructively by various metrics, unsort them back to their original order, or copy the sorting order of one directory to another.** This is useful for managing very large datasets and helping to pull bad images to the top so you can more easily find ones that should be discarded. This tool functions in the same way as the pairing utility and as such, is very efficient, even on extremely large datasets.

**Some things to be aware of when using these tools:**

> [!WARNING]
> These tools **rename files in-place** and can modify many files very quickly. Please read this section fully and use the **Preview** first. Before working with these tools on real data for the first time, it is advised to duplicate a small portion of your dataset to test their functionality.
> 1. **Renaming behavior (and undo).**<br>
   The sorter **renames files in your `Input` directory** by **prepending a tag** that reflects their position in the sorted list. You can reverse this with **Remove Sorting Tags**, which strips the sorting tag from the files, effectively unsorting them, even if some files were removed.
>
> 2. **Always run a Preview first (dry run).**<br>
   Every tool has a **Preview** button. The preview shows **exactly** what would be changed without touching your files. **Run Preview before Start.** These tools are fast and perform bulk file operations. Most actions are reversible via **Remove Sorting Tags**, but it’s always safer to confirm first.
>
> 3. **Only include the images you intend to sort.**<br>
   The directory should contain **image files only**. Non-image files currently trigger an exception that isn’t handled by the UI, requiring an app restart. This will be improved in a future update. For now, **make sure the folder contains only the images you want to sort**.

### Sort Images
| Setting | Description |
| :---: | --- |
| Input | The system path to the directory containing the image files you would like to sort. |
| Type | What image metric to sort by (e.g. # of white or black pixels, mean pixel value, RMS contrast, etc.). These are just a collection of sorting functions that I have found useful in my own time spent processing datasets. When I find more useful ones, they will likely get added as well. |
| Direction | What direction to sort (i.e. Ascending, Descending) |
| Start | Begins the sorting operation. |
| Preview | Displays a preview in the interface of what the results would be if the current sorting operation was run. |
### Remove Sorting Tags
| Setting | Description |
| :---: | --- |
| Input | The system path to the directory containing images which have previously been sorted with `Sort Images`, and which currently have sorting tags which you wish to remove (e.g. `1_myfile543.png`, `2_myfile198.png`, `3_myfile31.png`). |
| Start | Start the remove tags operation. **This is not threaded and will lock the UI until it is completed**, but behind the scenes, it is just a sequential pathlib rename operation so even for fairly large datasets, it should be very quick. |
| Preview | Display a preview of what the result of the current `Remove Tags` operation would be. |
### Copy Sort
This utility allows you to copy the sorting order from one directory of pre-sorted (with `Sort Images`) images, to the paired set of images. Helpful to see how eliminating some datapoints from A would affect the data in B.
> [!NOTE]
> This utility relies on the original file names (before sorting tags are prepended) of the images in the input and target directory to be the same, i.e. if an image from the input directory was original called `myfile153.png` before it was sorted, its twin in the target directory should also be called `myfile153.png`.

| Setting | Description |
| :---: | --- |
| From | The system path to the directory of sorted images to copy the sorting order from. |
| To | The system path to the directory of unsorted images to copy the sorting order to. |
| Start | Begins the copy sort operation. **This will lock the UI briefly while it processes,** it is generally fairly quick even on large datasets, though. |
| Preview | Preview the result of the current copy operation. |

### Split Dataset Images
This utility takes a directory of pre-paired dataset images, an `Output` directory, and a percentage for each of `train`, `test`, `val`. It then creates new directories for each category inside out the `Output` directory and roughly splits the provided dataset images into the new directories based on their respective percentage chance.

> [!NOTE]
> There are three percentage sliders, each with a corresponding spinbox. All three go up to 100%. In the split script, they are normalized based on the sum total of the three values, so you can use the sliders to get a rough estimate, or enter precise values in the spinboxes to get a split closer to the exact percentages. The split process is entirely stochastic so the resulting datasets likely will not be split in the exact specified percentage, but it will be within a relatively small margin of error.

| Setting | Description |
| :---: | --- |
| Input | The system path to the input directory containing the dataset images you would like to split. **Nothing else should be present in the directory apart from the dataset images.** |
| Output | The system path to the root output directory where you would like the split dataset to be exported to. A subdirectory each for `train`, `test`, and `val` will be created inside of this directory, into which the images will be placed. |
| Test | Split percentage for the `test` set. |
| Train | Split percentage for the `train` set. |
| Validate | Split percentage for the `val` set. |
| Start | Begins the splitting operation. This is also not threaded and, technically speaking, does lock the UI. But it is extremely fast. With smaller datasets, it oftentimes doesn't even appear that the UI locks because the copy operation is over so quickly. |
| Preview | Displays a preview of the results of the current copy operation were it to be run. |

---
*See Also:*
| [Toolbox - Home](../toolbox.md) | [Experiment](experiment.md) | [Dataset](dataset.md) | [Training](training.md) | [Testing](testing.md) | [Review](review.md) | [Settings](settings.md) |