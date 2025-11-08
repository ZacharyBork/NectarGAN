# NectarGAN – Scripts
Here you will find documentation related to the additional helper scripts included with NectarGAN.
## Pix2pix Dataset Download Script
**This script allows you to automatically download premade Pix2pix datasets.** 

> [!IMPORTANT]
> This script requires the Python `requests` module (`python -m pip install requests`).

> [!NOTE]
> This script pulls from: https://efrosgans.eecs.berkeley.edu/pix2pix/datasets/
> 
> They very kindly host these datasets to assist in research and development of ML models. **Please be kind to their servers!**
> 
> Most of these datasets are also available on Hugging Face if you'd prefer:<br>
> https://huggingface.co/datasets/huggan/facades<br>
> https://huggingface.co/datasets/huggan/cityscapes<br>
> https://huggingface.co/datasets/huggan/edges2shoes<br>
> https://huggingface.co/datasets/huggan/maps<br>
> https://huggingface.co/datasets/huggan/night2day

### Usage
From the project root, run:
```bash
python scripts/dowload_pix2pix_dataset.py -d facades
```
### Flags
#### `-d` / `--dataset`:

Allows you to specify a dataset to download. Valid options are:

- `cityscapes`
- `edges2handbags`
- `edges2shoes`
- `facades`
- `maps`
- `night2day`

If no value is supplied, the script will default to `facades`.

### Result
The dataset will be downloaded as a `.tar.gz` to `NectarGAN/data/datasets`, then the dataset files will be extracted to a new subdirectory of `NectarGAN/data/datasets` named after the dataset. After the files are extracted, the archive is deleted automatically, and you are left with just the directory containing the dataset files.

## Loss Graph Generator
**Generates publication-ready graphs from experiment loss logs.**

This script will parse the loss log from a provided experiment directory and use the data contained within to build a graph of the loss values and, optionally, loss weights, which can then be displayed on screen interactively, or exported as `.png` image files to a `graphs` directory inside of the given experiment directory.

***This script is highly configurable,*** and can be used to generate various graph arrangements depending upon the combination of flags provided. These include:

| Type | Description |
| --- | --- |
| One&nbsp;Graph&nbsp;per&nbsp;Loss | For each loss (and its weight, if selected) one unique graph will be generated, which will have its Y axis normalized to the values of just the given loss function. |
| Total&nbsp;Combined&nbsp;Loss | All losses (and, optionally, their weights) will be graphed on to a single graph titled `Combined Loss`. Good for quick global cost comparisons, usually bad for granular data visualization. Consider using the `-ls` flag to apply logarithmic scaling to the graph for readability. |
| Split&nbsp;G&nbsp;and&nbsp;D | Creates one graph for generator losses, and another for discriminator losses. The graphs will be stacked vertically and each will be titled according to the network whos losses are being graphed. An example of this type of loss graph can be seen [here](https://github.com/ZacharyBork/NectarGAN/blob/main/examples/UNet/basic/loss_graphs.png). |

### Running the Script
**All commands should be run from the project root (i.e. `Nectargan/`).**

*Note: For each command, replace `"/path/to/experiment"` with the actual system path to the experiment directory containing the loss logs you would like to graph.*

```bash
python scripts/build_loss_graphs.py -e "/path/to/experiment"
```
### Common Usages
**Graph all datapoints from loss log:**
```bash
python scripts/build_loss_graphs.py -e "/path/to/experiment" -f 1
```
**Export one graph per loss:**
```bash
python scripts/build_loss_graphs.py -e "/path/to/experiment" -og
```
**Build log-scaled graph for G and D:**
```bash
python scripts/build_loss_graphs.py -e "/path/to/experiment" -ls -s
```
**Combine everything, but make it as legible as possible:**
```bash
python scripts/build_loss_graphs.py -e "/path/to/experiment" -ls -w
```
### Arguments / Flags
**There is only one required flag, `-e`,** which should be set to the system path to the experiment directory containing the loss log you would like to graph the data from. However, there are many optional arguments which can alter the behaviour and output of the script. These are:

### `-f` / `--sample_frequency`
- **Description:** The value provided for this flag will be used to determine the frequency with which to sample the values for each loss from the loss loss. For example, say you had a dataset with 100 images in it, and you trained on that data for 100 epochs with loss logging enabled. Each set of loss values (and corresponding weights) would have 10,000 total values (`100 epochs` x `100 iterations`). 

    That can become a touch illegible on graphs which may only be a few inches wide. So this value can be used to tell the script to only graph every `x` number of values from the set. When reading the log, if the `iteration of the current value` % `--sample_frequency` != `0.0`, the value will not be graphed.
- **Type:** int
- **Default:** 50

### `-W` / `--graph_width`
- **Description:** This is the width of the graph (in inches). This, along with `-H` and `-dpi` are used to control the final output resolution of the graph image (and the on-screen display, if using `-p` / `--preview`).
- **Type:** float
- **Default:** 14.0

### `-H` / `--graph_height`
- **Description:** See `--graph_width`.
- **Type:** float
- **Default:** 7.0

### `-dpi` / `--output_dpi`
- **Description:** Controls the DPI of the output image. This, along with `-W` and `-H` control the final resolution of the output image. The default values will create a final image (for the split G and D graphs) which has a resolution of `3469 x 2012`, and an approximate file size of `1MB`.
- **Type:** int
- **Default:** 300

### `-lw` / `--line_width`
- **Description:** The width of the lines on the graph. Higher values = thicker lines.
- **Type:** float
- **Default:** 1.0

### `-og` / `--one_graph_per_loss`
- **Description:** If this flag is provided, one unique graph will be generated and exported (or displayed) per loss found in the loss log, rather than the default behaviour of combining many or all losses on to a single graph.
- **Type:** flag

### `-ls` / `--use_log_scaling`
- **Description:** If this flag is provided, generated graphs will use logarithmic scaling, rather than the default linear scaling. This is useful when combining many losses on to a single graph which have a wide variance in their total value ranges.
- **Type:** flag

### `-s` / `--split_g_and_d`
- **Description:** If this flag is provided, the losses for the generator and the discrimiator will be split on to separate graphs. They will be displayed and exported as a single figure (iamge file), however. This flag does nothing if the `-og` / `--one_graph_per_loss` flag is provided.
- **Type:** flag

**Note: If you are using your own custom loss subspec, the split mechanism triggered by this flag expects that the loss names in the log will match the standard pattern of `{network_tag}_{loss_name}` (i.e. `G_GAN`, `D_real`). _Losses which do not match this pattern will not be graphed!_**

### `-w` / `--graph_weights`
- **Description:** If this flag is provided, the weight values for each loss will be graphed along with the actual loss values. This can be useful when building and testing complex weight schedules, but sometimes requires that the `-ls` flag be provided for the graph to be legible (e.g. `L1` loss, which often has a standard lambda of `100.0`, will oftentimes dwarf the actual L1 loss values causing the loss graph to appear compressed).
- **Type:** flag

### `-p` / `--preview`
- **Description:** If this flag is provided, rather than exporting the graphs as `.png` to the `graphs` directory in the provided experiment directory, the graphs will instead be displayed on screen in a new window. If the `-og` flag is also provided, this will currently open one window per loss graph. This may be changed in a future update though.
- **Type:** flag

