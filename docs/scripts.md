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
