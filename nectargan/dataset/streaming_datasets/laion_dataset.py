
import requests
from os import PathLike
from pathlib import Path
from PIL import Image
from io import BytesIO
from typing import Any

import torch
from torch.utils.data import IterableDataset, DataLoader
from torchvision import transforms
from huggingface_hub import login
from datasets import load_dataset

from nectargan.config import DiffusionConfig
from nectargan.config.utils import config_from_file

TAGS = {
    'aesthetics': {
        'v2-4.5': 'laion/aesthetics_v2_4.5',
        'v2-5.0': 'laion/aesthetics_v2_5.0',
        'v2-6.25plus': 'xingjianleng/laion_aesthetics_v2_6.25plus',
        'square': 'opendiffusionai/laion2b-en-aesthetic-square',
        'square-cleaned': \
            'opendiffusionai/laion2b-en-aesthetic-square-cleaned'
    },
    '400m': { 'relaion400m': 'laion/relaion400m' },
    '5b-subsets': {
        'relaion2B-en-research': 'laion/relaion2B-en-research',
        'relaion2B-multi-research': 'laion/relaion2B-multi-research',
        'relaion1B-nolang-research': 'laion/relaion1B-nolang-research',
        'laion2B-en': 'laion/laion2B-en',
        'laion2B-multi': 'laion/laion2B-multi',
        'laion1B-nolang': 'laion/laion1B-nolang'
    },
    '5b': { '5b-research-safe': 'laion/re-laion-5b-research-safe' }
}

class LAIONDataset(IterableDataset):
    def __init__(
            self, 
            config: DiffusionConfig, 
            dataset: str,
            subset: str,
            split: str='train',
            min_aesthetic_score: float=6.0,
            max_caption_length: int=77,
            max_samples: int | None=None,
            cache_dir: PathLike | None=None,
            timeout: float=5.0,
            require_login: bool=False,
            cache_builder: bool=False,
            silent: bool=True
        ) -> None:
        if require_login: login()
        self.config = config
        self.max_caption_length = max_caption_length
        self.max_samples = max_samples
        self.cache_dir = cache_dir
        self.load_size = self.config.model.input_size
        self.timeout = timeout
        self.cache_builder = cache_builder
        self.silent = silent
        
        self.transform = transforms.Compose([
            transforms.RandomCrop(size=(self.load_size, self.load_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        dataset_tag = self._get_dataset_tag(dataset, subset)
        self._init_dataset(
            dataset_tag=dataset_tag, split=split,
            min_aesthetic_score=min_aesthetic_score)
        
    def _get_dataset_tag(self, set: str, subset: str) -> None:
        return TAGS[set][subset]

    def _init_dataset(
            self, 
            dataset_tag: str,
            split: str, 
            min_aesthetic_score: float=6.0
        ) -> None:
        dataset = load_dataset(
            dataset_tag, split=split, streaming=True, cache_dir=self.cache_dir)
        def aesthetic_filter(example):
            return (example['aesthetic'] > min_aesthetic_score)
        dataset = dataset.filter(aesthetic_filter)
        self.dataset = dataset
        self.length = self.dataset.info.splits.get(
            split, self.dataset.info.splits.get('train')
        ).num_examples if hasattr(self.dataset.info, 'splits') else None

    def __len__(self) -> int:
        return self.length

    def __iter__(self):
        sample_count = 0
        for item in self.dataset:
            if not self.max_samples is None and \
               sample_count >= self.max_samples:
                break
            url = item.get('URL')
            caption = item.get('TEXT')
            if len(caption) > self.max_caption_length: continue

            try:
                response = requests.get(url, timeout=self.timeout)
                response.raise_for_status()
                image = Image.open(BytesIO(response.content)).convert('RGB')
            except Exception as e:
                if not self.silent: 
                    print(f'Failed to load image: {e}. Skipping...')
                continue
            
            x = lambda y, z: int(round(y*z))
            scale = self.load_size / min(image.width, image.height)
            resolution = (x(image.width, scale), x(image.height, scale))
            image = image.resize(resolution, resample=Image.Resampling.BICUBIC)
            t = self.transform(image)

            sample_count += 1
            if not self.cache_builder: yield t, caption 
            else: yield t, caption, Path(url).stem

    def collate(
            self, 
            batch: list[tuple[torch.Tensor, str]]
        ) -> dict[str, Any]:
        images, captions, names = zip(*batch)
        if not self.cache_builder:
            return { 'image': torch.stack(images), 'caption': list(captions) }
        else:
            return { 
                'image': torch.stack(images), 
                'caption': list(captions),
                'name': list(names) }

if __name__ == "__main__":
    config = config_from_file(
        '/media/zach/UE/ML/NectarGAN/nectargan/config/defaults/diffusion.json')
    dataset = LAIONDataset(
        config=config, 
        dataset='aesthetics', subset='square',
        cache_builder=True)
    dataloader = DataLoader(
        dataset, batch_size=8, collate_fn=dataset.collate, num_workers=0)
    for batch in dataloader: 
        tensor = batch['image']
        caption = batch['caption']
        names = batch['name']
        print(names)
        break