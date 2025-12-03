import json
from os import PathLike
from pathlib import Path

from nectargan.dataset.data import TextEmbeddedMetadata

def load_metadata_file(metadata_file: PathLike) -> TextEmbeddedMetadata:
    metadata_file = Path(metadata_file)
    if not metadata_file.exists():
        raise FileNotFoundError(
            f'Unable to locate metadata file at path: '
            f'{metadata_file.as_posix()}')
    with open(metadata_file, 'r') as file:
        metadata = json.loads(file.read())
    metadata = TextEmbeddedMetadata(
        schema_version=metadata['info']['schema_version'],
        total_captions=metadata['info']['total_captions'],
        total_images=metadata['info']['total_images'],
        items=metadata['items'])
    return metadata


