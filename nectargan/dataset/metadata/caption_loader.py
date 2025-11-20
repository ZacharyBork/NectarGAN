import sys
import json
from os import PathLike
from pathlib import Path
from typing import Any

# example = {
#     'image_path': Path,
#     'captions': list[str]
# }
# output = list[example1, example2, example3]

def print_progress(
        current: int,
        max: int
    ) -> None:
    sys.stdout.write('\x1b[1A')
    sys.stdout.write('\x1b[2K')

    progress = float(current) / float(max)
    progress_msg = (
        f'Progress: {current} / {max} | '
        f'{round(progress*100.0, 2)}% ')
    print(progress_msg)

class CaptionLoader():
    def __init__(
            self, 
            dataroot: PathLike,
            metadata_file: PathLike,
            image_key: str='images',
            caption_key: str='annotations',
            new_file_suffix: str='rebuilt'
        ) -> None:
        self.dataroot = Path(dataroot)
        self.metadata_file = Path(metadata_file)
        self.image_key = image_key
        self.caption_key = caption_key
        self.new_file_suffix = new_file_suffix

    def _validate_paths(self) -> None:
        print('CaptionLoader: Validating input paths...')
        if not self.dataroot.exists():
            raise FileNotFoundError(
                f'Unable to locate dataroot at path: '
                f'{self.dataroot.as_posix()}')
        if not self.metadata_file.exists():
            raise FileNotFoundError(
                f'Unable to locate metadata file at path: '
                f'{self.metadata_file.as_posix()}')
        
    def _build_output_path(self) -> bool:
        metadata_directory = self.metadata_file.parent.resolve()
        name = f'{self.metadata_file.stem}_{self.new_file_suffix}'
        self.output_path = Path(metadata_directory, f'{name}.json') 
        if self.output_path.exists():
            print(f'CaptionLoader: Found existing metadata file at path: '
                  f'{self.output_path.as_posix()}')
            return True
        return False

    def _load_metadata_file(self) -> tuple[list[dict[str, Any]]]:
        print('CaptionLoader: Extracting metadata...')
        with open(self.metadata_file, 'r') as file:
            metadata = json.loads(file.read())
        try: images = metadata[self.image_key]
        except KeyError:
            raise KeyError(
                f'Image key not found in metadata file: {self.image_key}')
        try: captions = metadata[self.caption_key]
        except KeyError:
            raise KeyError(
                f'Captions key not found in metadata file: {self.caption_key}')
        return images, captions
    
    def _handle_zero_captions(self, image_file: Path) -> bool:
        print(f'Image found with no metadata!\n'
              f'File: {image_file.as_posix()}\n')
        while True:
            print(f'Bypass image and continue?')
            confirm = input('y | n -> ').strip().casefold()
            match confirm:
                case 'y': return True
                case 'n': exit('Operation canceled')
                case _: print('Input not valid.')

    def _build_new_metadata(
            self, 
            images: list[dict[str, Any]], 
            captions: list[dict[str, Any]]
        ) -> list[dict[str, Any]]:
        print('CaptionLoader: Building new metadata...')
        metadata = []
        img_count = len(images)
        for idx, image in enumerate(images):
            image_id = image['id']
            filepath = Path(self.dataroot, image['file_name'])
            
            image_captions = []
            for caption in captions:
                if not image_id == caption['image_id']: continue
                image_captions.append(caption['caption'])
                captions.remove(caption)
            if len(image_captions) == 0: 
                self._handle_zero_captions()
                continue
                            
            metadata.append({
                'filepath': filepath.as_posix(),
                'captions': image_captions})
            
            print_progress(idx+1, img_count)
        
        return metadata

    def _write_metadata_file(
            self, 
            new_metadata: list[dict[str, Any]]
        ) -> None:
        print('CaptionLoader: Writing new metatdata file...')
        with open(self.output_path, 'w') as file:
            file.write(json.dumps(new_metadata))

    def load(self) -> Path:
        self._validate_paths()
        if not self._build_output_path():
            images, captions = self._load_metadata_file()
            new_metadata = self._build_new_metadata(images, captions)
            self._write_metadata_file(new_metadata)
        return self.output_path

if __name__ == "__main__":
    root = Path()
    loader = CaptionLoader(
        dataroot=Path(root, 'val2017'),
        metadata_file=Path(root, 'annotations/captions_val2017.json'))
    loader.load()
