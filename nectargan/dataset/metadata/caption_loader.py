import sys
import json
from os import PathLike
from pathlib import Path
from typing import Any, Literal

class CaptionLoader():
    def __init__(self, schema_version: int=1, silent: bool=False) -> None:
        self.schema_version = schema_version
        self.silent = silent

    def print_progress(
            self,
            current: int,
            max: int
        ) -> None:
        sys.stdout.write('\x1b[1A')
        sys.stdout.write('\x1b[2K')

        progress = float(current) / float(max)
        progress_msg = (
            f'Progress: {current} / {max} | '
            f'{round(progress*100.0, 2)}%')
        print(progress_msg)

class CaptionLoader_CUB200(CaptionLoader):
    # Captions: https://github.com/reedscot/cvpr2016
    #  Images : https://data.caltech.edu/records/65de6-vp158
    def __init__(
            self,
            image_root: Path,
            caption_root: Path,
            **kwargs
        ) -> None:
        super().__init__(**kwargs)
        self.image_root = image_root
        self.caption_root = caption_root

    def _validate_paths(self) -> None:
        if not self.silent: print('CaptionLoader: Validating input paths...')
        if not self.image_root.exists():
            raise FileNotFoundError(
                f'Unable to locate image root directory at path: '
                f'{self.image_root.as_posix()}')
        if not self.caption_root.exists():
            raise FileNotFoundError(
                f'Unable to locate caption root directory at path: '
                f'{self.caption_root.as_posix()}')
        
    def _build_output_path(self) -> bool:
        root_dir = self.image_root.parent
        self.metadata_file = Path(root_dir, 'metadata.json')
        if self.metadata_file.exists():
            if not self.silent: 
                print(f'CaptionLoader: Found existing metadata file at path: '
                      f'{self.metadata_file.as_posix()}')
            return True
        return False

    def _build_new_metadata(self) -> dict[str, Any]:
        metadata = { 
            'info': { 
                'schema_version': self.schema_version,
                'total_captions': 0, 
                'total_images': 0 },
            'items': {},
            'other': {}}
        img_count = len(list(self.image_root.rglob('*.jpg')))
        if not self.silent: print('CaptionLoader: Building new metadata...')
        for subdirectory in list(self.image_root.iterdir()):
            if not subdirectory.is_dir(): continue
            for image in list(subdirectory.glob('*.jpg')):
                caption_file = Path(
                    self.caption_root, subdirectory.name, f'{image.stem}.txt')
                assert caption_file.exists()
                with open(caption_file, 'r') as f:
                    image_captions = [
                        i for i in f.read().split('\n') if not i == '']
                metadata['info']['total_captions'] += len(image_captions)
                metadata['info']['total_images'] += 1
                metadata['items'][image.stem] = {
                    'filepath': image.resolve().as_posix(),
                    'captions': image_captions}
                if not self.silent: 
                    self.print_progress(
                        metadata['info']['total_images'], img_count)
        return metadata

    def _write_metadata_file(
            self, 
            new_metadata: dict[str, Any]
        ) -> None:
        if not self.silent: 
            print('CaptionLoader: Writing new metatdata file...')
        with open(self.metadata_file, 'w') as file:
            file.write(json.dumps(new_metadata))

    def load(self) -> Path:
        self._validate_paths()
        if not self._build_output_path():
            new_metadata = self._build_new_metadata()
            self._write_metadata_file(new_metadata)
        return self.metadata_file

class CaptionLoader_COCO(CaptionLoader):
    def __init__(
            self, 
            dataroot: PathLike,
            metadata_file: PathLike,
            image_key: str='images',
            caption_key: str='annotations',
            new_file_suffix: str='REBUILT',
            bypass_mode: Literal['allow', 'disallow', 'confirm']='confirm',
            **kwargs
        ) -> None:
        '''Initializes a COCO-style caption loader.
        
        Args:
            dataroot : The path to the directory containing the image files
                which the metadata file describes.
            metadata_file : The path to the metadata file to parse.
            image_key : The lookup key which points to the "images" data in the 
                metadata file.
            caption_key : The lookup key which points to the "captions" data in 
                the metadata file.
            new_file_suffix : The suffix which will be appended to the newly
                created metadata file. For example, if your base file is called
                "my_metadata.json", and new_file_suffix is "REBUILT", the new
                metadata file will be called "my_metadata_REBUILT.json"
            bypass_mode : Defines loader behavior in the event it encounters an
                image with no associated captions. Modes:
                    - "allow"    : Silently bypass the image.
                    - "disallow" : Stop and raise an exception.
                    - "confirm"  : Stop and wait for manual bypass.
            schema_version : The current metadata schema version. This is not
                used currently, but may be in the future.
            silent : If True, the converter will not print any progress updates
                to the console.
        '''
        super().__init__(**kwargs)
        self.dataroot = Path(dataroot)
        self.metadata_file = Path(metadata_file)
        self.image_key = image_key
        self.caption_key = caption_key
        self.new_file_suffix = new_file_suffix
        self.bypass_mode = bypass_mode

    def _validate_paths(self) -> None:
        if not self.silent: print('CaptionLoader: Validating input paths...')
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
            if not self.silent: 
                print(f'CaptionLoader: Found existing metadata file at path: '
                      f'{self.output_path.as_posix()}')
            return True
        return False

    def _load_metadata_file(self) -> tuple[list[dict[str, Any]]]:
        if not self.silent: print('CaptionLoader: Extracting metadata...')
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

    def _build_caption_map(
            self, 
            captions: list[dict[str, Any]]
        ) -> dict[str, list[str]]:
        if not self.silent: print('CaptionLoader: Building caption map...')
        caption_map = {}
        for idx, caption in enumerate(captions):
            try: entry = caption_map[caption['image_id']]
            except KeyError: entry = caption_map[caption['image_id']] = []
            entry.append(caption['caption'])
            if not self.silent: self.print_progress(idx+1, len(captions))
        return caption_map

    def _build_new_metadata(
            self, 
            images: list[dict[str, Any]], 
            caption_map: dict[str, Any]
        ) -> dict[str, Any]:
        if not self.silent: print('CaptionLoader: Building new metadata...')
        metadata = { 
            'info': { 
                'schema_version': self.schema_version,
                'total_captions': 0, 
                'total_images': 0 },
            'items': {},
            'other': {}}
        img_count = len(images)
        for image in images:
            image_id = image['id']
            file_name = image['file_name']
            filepath = Path(self.dataroot, file_name)

            try: image_captions = caption_map[image_id]
            except KeyError:
                if self.silent: continue
                match self.bypass_mode:
                    case 'allow': pass
                    case 'disallow':
                        raise RuntimeError(
                            f'Encountered image with no associated captions!\n'
                            f'Image path: {filepath.as_posix()}\n\n'
                            f'Stopping...')
                    case 'confirm': self._handle_zero_captions(filepath)
                continue

            metadata['info']['total_captions'] += len(image_captions)
            metadata['info']['total_images'] += 1
            metadata['items'][file_name.split('.')[0]] = {
                'filepath': filepath.as_posix(),
                'captions': image_captions}
            if not self.silent: 
                self.print_progress(
                    metadata['info']['total_images'], img_count)
        
        return metadata

    def _write_metadata_file(
            self, 
            new_metadata: dict[str, Any]
        ) -> None:
        if not self.silent: 
            print('CaptionLoader: Writing new metatdata file...')
        with open(self.output_path, 'w') as file:
            file.write(json.dumps(new_metadata))

    def load(self) -> Path:
        self._validate_paths()
        if not self._build_output_path():
            images, captions = self._load_metadata_file()
            caption_map = self._build_caption_map(captions)
            new_metadata = self._build_new_metadata(images, caption_map)
            self._write_metadata_file(new_metadata)
        return self.output_path

if __name__ == "__main__":
    root = Path('/media/zach/UE/ML/test_data/diffusion/CUB200')
    loader = CaptionLoader_CUB200(
        image_root=Path(root, 'images'),
        caption_root=Path(root, 'text_c10'))
    loader.load()
