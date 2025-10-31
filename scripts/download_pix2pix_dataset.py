# NOTE: THIS FILE PULLS FROM https://efrosgans.eecs.berkeley.edu/pix2pix/datasets/
#
# They very kindly host these datasets to assist in research and development of 
# ML models. Please be kind to their servers!
#
# Most of these datasets are also available on Hugging Face if you'd prefer:
# https://huggingface.co/datasets/huggan/facades
# https://huggingface.co/datasets/huggan/cityscapes
# https://huggingface.co/datasets/huggan/edges2shoes
# https://huggingface.co/datasets/huggan/maps
# https://huggingface.co/datasets/huggan/night2day

from os import PathLike
from pathlib import Path
import requests
import tarfile
import argparse

def get_dataset_name() -> str:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', '--dataset', type=str, default='facades',
        choices=[
            'cityscapes', 'edges2handbags', 'edges2shoes', 
            'facades', 'maps', 'night2day'],
        help=f'Specify which dataset to download'
    )
    args = parser.parse_args()
    return args.dataset

def build_download_directory() -> Path:
    root = Path(__file__).parent.parent
    download_dir = Path(root, 'data/datasets')
    if not download_dir.exists(): 
        print(f'Creating download directory: {download_dir}')
        download_dir.mkdir(parents=True)
    else: print('Download directory exists. Skipping creation.')
    return download_dir

def check_final_output_path(final_export_dir: Path) -> None:
    if final_export_dir.exists():
        print(f'Output directory already exists at {download_dir}')
        print('Please delete it to continue.')
        print('Stopping...')
        exit(1)

def download_file(
        url: str, 
        target_dir: PathLike, 
        filename: str, 
        chunk_size: int=8192
    ) -> Path:
    filepath = Path(target_dir, filename)

    if filepath.exists():
        msg = f'File already exists: {filepath.as_posix()}\nStopping Download.'
        print(msg)
        return filepath

    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = float(response.headers.get('content-length', 0)) * 0.001
        current = 0.0
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                f.write(chunk)
                current = round(current + float(chunk_size) * .001, 2)
                print(f'Downloaded: {current}KB / {round(total_size, 2)}KB')
        print(f'File downloaded: {Path(download_dir, filename).as_posix()}')
    except requests.exceptions.RequestException as e:
        print(f'Error downloading file: {e}')        
        if filepath.exists(): filepath.unlink()
        exit(1)  
    return filepath

def extract_dataset(archive_path: PathLike, export_directory: PathLike) -> None:
    print('Beginning dataset extraction.')
    try:
        if tarfile.is_tarfile(archive_path):
            with tarfile.open(archive_path, 'r:*') as tar:
                tar.extractall(path=export_directory)
        else:
            msg = f'Warning: Unknown archive type for file: {archive_path}'
            raise TypeError(msg)
    except Exception as e:
        raise RuntimeError(f'Unable to extract dataset archive.') from e

if __name__ == "__main__":
    dataset_name = get_dataset_name()

    download_dir = build_download_directory()
    final_export_dir = Path(download_dir, dataset_name)
    check_final_output_path(final_export_dir)
    
    root_url = 'https://efrosgans.eecs.berkeley.edu'
    url = root_url + f'/pix2pix/datasets/{dataset_name}.tar.gz'
    filename = url.split('/')[-1]
    
    print(f'Preparing to download dataset archive from: {url}')
    archive_path = download_file(url, download_dir, filename)

    extract_dataset(archive_path, download_dir)
    print(f'Extraction complete. Path: {final_export_dir.as_posix()}')

    print('Removing archive file.')
    if archive_path.exists(): archive_path.unlink()
    print(f'Archive file removed: {archive_path.as_posix()}')

    exit(0)
   