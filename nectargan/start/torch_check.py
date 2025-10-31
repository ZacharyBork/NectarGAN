from __future__ import annotations

def validate_torch() -> None:
    '''Ensures that PyTorch is installed in the current environment.

    Raises:
        ImportError: If unable to import PyTorch, providing a link to the
            PyTorch download page.
    '''
    try:
        import torch
        print(f'Torch Version: {torch.__version__}')

        from torch import version as _tv
        print(f'CUDA Version: {getattr(_tv, 'cuda', None)}')
        
        if not getattr(torch, 'cuda', None) is None \
            and torch.cuda.is_available():
            print(f'CUDA Device Count: {torch.cuda.device_count()}')

    except Exception:
        raise ImportError(
            'Unable to locate PyTorch.\n\n'
            'PyTorch is required for NectarGAN but it is not installed ' \
            'automatically. Please visit the PyTorch website for instructions '
            'regarding PyTorch installation:\n\n'
            'https://pytorch.org/get-started/locally/\n')
    
if __name__ == "__main__":
    validate_torch()
