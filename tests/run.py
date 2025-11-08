import subprocess
from pathlib import Path

def run_tests() -> None:
    root = Path(__file__).parent.resolve()
    try:
        proc = subprocess.Popen([
            'pytest', 
            f'{Path(root, 'unit').as_posix()}', 
            f'{Path(root, 'integration').as_posix()}'],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
            text=True, bufsize=1)
    except Exception as e:
        raise RuntimeError(
            'Encountered and exception while attempting to start test suite'
        ) from e
    for line in proc.stdout: print(line, end='')
    proc.wait()

if __name__ == "__main__":
    run_tests()
