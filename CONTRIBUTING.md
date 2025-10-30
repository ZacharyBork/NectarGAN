# Contributing to NectarGAN

Thank you for your interest in contributing to NectarGAN! This guide will explain how to propose changes, report issues, and set up a development environment.

---

## Ways to Contribute

- **Bug fixes:** *found a problem? See `Reporting Bugs` section below.*
- **New features:** *discuss via an issue before big changes.*
- **Documentation:** *improve docs, tutorials, FAQ etc.*
- **Tests:** *add or improve unit/integration tests.*
- **Examples:** *config files, loss specifications, etc.*

##  Contribution workflow

1. Fork the project.
2. Create a new branch for your changes.
3. Commit your changes when finished.
4. Create a new pull request with tests/docs (if applicable).
5. Wait for review.

## Style guide

### Roughly follow [PEP 8](https://peps.python.org/pep-0008/) for code style:
- 4 spaces per indentation level.
- Line length: code ≤ 79.
- Individual package imports on separate lines.
- Avoid extra white space.

**Docstrings do not follow PEP 8.** See below.

### Imports should be grouped in the following order:
1) **Native libraries** (built-ins)  
2) **Third-party packages** (from PyPI)  
3) **Local/project modules** (this repo)

### Type hints:
- All public functions and class methods should have both their **input arguments**, and their **return value** type hinted in any API code.
- Private methods do not need to be type hinted.

### Docstrings ([Google](https://google.github.io/styleguide/pyguide.html#s3.8-comments-and-docstrings)):
- One-line summary on its own line.
- Sections: Args, Returns, Raises, Examples, etc.
- Public API functions and methods should have thorough docstrings. Please see [here](/nectargan/losses/loss_manager.py) and [here](/nectargan/trainers/pix2pix_trainer.py) for docstring examples.
- Docstrings should be wrapped to 79 characters for consistancy.

## Reporting Bugs

Before submitting a bug report, please search existing issues.

When opening a bug report, please include:

- **Steps to reproduce**
- **Expected vs. actual behavior**
- **Environment**: OS, Python, CUDA/driver/GPU, PyTorch, project version/commit.
- **Config**: the config file and/or flags to reproduce, if applicable.
- **Logs/traceback**: the error output, formatted in a code block.
- **Data info**: sample or a description sufficient to reproduce, if applicable.

### Report Checklist
- [ ] Reproduces on the latest `main`
- [ ] Minimal reproducible example (remove unrelated code/flags)
- [ ] No secrets or private data in logs/configs

> [!WARNING]
> If the issue may be a **security vulnerability**, please do **not** open a public. Please instead submit a report to the following email with the above data: `nectargan.dev@gmail.com`.

## Getting Help vs. Opening an Issue

Before filing, please search existing issues and discussions.

- **Bug report**: See above.
- **Feature request**: describe the use case, alternatives, and any prototype code.
- **Questions**: use Discussions or open a `Question` issue.

## Development Setup

**Requirements**
- Python **3.12+**
- (Optional) CUDA-capable GPU for GPU tests

**Create an environment**
```bash
# venv
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.dev.txt

# conda
conda env create -f environment.yml
conda activate nectargan
pip install -e .[dev]
```