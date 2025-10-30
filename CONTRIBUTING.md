# Contributing to NectarGAN

Thank you for your interest in contributing to NectarGAN! This guide will explain how to propose changes, report issues, and set up a development environment.

---

## Ways to Contribute

- **Bug fixes:** *found a problem? See `Reporting Bugs` section below.*
- **New features:** *discuss via an issue before big changes.*
- **Documentation:** *improve docs, tutorials, FAQ etc.*
- **Tests:** *add or improve unit/integration tests.*
- **Examples:** *config files, loss specifications, etc.*

##  Contribution workflow:

1. Fork the project.
2. Create a new branch for your changes.
3. Commit your changes when finished.
4. Create a new pull request with tests/docs (if applicable).
5. Wait for review.

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