# NectarGAN - Testing Your Installation
#### This document will guide you through the process of testing your NectarGAN installation to ensure the various components are functioning correctly.

Before we can run the tests, we first need to install the requisite packages. This can be accomplished by running the following command inside of the environment in which you installed NectarGAN:
```bash
pip install -e ".[dev]"
```

**This command will install the [pytest](https://pypi.org/project/pytest/) and [pytest-cov](https://pypi.org/project/pytest-cov/) packages**. After installation has completed, we can run:
```bash
pytest
```
This will begin the automatic testing which will take a few seconds. When it is complete, you will see something along the lines of:
```bash
================================================== 8 passed in 9.66s ==================================================
```
If the testing was successful. Otherwise you will get an assertation error informing you which component is failing. The testing suite currently covers:

- General module imports for critical components.
- Testing of the loss management framework.
- Testing of the UNet and PatchGAN models.
- Validation of the Scheduler module and the base schedule functions. 

---
