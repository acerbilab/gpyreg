# GPyReg
### What is it?
GPyReg is a lightweight package for Gaussian process regression in Python. It was developed for use with [PyVBMC](https://github.com/acerbilab/pyvbmc) (a Python package for efficient black-box Bayesian inference) but is usable as a standalone package.

### Documentation
The documentation is currently hosted on [github.io](https://acerbilab.github.io/gpyreg/).

## Installation
GPyReg is not yet available on `pip`/`conda-forge`, but can be installed in a few steps:

1. Clone the GitHub repo locally:
   ```console
   git clone https://github.com/acerbilab/gpyreg
   ```
2. (Optional) Create a new environment in `conda` and activate it. We recommend using Python 3.9 or newer, but older versions *might* work:
   ```console
   conda create --name gpyreg-env python=3.9
   conda activate gpyreg-env
   ```
3. Install the repos:
   ```console
   cd ./gpyreg
   pip install -e .
   ```
## Troubleshooting and contact

If you have trouble doing something with GPyReg, spot bugs or strange behavior, or you simply have some questions, please feel free to:
- [open an issue](https://github.com/acerbilab/gpyreg/issues/new) on GitHub, or,
- contact the project lead at <luigi.acerbi@helsinki.fi>, putting 'GPyReg' in the subject of the email.

You can also demonstrate your appreciation for GPyReg in the following ways:

- *Star :star:* the repository on GitHub;
- [Follow Luigi Acerbi on Twitter](https://twitter.com/AcerbiLuigi) for updates about our other projects;

If you are interested in applications of Gaussian process regression to Bayesian inference and optimization, you may also want to check out [PyVBMC](https://github.com/acerbilab/pyvbmc) for efficient black-box inference, and [Bayesian Adaptive Direct Search](https://github.com/acerbilab/bads) (BADS), our method for fast Bayesian optimization. BADS is currently available only in MATLAB, but a Python version will be released soon.
