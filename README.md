# GPyReg
![Version](https://img.shields.io/badge/dynamic/json?label=python&query=info.requires_python&url=https%3A%2F%2Fpypi.org%2Fpypi%2Fgpyreg%2Fjson)
[![Conda](https://img.shields.io/conda/v/conda-forge/gpyreg)](https://anaconda.org/conda-forge/gpyreg)
[![PyPI](https://img.shields.io/pypi/v/gpyreg)](https://pypi.org/project/gpyreg/)
<br />
[![Discussion](https://img.shields.io/badge/-discussion-blue?logo=github)](https://github.com/orgs/acerbilab/discussions)
[![tests](https://img.shields.io/github/actions/workflow/status/acerbilab/gpyreg/tests.yml?branch=main&label=tests)](https://github.com/acerbilab/gpyreg/actions/workflows/tests.yml)
[![docs](https://img.shields.io/github/actions/workflow/status/acerbilab/gpyreg/build.yml?branch=main&label=docs)](https://github.com/acerbilab/gpyreg/actions/workflows/docs.yml)
[![build](https://img.shields.io/github/actions/workflow/status/acerbilab/gpyreg/docs.yml?branch=main&label=build)](https://github.com/acerbilab/gpyreg/actions/workflows/build.yml)
### What is it?
GPyReg is a lightweight package for Gaussian process regression in Python. It was developed for use with [PyVBMC](https://github.com/acerbilab/pyvbmc) (a Python package for efficient black-box Bayesian inference) but is usable as a standalone package.

### Documentation
The documentation is currently hosted on [github.io](https://acerbilab.github.io/gpyreg/).

## Installation
GPyReg is available via `pip` and `conda-forge`:
```console
python -m pip install gpyreg
```
or:
```console
conda install --channel=conda-forge gpyreg
```
GPyReg requires Python version 3.9 or newer.

## Troubleshooting and contact

If you have trouble doing something with GPyReg, spot bugs or strange behavior, or you simply have some questions, please feel free to:
- Post in the lab's [Discussions forum](https://github.com/orgs/acerbilab/discussions) with questions or comments about GPyReg, your problems & applications;
- [Open an issue](https://github.com/acerbilab/gpyreg/issues/new) on GitHub;
- Contact the project lead at <luigi.acerbi@helsinki.fi>, putting 'GPyReg' in the subject of the email.

You can also demonstrate your appreciation for GPyReg in the following ways:

- *Star :star:* the repository on GitHub;
- [Subscribe](http://eepurl.com/idcvc9) to the lab's newsletter for news and updates (new features, bug fixes, new releases, etc.);
- [Follow Luigi Acerbi on Twitter](https://twitter.com/AcerbiLuigi) for updates about our other projects;

If you are interested in applications of Gaussian process regression to Bayesian inference and optimization, you may also want to check out [PyVBMC](https://github.com/acerbilab/pyvbmc) for efficient black-box inference, and [Bayesian Adaptive Direct Search](https://github.com/acerbilab/bads) (BADS), our method for fast Bayesian optimization. BADS is currently available only in MATLAB, but a Python version will be released soon.

### License

GPyReg is released under the terms of the [BSD 3-Clause License](LICENSE).

### Acknowledgments

GPyReg was developed by [members](https://www.helsinki.fi/en/researchgroups/machine-and-human-intelligence/people) (past and current) of the [Machine and Human Intelligence Lab](https://www.helsinki.fi/en/researchgroups/machine-and-human-intelligence/) at the University of Helsinki. Development is being supported by the Academy of Finland Flagship programme: [Finnish Center for Artificial Intelligence FCAI](https://fcai.fi/).
