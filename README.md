# GPyReg
![Version](https://img.shields.io/badge/dynamic/json?label=python&query=info.requires_python&url=https%3A%2F%2Fpypi.org%2Fpypi%2Fgpyreg%2Fjson)
[![Conda](https://img.shields.io/conda/v/conda-forge/gpyreg)](https://anaconda.org/conda-forge/gpyreg)
[![PyPI](https://img.shields.io/pypi/v/gpyreg)](https://pypi.org/project/gpyreg/)
<br />
[![Discussion](https://img.shields.io/badge/-discussion-blue?logo=github)](https://github.com/orgs/acerbilab/discussions)
[![tests](https://img.shields.io/github/actions/workflow/status/acerbilab/gpyreg/tests.yml?branch=main&label=tests)](https://github.com/acerbilab/gpyreg/actions/workflows/tests.yml)
[![docs](https://img.shields.io/github/actions/workflow/status/acerbilab/gpyreg/docs.yml?branch=main&label=docs)](https://github.com/acerbilab/gpyreg/actions/workflows/docs.yml)
[![build](https://img.shields.io/github/actions/workflow/status/acerbilab/gpyreg/build.yml?branch=main&label=build)](https://github.com/acerbilab/gpyreg/actions/workflows/build.yml)
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
- Follow Luigi Acerbi on [X](https://x.com/AcerbiLuigi) or [Bluesky](https://bsky.app/profile/lacerbi.bsky.social) for updates about our other projects;

If you are interested in applications of Gaussian process regression to Bayesian inference and optimization, you may also want to check out [PyVBMC](https://github.com/acerbilab/pyvbmc) for efficient black-box inference, and [PyBADS](https://github.com/acerbilab/pybads), the Python implementation of Bayesian Adaptive Direct Search (BADS), our method for fast Bayesian optimization.

### License

GPyReg is released under the terms of the [BSD 3-Clause License](LICENSE).

### Acknowledgments

GPyReg is developed by [members](https://www.helsinki.fi/en/researchgroups/machine-and-human-intelligence/people) (past and current) of the [Machine and Human Intelligence Lab](https://www.helsinki.fi/en/researchgroups/machine-and-human-intelligence/) at the University of Helsinki and [ELLIS Institute Finland](https://www.ellisinstitute.fi/). Development of GPyReg from version 1.1 onwards has been assisted by coding agents, including Anthropic's [Claude Fable 5.1](https://www.anthropic.com/claude-fable-and-mythos-5-1) and OpenAI's [GPT-6 Astra](https://developers.openai.com/api/docs/models/gpt-6-astra).
Work on the GPyReg package is supported by the Research Council of Finland (grants 356498 and 358980 to Luigi Acerbi) and its Flagship programme: [Finnish Center for Artificial Intelligence FCAI](https://fcai.fi/).
