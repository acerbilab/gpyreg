## What this is

GPyReg is a lightweight Gaussian process regression library (NumPy/SciPy, no autodiff framework). It is a Python port of the MATLAB toolbox `gplite` kept under `matlab/gplite/` for reference, and it is the GP backend of PyVBMC and PyBADS, so its public API and numerical behavior are downstream-visible. PyVBMC requires the latest gpyreg release, and moves its minimum gpyreg version and its CI pin to each one. PyBADS's minimum gpyreg names one release and its CI runs at that release's tagged commit; it moves both to a new release only after comparing its benchmark runs under that release with its reference runs, and its scheduled CI (the 13th and 28th of each month) runs against gpyreg's `main`, so a change merged to `main` reaches PyBADS's tests before any release. PyBADS's GPs can reach the low-noise representation (below), as the lower bound of their noise at its default options is a variance of about 1.4e-7. Code must stay Python 3.9 compatible (CI tests 3.9–3.11 on Linux, Windows, macOS).

## Commands

Install for development:

```console
python -m pip install -e ".[test]"   # pytest, pytest-rerunfailures, numdifftools
python -m pip install -e ".[dev]"    # the same, plus sphinx, numpydoc, build for docs/packaging
```

The runtime dependencies are NumPy, SciPy and matplotlib. What the tests need is the `test` extra, which CI installs and `dev` repeats. No module outside `gpyreg/testing/` imports pytest or numdifftools; `import gpyreg.testing` needs the extra, since `gpyreg/testing/__init__.py` imports a test module.

Tests live inside the package at `gpyreg/testing/` (not a top-level `tests/`). There is no pytest config file, so run from the repo root:

```console
python -m pytest                                   # whole suite (several hundred tests)
python -m pytest --reruns=5 -x -vv                 # exactly what CI runs
python -m pytest gpyreg/testing/test_mean_functions.py
python -m pytest gpyreg/testing/test_gaussian_process.py -k "test_fit and not isotropic"
```

Several tests are stochastic (slice-sampler moment checks, `fit` with MCMC samples) and can fail with small probability. That is why CI passes `--reruns=5`. A single failure of one of those is not by itself evidence of a bug; a deterministic or repeated failure is.

Formatting is black + isort at line length 79, plus pycln, via pre-commit:

```console
pre-commit run --all-files
```

Docs are Sphinx with numpydoc. Build locally with `make html` in `docsrc/` (output in `docsrc/_build/html`). `make github` additionally copies the result into a gitignored `docs/` at the repo root; the docs workflow does this on every push to `main` and commits it to the `gh-pages` branch. On Windows, `docsrc\make.bat` accepts the same targets, including `github`.

Packaging: `python -m build .`. The version comes from git tags through setuptools_scm, which writes the gitignored `gpyreg/_version.py`. Creating a GitHub release triggers the PyPI upload workflow. conda-forge's `gpyreg-feedstock` lists the run requirements by hand in `recipe/meta.yaml`, and its bot merges its own version-update PRs once their CI passes (`automerge` in the feedstock's `conda-forge.yml`), a CI that only imports gpyreg and runs `pip check`: a release that changes the runtime dependencies has the recipe's list changed in its update PR before that PR merges. Release notes go in `docsrc/source/release_notes.rst` under a `<version> (<YYYY-MM-DD>)` heading.

The PR test workflow only runs when files under `gpyreg/`, `pyproject.toml`, or `setup.py` change. Commit messages follow Conventional Commits (`feat:`, `fix:`, `perf:`, `docs:`, `chore:`); work happens on topic branches merged into `main` by PR.

## Architecture

### One hub class plus three pluggable component families

`gpyreg/gaussian_process.py` holds `GP`, the only class users interact with. Its constructor takes `(D, covariance, mean, noise)` where the three components are instances from `covariance_functions.py` / `isotropic_covariance_functions.py`, `mean_functions.py`, and `noise_functions.py`. The GP consumes them through a duck-typed protocol; only covariances have a real ABC (`AbstractKernel`):

- `hyperparameter_count(D)` and `hyperparameter_info(D)` returning a list of `(name, count)` tuples. Noise functions take no `D` argument in either method.
- `get_bounds_info(X, y)` returning a dict with keys `LB`, `UB`, `PLB`, `PUB`, `x0`, each a 1-D array of length equal to the component's hyperparameter count.
- `compute(...)`: covariance `compute(hyp, X, X_star=None, compute_diag=False, compute_grad=False)` returns `K` or `(K, dK)` with `dK` shaped `(N, N, cov_N)`; mean `compute(hyp, X, compute_grad=False)` returns `m` shaped `(N,)` or `(m, dm)`; noise `compute(hyp, X, y, s2, compute_grad=False)` returns a scalar when noise is homoskedastic and an `(N, 1)` array otherwise. GP handles both noise shapes explicitly.
- Mean functions may optionally define `compute_batched(hyp_stack, X)`; GP probes for it with `getattr` plus an MRO check, so a custom mean without it still works.

Isotropic kernels use cooperative multiple inheritance, e.g. `MaternIsotropic(AbstractIsotropicKernel, Matern)`: the isotropic mixin overrides the hyperparameter count (always 2), info, and bounds, and the constructor, which passes its arguments on to the ARD parent's; each isotropic kernel defines its own `compute` and inherits the rest, such as Matern's degree, from its ARD parent.

The ARD kernels (`SquaredExponential`, `Matern`, `RationalQuadraticARD`) take fixed `periods` at construction: constants, not hyperparameters, so the hyperparameter count, info and bounds do not depend on them. `kernel.periods` is `None` when no dimension is periodic, as it is for a kernel pickled without the attribute, which reads the class attribute `AbstractKernel.periods`. In each `compute` the periodic code is a branch of its own beside the non-periodic code, which does not go through the periodic helpers (`_scaled_sq_dist`, `_scaled_sq_diff`, and `_scaled_coordinates` and `_on_circle` beneath them), so that a change to the periodic branch moves no result of a kernel without periods. The helpers evaluate the trigonometric functions once per point and periodic dimension, not once per pair of points, and take the distances with `cdist`, so that a periodic kernel costs about what the kernel without periods costs. The isotropic kernels refuse periods (`AbstractIsotropicKernel.__init__`), and `GP.quad` refuses a periodic kernel.

The hyperparameter name strings returned by `hyperparameter_info` (`covariance_log_lengthscale`, `mean_const`, `noise_log_scale`, ...) are the dict keys used by `set_bounds`, `set_priors`, `hyperparameters_to_dict`, and friends. `set_bounds` and `set_priors` require every name to be present.

### Hyperparameter vector layout

Hyperparameters are a single flat float vector ordered **covariance, then noise, then mean**. Every slicing site in `gaussian_process.py` uses `hyp[0:cov_N]`, `hyp[cov_N:cov_N+noise_N]`, `hyp[cov_N+noise_N:]`. Scale-like quantities are stored in log space: lengthscales as `exp(hyp)`, output scale and noise scale as standard deviations so that variances are `exp(2*hyp)`. The rectified-linear noise threshold is on the raw `y` scale, not log scale. A new component must keep its `hyperparameter_info` order, `get_bounds_info` array order, and `compute` slicing mutually consistent.

### Lifecycle

- `GP.fit(X, y, s2, hyp0, options)`: fill unset bounds from the components' recommended bounds, evaluate the objective on a Sobol space-filling design of initial candidates built by `f_min_fill` (which maps unit-cube points through the hyperparameter priors), run `scipy.optimize.minimize` with analytic gradients from the best candidates, optionally run `SliceSampler` for `n_samples` posterior draws, then call `update` to build posteriors. Returns `(hyp, optimize_result, sampling_result)`.
- `Posterior` (bottom of `gaussian_process.py`) is a plain class holding `hyp, alpha, sW, L, sn2_mult, L_chol, sl, L_factor`; `GP.posteriors` is an object array with one entry per hyperparameter sample. `predict` averages over posteriors and adds between-sample variance unless `separate_samples=True`.
- `GP.update` appends data. Adding exactly one point with `y_new` and no replacement `hyp` extends each posterior: in the high-noise branch (below) by a rank-one update of the Cholesky factor with the noise scale stored in `Posterior.sl`, in the low-noise branch by extending both the inverse `L` and the factor `L_factor`, with the predictive variance of the new point and its solve taken from the factor. A posterior falls back to a full recomputation where the extension is numerically unsafe, or, in the low-noise branch, where it has no `L_factor`; anything else recomputes every posterior. When data are appended, the stored `s2` keeps one row per training input: points without a user-provided variance get zero.
- A `fit` or an `update` (and `set_hyperparameters`, which goes through `update`) that raises leaves the GP as it was before the call. Once it has checked its arguments, and before it changes anything, it takes `GP.__state()`: a shallow copy of the GP's `__dict__`, of the entries of its array of posteriors and of each posterior's `__dict__`, which `__restore` puts back on any exception. The code these calls run therefore replaces an attribute of the GP that it changes and never writes into the object that the attribute held before the call (the data, the bounds, the normalization constants, `hyper_priors`), since such a write survives the restore. The exceptions are the entries of the array of posteriors and the attributes of each posterior, which the single-point extension of `update` replaces and the state covers (it builds new arrays for them rather than writing into the old ones), and `hyper_priors["df"]`, which the fit fills for its duration and puts back itself.
- `GP.quad` (Bayesian quadrature) hard-codes the `SquaredExponential` hyperparameter layout and rejects other kernels and a periodic `SquaredExponential`.
- `GP.temporary_data` is a free-form dict used by PyVBMC to stash per-GP scratch; `GP.clean` empties it and drops cached quantities.

### The single numerical engine and its invariants

`GP.__core_computation` computes both the negative log marginal likelihood (with gradient) and `Posterior` objects; `log_likelihood` / `log_posterior` merely negate its output. It switches between two parametrizations on the smallest noise variance at the training inputs. From 1e-6 up, the high-noise branch (`L_chol=True`): `L` is the **upper** triangular Cholesky factor of `(K + sn2 I)/sl`, SciPy's default, so solves use `trans=1` then `trans=0`, and `L_factor` is `None`. Below 1e-6, the low-noise branch (`L_chol=False`): `L = -inv(K + sn2 I)`, as `gplite_core.m` has it, and `L_factor` is the upper triangular Cholesky factor of `K + sn2 I`, unscaled, from which `predict`, `predict_full`, `quad`, `random_function` and the single-point update form the covariance, since a covariance formed from the inverse carries its rounding, which grows as the noise shrinks. A posterior pickled by gpyreg 1.3.1 or earlier has no `L_factor`, and has it computed again where it is needed. In both branches `Posterior.sl` is `min(sn2) * sn2_mult`, the scale of `L` in the high-noise branch. Jitter is multiplicative, not additive: on factorization failure `sn2_mult` is multiplied by 10 and retried up to 10 times, and `sn2_mult` must be carried into every predictive noise term (`None` means 1). A GP created with `raise_on_cholesky_failure=True` makes one attempt instead: the first failure raises `LinAlgError`, and the space-filling design of `fit` ranks a starting point whose factorization fails last and starts no optimization from it unless every point failed.

Several performance shortcuts are asserted **bit-identical** by tests using `np.array_equal`, so changes near them must not reorder floating-point operations:

- Module-level `_REUSE_CHOLESKY` lets `fit` reuse the factorization across consecutive objective evaluations that differ only in mean hyperparameters.
- `_solve_triangular` and `_cholesky` call LAPACK directly and must match `scipy.linalg.solve_triangular` and `scipy.linalg.cholesky` exactly.
- `_prior_cache` (built by `__prior_masks`) must be invalidated by anything that changes priors or bounds.
- `_ZERO_COPY_CROSS_COVARIANCE_COMPUTES` whitelists the bundled kernels whose `compute` returns a fresh matrix, so `predict(..., return_cross_covariance=True)` can hand those matrices out without copying; treat returned cross-covariances as read-only. A user subclass or override falls back to a defensive copy. When the cross-covariance is not returned, `predict` scales a whitelisted kernel's matrix by `sW` in place, and never a matrix of another kernel, which may be the kernel's own buffer.
- The ARD kernels compute `K` and `dK` in place, in the fresh distance array and in arrays of their own, with the operations of the expressions of their definition in the same order (swapping the two operands of a product or a sum changes no bit); the squared differences of the length scales' gradients come from one broadcast (`_sq_diffs`), equal to `squareform(pdist(...))` of each dimension. `test_kernel_equals_its_direct_formulas` compares every output with those expressions. The powers keep the `**` operator, since NumPy takes an exponent of -1 as a reciprocal.
- The Gaussian priors' masses (`__recompute_normalization_constants`) and the design of `f_min_fill` take `scipy.special.ndtr` and `ndtri`, which equal `scipy.stats.norm`'s functions for a positive scale (`test_standard_normal_functions_match_scipy_stats`).
- The gradient of the objective sums each `Q * dK[:, :, i]` in the array that the first product allocates, whose layout NumPy chooses for the operands, with `np.add.reduce`, the reduction and the order of `np.sum` of the product. No test compares it with `np.sum(Q * dK[:, :, i])`, so a change to it is checked by comparing fits before and after it, bit for bit.

Shapes: `X` is `(N, D)`, `y` and `s2` are `(N, 1)`; `_convert_shapes` reshapes inputs and broadcasts a scalar `s2`. Prediction outputs averaged across samples are `(M, 1)`.

### Random number handling

Every public entry point that draws randomness (`GP.fit`, `GP.random_function`, `SliceSampler`, `f_min_fill`) takes a trailing `rng=None` keyword and calls `gpyreg.rng.resolve_rng(rng)` once at the top, then threads the resolved object down. `None` resolves to a stateless proxy over NumPy's global legacy stream so `np.random.seed` keeps working and the proxy never captures global state (samplers must stay picklable and copyable); a `numpy.random.Generator` is passed through unchanged so the caller shares the stream. Downstream code may only use the surface common to both (`random`, `uniform`, `standard_normal`, `shuffle`); for integers use `gpyreg.rng.random_integer(rng, high)` because the two backends name that method differently. `docsrc/source/rng.rst` documents this contract for users.

### Tests

Plain pytest module-level functions, heavily parametrized over seeds and dimensions. `gpyreg/testing/test_utils.py` provides `check_grad` (numerical gradients via numdifftools) used by the GP and kernel gradient tests; despite its `test_` prefix it is a helper module. Correctness is checked against analytic and numerical references, not against stored MATLAB outputs. Two seeding styles coexist: saving and restoring `np.random.get_state()` around `np.random.seed(...)` for the legacy path, and `np.random.default_rng(seed)` passed via `rng=` for the modern path; tests of the `rng=None` path must not leak global state. `gpyreg.testing` is not listed among the `packages` of `pyproject.toml`, but the wheels and the sdist ship it: with `include-package-data`, setuptools_scm's file finder makes every file under `gpyreg/` that git tracks package data of `gpyreg`, so a test module, or any file added under `gpyreg/testing/`, is installed with the package.

Coverage runs use the root `.coveragerc`, which omits the test modules, examples, and the generated version file. With pytest-cov installed: `python -m pytest --cov=gpyreg`.

### Documentation conventions

Docstrings are NumPy style (numpydoc renders them, and `undoc-members` is on so every public member appears). The per-module `.rst` pages under `docsrc/source/` list classes explicitly with `autoclass` directives; a new public class or function needs an entry there to be documented.

### Relation to the MATLAB reference

`matlab/gplite/` is the original implementation and is the place to check intended semantics: `gplite_train/post/pred/clean/rnd/plot/quad.m` and `private/gplite_core.m` map to `GP` methods, `gplite_covfun/meanfun/noisefun.m` to the three component modules, `private/slicesamplebnd.m` to `slice_sample.py`, and `private/fminfill.m` to `f_min_fill.py`. Isotropic kernels and the `periods` of the ARD kernels are Python-only additions; MATLAB BADS's own periodic kernel, `covPPERard_fast.m` (GPML, not gplite), maps a periodic input onto the unit circle and keeps its periods among the hyperparameters, so that its length scale is the one of `periods` times `2 * pi / p`. `gplite_sample.m` (samples of the density a GP represents), `gplite_fmin.m` (the optimum of a GP), `gplite_qpred.m` and the `outwarp_*.m` output-warping functions are not ported. `MANIFEST.in` prunes `matlab/` and `docsrc/` from distributions.
