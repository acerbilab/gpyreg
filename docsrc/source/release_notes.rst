Release notes
=============

1.3.0 (unreleased)
------------------

The first eight points move numbers that a 1.2.1 user can observe at the
defaults of what they name, and the five after them do so in the cases
they repair; the rest repair calls that raised, refuse inputs that were
taken in silence, remove one thing and document.

* The recommended bounds of the hyperparameters take the statistics of
  the training inputs per input dimension. The starting value of each
  length scale is the logarithm of the standard deviation of its own
  dimension, and the location and the scale of
  :class:`gpyreg.mean_functions.NegativeQuadratic` have, per dimension, a
  hard box, a plausible box and a starting value built from that
  dimension's minimum, maximum, median, width and standard deviation.
  They were taken over all the entries of ``X`` at once, one number for
  every dimension, which mixes the locations of the dimensions with their
  spreads: where the dimensions lie apart, every recommendation came out
  as wide as the widest gap between them. The per-dimension statistics
  are those of MATLAB's gplite. A :meth:`gpyreg.GP.fit` that starts from
  the recommendation, or leaves the bounds of the mean to it, sees its
  hyperparameter samples move; the widths beside the length scales were
  per dimension already.
* The recommended bounds and starting length scale of the isotropic
  kernels take the means of the logarithms of the per-dimension widths
  and standard deviations, as MATLAB's gplite does, where they took the
  logarithm of the mean width and the standard deviation pooled over all
  entries of ``X``; a fit with
  :class:`gpyreg.isotropic_covariance_functions.SquaredExponentialIsotropic`
  or :class:`gpyreg.isotropic_covariance_functions.MaternIsotropic` starts
  from a different design and feasible set than in 1.2.1.
* :meth:`gpyreg.GP.fit` takes its starting points as a copy of the
  space-filling design, so the default widths of the slice sampler are
  the standard deviation of the design as it was drawn; a script that
  calls ``fit`` at the default ``opts_N = 3`` sees its sampled
  hyperparameters move.
* The plausible upper bound of the shape parameter of
  :class:`gpyreg.covariance_functions.RationalQuadraticARD` is its own
  (5) and the output scale keeps the range of the targets; the shape's
  line wrote into the output scale's slot. The space-filling design of a
  fit with that kernel changes.
* Slice sampling with a burn-in of two or three iterations no longer
  returns the same point once per requested sample: a coordinate whose
  burn-in variance estimate is not positive keeps the width it has
  instead of being floored to zero, and a window of fewer than two
  iterations adapts nothing.
* An infinite entry of ``SliceSampler(widths=...)``, which the
  documentation of ``widths`` allows for an unbounded coordinate, no
  longer comes back after the burn-in and fills the chain with NaN.
* :class:`gpyreg.covariance_functions.Matern` and
  :class:`gpyreg.isotropic_covariance_functions.MaternIsotropic` of degree
  1 can be fitted: the gradient with respect to the length scale is zero,
  not NaN, where two inputs coincide, so the gradient of the marginal
  likelihood is finite.
* A training set whose targets are all equal is given a range of one,
  with a warning, instead of bounds of ``-inf`` that ended
  :meth:`gpyreg.GP.fit` with ``KeyError: (-inf, -inf)``; such a fit now
  completes.
* :meth:`gpyreg.GP.quad` with ``compute_var=True`` normalizes by the
  noise scale the stored Cholesky factor carries, so the variance of an
  integral is right after a rank-one :meth:`gpyreg.GP.update` that
  appended a point of lower total noise, where it used to be clamped to
  machine epsilon.
* :meth:`gpyreg.GP.random_function` draws a sample where the predictive
  covariance is numerically singular, as it is on a dense
  one-dimensional grid: eigenvalues that are negative but of rounding
  size count as the zeros they are, where the draw used to collapse onto
  the predictive mean without a word, and a negative eigenvalue beyond
  the rounding band raises ``LinAlgError``. The eigenvalue fallback also
  fixes the signs of whole eigenvectors and uses the symmetric
  eigensolver, so the factor it builds is a factor of the matrix it was
  given; a draw at closely spaced or duplicated test points came from
  the wrong covariance before. Draws through this path change for a
  given generator.
* :meth:`gpyreg.GP.predict_full` with ``add_noise=True`` adds the
  observation noise on the diagonal; with a noise that varies from point
  to point the returned matrix was neither symmetric nor a covariance
  matrix, while its diagonal was already right. A constant noise
  function is unaffected.
* A hyperprior whose degrees of freedom are infinite or NaN is the
  Gaussian family it names, MATLAB's ``HPRIOR.nu = Inf`` convention,
  where it used to contribute no prior at all; and a smooth-box prior set
  on a block of several hyperparameters has one normalization constant
  per coordinate of the block, where it doubled the log density with two
  coordinates on different sides of the box and raised with three.
  :meth:`gpyreg.GP.get_priors` returns such a prior instead of raising.
* A single-observation :meth:`gpyreg.GP.update` of a posterior in the
  low-noise representation falls through to a full recomputation, with a
  warning, where the predictive variance of the new point is at or below
  what the variance clamp can produce, an observation at an existing
  training input for one, as the Cholesky representation already did for
  its own stability test.
* :meth:`gpyreg.GP.fit` no longer leaves the plausible bounds inverted
  where the plausible box lies outside the hard box, which made the
  space-filling design fail an internal assertion for training targets
  whose standard deviation falls below the lower bound of the noise.
* The convergence diagnostics of :class:`gpyreg.slice_sample.SliceSampler`
  report a parameter whose recorded chain did not move as undefined
  (``exit_flag`` -3, ``R`` and ``eff_N`` NaN, and the "did not move"
  message) whatever value it is frozen at; before, this held only for a
  constant whose mean is exact.
* The normalization constant of a smooth-box Student's t prior, and the
  cumulative distribution and quantile functions of that distribution,
  are computed through the logarithms of the gamma functions, so degrees
  of freedom above about 340 no longer make the log posterior, or the
  space-filling design, NaN.
* :meth:`gpyreg.GP.log_likelihood` and :meth:`gpyreg.GP.log_posterior`
  accept the dictionary of hyperparameters their docstrings document.
* :meth:`gpyreg.GP.fit` reads the sampler under ``sampler_name``, the
  name it documents, as well as under ``sampler``, and fills ``df_base``
  into a copy of the priors, so a fitted GP keeps the priors the caller
  set and a second fit with another value uses it.
* A single-observation :meth:`gpyreg.GP.update` on a GP that carries no
  posterior factors, after :meth:`gpyreg.GP.clean` or after an update
  with ``compute_posterior=False``, recomputes the posterior in full
  instead of raising ``TypeError``.
* The gradient of the marginal likelihood no longer raises where the
  noise function carries a scale for a user-provided variance that is
  never given.
* Inputs that were taken in silence are refused with a message: a prior
  needs a finite, positive ``sigma`` (no prior is written as ``None``, not
  as an infinite scale); :meth:`gpyreg.GP.set_priors` and
  :meth:`gpyreg.GP.set_bounds` reject a hyperparameter name the model
  has not; :meth:`gpyreg.GP.get_recommended_bounds` takes any array_like
  and rejects a bound pair given inverted; :meth:`gpyreg.GP.update` with
  ``hyp`` checks the width of the hyperparameter row, and with
  ``compute_posterior=True`` refuses hyperparameters that were never
  set, naming them; :meth:`gpyreg.GP.quad` refuses a GP without training
  data or posterior factors, a mean function it cannot place, and a
  measure that has not one column per input dimension; a noise variance
  may be any number or 0-d array, while an array whose row count is not
  that of the inputs is refused; and
  :meth:`gpyreg.slice_sample.SliceSampler.sample` refuses a ``thin`` or
  ``burn`` that is not a whole number with ``ValueError`` instead of
  raising ``TypeError`` from ``range``. A failed Cholesky decomposition
  reports ``LinAlgError`` in both noise representations, where the
  low-noise one raised ``TypeError``. **Upgrading:** a script that passes
  a fractional ``thin`` or ``burn`` must round it; a whole number of any
  type is still accepted.
* The covariance kernels refuse ``compute(compute_diag=True,
  compute_grad=True)`` with ``ValueError``. **Upgrading:** that
  combination returned the diagonal beside the gradient of the full
  matrix, a pair that meant nothing; ask for the two separately.
* The undocumented Metropolis step of
  :class:`gpyreg.slice_sample.SliceSampler` is removed. **Upgrading:** it
  never ran (its option key was misspelled), and the attributes
  ``metropolis_pdf``, ``metropolis_rnd`` and ``metropolis_flag`` no longer
  exist, so a script that set them must drop them.
* ``uuinv`` returns NaN for a ``p`` outside [0, 1] in every case, not only
  in the general one, and its documentation states the mixture it
  implements: the weight outside the plausible box is spread over the
  two tails in proportion to their lengths.
* Documentation: the isotropic kernels have a page;
  ``AbstractKernel.compute`` documents the ``(N, 1)`` shape of the
  diagonal it returns; the default of ``add_noise`` of
  :meth:`gpyreg.GP.predict_full`, that its diagonal is not clamped and
  can be negative on a nearly singular posterior; which variance
  :meth:`gpyreg.GP.predict` returns by default; that the log predictive
  density always carries the observation noise and over several
  hyperparameter samples is the density of the moment-matched Gaussian;
  and that :meth:`gpyreg.GP.log_posterior` renormalizes each prior over
  its bounds.

1.2.1 (2026-09-14)
------------------

* :meth:`gpyreg.GP.quad` returns the correct integral variance when the
  training noise is heteroskedastic, for example with user-provided
  observation variances. The variance previously collapsed to machine
  epsilon in that case. ``quad`` also handles
  ``SquaredExponentialIsotropic`` in more than one dimension, where it
  previously read the hyperparameters in the wrong order. Results with
  homoskedastic noise, and with the isotropic kernel in one dimension, are
  unchanged.
* :meth:`gpyreg.GP.update` applies replacement hyperparameters passed
  together with a single new observation. The rank-one shortcut is taken
  only when the hyperparameters stay in place; it previously discarded the
  new values. Calls without ``hyp`` are unchanged.
* The rank-one update of a single appended observation extends the
  Cholesky factor with the noise scale the factor was built with, stored
  as the :class:`gpyreg.gaussian_process.Posterior` attribute ``sl``. It
  is therefore exact for heteroskedastic noise, where it previously
  assumed the new point's noise equal to the smallest training noise, and
  it works with output-dependent noise, which previously raised an error.
  The shortcut now also applies when ``s2_new`` is supplied, which
  previously triggered a full recomputation; the result is the same up to
  floating-point rounding. The homoskedastic case is unchanged.
* :meth:`gpyreg.GP.update` keeps the stored user-provided noise aligned
  with the training inputs. Points added without ``s2_new`` to a GP that
  stores ``s2``, or earlier points of a GP that receives ``s2_new`` for
  the first time, get a variance of zero. Previously ``s2`` could end up
  with fewer rows than ``X``, which made the next posterior computation
  fail, and in the second case a single new variance was silently applied
  to every training point.
* :class:`gpyreg.slice_sample.SliceSampler` accepts bounds and widths
  given as lists. A coordinate with equal list bounds is now recognized as
  fixed; previously it went undetected and list widths raised an error.
  The sampler also logs its no-violation message when the diagnostics
  report success, which previously never happened.
* :class:`gpyreg.slice_sample.SliceSampler` with ``step_out=True`` in more
  than one dimension evaluates the stepping-out brackets on the current
  coordinate line. Previously the bracket ends of a later coordinate
  carried an earlier coordinate's shrunk bracket edge. Sampling with the
  default ``step_out=False`` is unchanged.
* The convergence diagnostics of :class:`gpyreg.slice_sample.SliceSampler`
  report a positive, bounded effective sample size, following Geyer's
  initial positive sequence with the floor used by Stan and ArviZ; it
  could previously be negative. A coordinate fixed by ``LB == UB`` is
  excluded from the checks, and undefined diagnostics or fewer than eight
  recorded samples give ``exit_flag`` -3 instead of reporting success.
  ``R`` and the effective sample size are unchanged for chains that mix.
* The documentation gains a section on BLAS threading, and the attributes
  of :class:`gpyreg.gaussian_process.Posterior` are documented.

1.2.0 (2026-09-13)
------------------

* :meth:`gpyreg.GP.predict` accepts the keyword-only option
  ``return_cross_covariance=True``. It appends a tuple containing the latent
  training-to-prediction kernel matrix for each hyperparameter sample.
  Downstream calculations can reuse these matrices instead of computing
  them again. Ordinary calls retain their existing return values.
* Kernel matrices remain separate when predictions are averaged across
  hyperparameter samples. They exclude observation noise and are
  unconditioned kernels, rather than posterior predictive covariances.
  Prior-only GPs return ``None`` for each sample. See
  :doc:`gaussian_process` for examples, shapes and memory considerations.
* The prediction reference corrects the documented default of ``add_noise``
  to ``False`` and the averaged-output shape to ``(M, 1)``. These are
  documentation corrections; the corresponding behavior is unchanged.

1.1.0 (2026-09-05)
------------------

* :meth:`gpyreg.GP.fit`, :meth:`gpyreg.GP.random_function`,
  :class:`gpyreg.slice_sample.SliceSampler` and ``f_min_fill`` accept
  ``rng=``. Pass a NumPy generator to share a stream with the caller, or a
  seed to create a generator. The default ``rng=None`` continues to use
  NumPy's global stream controlled by ``np.random.seed``. See :doc:`rng`.
* Prediction and hyperparameter fitting have lower computational overhead,
  including batched evaluation of bundled mean functions and reuse of a
  factorization when only mean hyperparameters change during fitting.
  Custom mean and covariance implementations retain their supported paths.
* :meth:`gpyreg.GP.log_likelihood` and :meth:`gpyreg.GP.log_posterior`
  correctly return ``(value, gradient)`` when ``compute_grad=True``;
  these calls previously raised ``TypeError``.
* Slice samplers using the default stream remain copyable and picklable.
  Their saved state does not capture NumPy's global random state. Samplers
  with an explicit generator preserve its state when copied or pickled;
  older sampler saves resume using the global stream.
