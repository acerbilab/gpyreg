Release notes
=============

1.3.2 (unreleased)
------------------

A point marked **Upgrading** says what a script written for 1.3.1 may have
to change.

* :meth:`gpyreg.GP.get_recommended_bounds`, through which
  :meth:`gpyreg.GP.fit` fills the bounds that the caller leaves unset,
  refuses with ``ValueError`` training inputs without spread in a column,
  as a single training point has none in any column, and the message
  names the columns and the hyperparameters concerned. The recommended
  bounds of a length scale of the kernel, and of the scale of
  :class:`gpyreg.mean_functions.NegativeQuadratic`, take their scale from
  the width of the inputs, and a column of width zero gave them the pair
  ``(-inf, -inf)``, on which the optimizer of ``fit`` ended with
  ``KeyError``; with a lower bound of ``-inf`` from the caller instead,
  the fit returned an infinite length scale and predicted NaN. Such a
  hyperparameter is fitted between finite lower and upper bounds that the
  caller gives it. **Upgrading:** a script that catches the ``KeyError``
  catches ``ValueError``, and a script that fits on such inputs, or reads
  the recommended bounds for them, gives the hyperparameters that the
  message names finite bounds first.
* :meth:`gpyreg.GP.set_priors` leaves the GP as it was when it refuses its
  argument. On a GP without priors, a refused call marked the GP as
  having priors, so that ``str`` reported them as present and
  :meth:`gpyreg.GP.fit` added to its objective a log prior of zero.

1.3.1 (2026-09-23)
------------------

A point marked **Upgrading** says what a script written for 1.3.0 may have
to change.

* The gradient of the log prior is zero, not NaN, for a hyperparameter
  whose lower and upper bounds are equal and that has no prior, or a
  smooth-box prior whose box holds its value. The NaN reached
  :meth:`gpyreg.GP.log_posterior` with ``compute_grad=True``, and, where
  another hyperparameter has a prior, the optimizer of
  :meth:`gpyreg.GP.fit`, which stopped within an iteration, short of the
  optimum.
* The probability that a hyperprior puts inside the bounds of its
  hyperparameter, by which :meth:`gpyreg.GP.log_posterior` renormalizes
  the prior, is computed from the survival function where both bounds lie
  above the centre of the prior. As a difference of two values of the
  cumulative distribution function it was zero with both bounds far in
  the upper tail (beyond about 8.3 scales of a Gaussian prior), which made
  the log posterior infinite everywhere and sent :meth:`gpyreg.GP.fit` to
  a poor point. Where the lower bound is not above the centre, the value
  is the same as before, to the last bit. ``gpyreg.f_min_fill`` has the
  survival functions of the two smooth-box families, ``smoothbox_sf`` and
  ``smoothbox_student_t_sf``.
* The space-filling design of :meth:`gpyreg.GP.fit`, drawn by
  ``gpyreg.f_min_fill``, draws the starting values of a hyperparameter
  whose bounds both lie above the centre of its prior through the
  survival function of the prior and its inverse, where it drew them
  through the cumulative distribution function and the percent point
  function. With both bounds far in the upper tail, every draw of that
  hyperparameter lay at infinity (the starting points the fit was given
  stayed finite), and a fit with hyperparameter samples raised
  ``ValueError`` because the widths of the slice sampler were NaN. Where
  the lower bound is not above the centre, the design is the same as
  before, to the last bit. ``gpyreg.f_min_fill`` has the inverse survival
  functions of the two smooth-box families, ``smoothbox_isf`` and
  ``smoothbox_student_t_isf``.
* The space-filling design of :meth:`gpyreg.GP.fit`, drawn by
  ``gpyreg.f_min_fill``, gives a hyperparameter whose lower and upper
  bounds are equal their value at every point when the hyperparameter has
  a prior, as it did for one without. Mapped through the prior's quantile
  function, the value came back an ulp or two off, where the log prior is
  ``-inf``, so that the objective was infinite at every point of the
  design but the starting points the fit was given, and the ranking of
  the design that picks the starts of the optimization was lost. A GP
  that PyVBMC builds meets this when the noise is held at its lower bound
  (targets of a range below about 3e-3) and the option ``noise_size``
  moves the centre of the noise prior away from that bound.
* :meth:`gpyreg.GP.set_priors` refuses, with a message that says what is
  wrong, a coordinate whose ``sigma`` is finite beside a location that is
  not: an infinite or NaN ``mu`` of a Gaussian or Student's t prior, or an
  infinite or NaN end ``a`` or ``b`` of a smooth box, as it refuses a
  ``sigma`` that is not finite and positive. 1.3.0 took such a prior, and
  the log posterior was NaN with the bounds that :meth:`gpyreg.GP.fit`
  fills. A coordinate of a block without a prior is written, as before,
  with NaN for both its location and its ``sigma``. **Upgrading:** a
  script that wrote a non-finite ``mu`` for no prior, which gplite reads
  that way, sets the prior of a hyperparameter none of whose coordinates
  has a prior to ``None``, and gives a coordinate of a block without a
  prior NaN for both its location and its ``sigma``, as the message says.
  A smooth box with an infinite or NaN end has no such replacement: the
  script writes the prior it means.
* :meth:`gpyreg.GP.set_priors` refuses, with a message that says what is
  wrong, a smooth box of either family whose lower end ``a`` is above its
  upper end ``b``. 1.3.0 took such a box, whose normalization constant is
  then below one or negative, and gave a log prior that was wrong or NaN.
  A smooth box with ``a == b``, which has no plateau and is the Gaussian
  or the Student's t centred at ``a``, is taken as before. **Upgrading:**
  a script that wrote an inverted box writes the box it means, with
  ``a <= b``.

1.3.0 (2026-09-23)
------------------

A point marked **Upgrading** says what a script written for 1.2.1 may have
to change.

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
  are those of MATLAB's gplite. With one input dimension the two agree,
  and nothing changes. In more than one, a :meth:`gpyreg.GP.fit` with
  that mean function takes its plausible box from the recommendation,
  and its hard box wherever the caller leaves it unset, so its
  space-filling design, its default starting point and its slice sampler
  change, and its hyperparameters move; a fit that starts from given
  hyperparameters (``hyp0``, or those the GP already has), with no
  space-filling design (``init_N = 0``) and no hyperparameter samples
  (``n_samples = 0``), changes only where its start or its optimizer
  meets a hard bound that moved. ``fit`` does not start from the
  recommended length scales, which reach a caller that reads them from
  ``get_bounds_info``; the bounds of the length scales were per dimension
  already.
* The recommended bounds and starting length scale of the isotropic
  kernels take the means of the logarithms of the per-dimension widths
  and standard deviations, as MATLAB's gplite does, where they took the
  logarithm of the mean width and the standard deviation pooled over all
  entries of ``X``. With one input dimension the two agree; in more than
  one, a fit with
  :class:`gpyreg.isotropic_covariance_functions.SquaredExponentialIsotropic`
  or :class:`gpyreg.isotropic_covariance_functions.MaternIsotropic` takes
  a different plausible box for the length scale than in 1.2.1, and a
  different hard box wherever the caller leaves it unset.
* :meth:`gpyreg.GP.fit` takes its starting points as a copy of the
  space-filling design, so the default widths of the slice sampler are
  the standard deviation of the design as it was drawn; a fit that draws
  hyperparameter samples with a noise hyperparameter and
  ``1 < opts_N < init_N``, as at the defaults, sees its samples move.
* The plausible upper bound of the shape parameter of
  :class:`gpyreg.covariance_functions.RationalQuadraticARD` is its own
  (5) and the output scale keeps the range of the targets; the shape's
  line wrote into the output scale's slot. The plausible box of a fit
  with that kernel changes, and with it the space-filling design and the
  default starting point.
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
* A training set whose targets are all equal and finite is given a range
  of one, with a warning, instead of bounds of ``-inf`` that ended
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
  the predictive mean without a word. The rounding is measured against
  the prior variance at the test points, from which the predictive
  covariance is formed by subtraction, so a grid inside the training data
  draws as well. This holds in both representations of the posterior.
  Where the smallest noise variance at the training inputs is below 1e-6,
  as it can be after a default fit on noiseless targets, the posterior
  holds the inverse of the training covariance instead of its Cholesky
  factor (the low-noise representation). The predictive covariance that
  1.2.1 formed from that inverse carried a rounding error that grows as
  the noise shrinks, so that the draws of 1.2.1 from such a posterior
  often came out as the predictive mean, inside the training data and
  outside it;
  the draw now forms the covariance from a Cholesky factor of the
  training covariance. It raises ``LinAlgError`` where the covariance it
  computes has a negative eigenvalue beyond the rounding of the prior
  variance, and, in the low-noise representation, where the Cholesky
  decomposition of the training covariance fails even with the noise
  raised. The eigenvalue fallback also fixes the signs of whole
  eigenvectors and uses the symmetric eigensolver, so the factor it
  builds is a factor of the matrix it was given; a draw at closely spaced
  or duplicated test points came from the wrong covariance before. Draws
  through the eigenvalue fallback, and draws from a posterior in the
  low-noise representation, change for a given generator.
* :meth:`gpyreg.GP.predict_full` with ``add_noise=True`` adds the
  observation noise on the diagonal; with a noise that varies from point
  to point the returned matrix was neither symmetric nor a covariance
  matrix, while its diagonal was already right. A constant noise
  function is unaffected.
* A hyperprior whose degrees of freedom are infinite or NaN is the
  Gaussian family it names, MATLAB's ``HPRIOR.nu = Inf`` convention,
  where it used to contribute no prior at all; :meth:`gpyreg.GP.fit`
  fills NaN degrees of freedom with its option ``df_base`` for its own
  duration (below). A smooth-box prior set on a block of several
  hyperparameters has one normalization constant per coordinate of the
  block, where it doubled the log density with two coordinates on
  different sides of the box and raised with three.
  :meth:`gpyreg.GP.get_priors` returns, in the form
  :meth:`gpyreg.GP.set_priors` reads back, such a smooth-box prior, where
  it raised, and a prior with NaN degrees of freedom or a block with a
  coordinate that has no prior, where it returned ``None``; so
  ``set_priors(get_priors())`` keeps them.
* A single-observation :meth:`gpyreg.GP.update` of a posterior in the
  low-noise representation falls through to a full recomputation, with a
  warning, where rounding drives the latent variance of the new point to
  zero or below, as it can for an observation at an existing training
  input, and its predictive variance is the noise alone; the Cholesky
  representation already did so for its own stability test.
* :meth:`gpyreg.GP.fit` collapses an inverted plausible pair onto its
  upper bound, inside the hard box, before the space-filling design,
  which needs the pair ordered. For training targets whose standard
  deviation is below 1e-3 (the noise's plausible lower bound) and whose
  range is above 1e-6 (below that the hard pair collapses first), the
  noise's recommended plausible pair is inverted, and in 1.2.1 a fit with
  a space-filling design whose noise has no prior raised
  ``AssertionError``.
* The convergence diagnostics of :class:`gpyreg.slice_sample.SliceSampler`
  recognize a parameter whose recorded chain did not move by its range,
  whatever value it is frozen at: ``R`` and ``eff_N`` are NaN for it, and
  for a free parameter ``exit_flag`` is -3, with the "did not move"
  message. Before, this held only for a constant whose mean is exact, and
  a parameter fixed by ``LB == UB`` at, say, 0.3 reported finite ``R``
  and ``eff_N`` that were rounding noise.
* The normalization constant of a smooth-box Student's t prior, and the
  cumulative distribution and quantile functions of that distribution,
  are computed through the logarithms of the gamma functions, so degrees
  of freedom above about 340 no longer make the log posterior NaN.
* :meth:`gpyreg.GP.log_likelihood` and :meth:`gpyreg.GP.log_posterior`
  accept the dictionary of hyperparameters their docstrings document.
* :meth:`gpyreg.GP.fit` reads the sampler under ``sampler_name``, the
  name it documents, as well as under ``sampler``, and fills ``df_base``
  into a copy of the priors for its own duration, so a fitted GP keeps
  the priors the caller set and a second fit with another value uses it.
  A prior left with NaN degrees of freedom therefore reads as Gaussian
  outside the fit, in :meth:`gpyreg.GP.log_posterior` after it for one,
  where 1.2.1 wrote ``df_base`` into the GP's priors and read a Student's
  t from then on. **Upgrading:** a script that relies on that Student's t
  after the fit gives the prior its degrees of freedom.
* A single-observation :meth:`gpyreg.GP.update` on a GP that carries no
  posterior factors, after :meth:`gpyreg.GP.clean` or after an update
  with ``compute_posterior=False``, recomputes the posterior in full
  instead of raising ``TypeError``.
* The gradient of the marginal likelihood no longer raises where the
  noise function carries a scale for a user-provided variance that is
  never given.
* :meth:`gpyreg.GP.quad` reads a one-dimensional ``mu`` or ``sigma`` of
  length ``D`` as one measure in ``D`` dimensions, as MATLAB's gplite
  does, where 1.2.1 raised.
* Inputs that 1.2.1 took in silence are refused with a message: a
  coordinate that has a prior needs a finite, positive ``sigma`` (a
  hyperparameter without a prior is written as ``None``, and a coordinate
  of a block without one as NaN location and ``sigma``, as 1.2.1 accepted
  it); :meth:`gpyreg.GP.set_priors` and :meth:`gpyreg.GP.set_bounds`
  reject a hyperparameter name the model has not;
  :meth:`gpyreg.GP.get_recommended_bounds` takes any array_like and
  rejects a bound pair given inverted, and so does :meth:`gpyreg.GP.fit`,
  which fills its bounds through it; :meth:`gpyreg.GP.update` with
  ``hyp`` checks the width of the hyperparameter row, and with
  ``compute_posterior=True`` refuses hyperparameters that are NaN (never
  set), naming them; :meth:`gpyreg.GP.quad` refuses a mean function it
  cannot place, and a ``mu`` with more columns than input dimensions, for
  which 1.2.1 ignored the extra columns, broadcast them into a wrong
  integral or raised a broadcasting error, depending on the mean function
  and the dimension; and a noise
  variance given as an array whose row count is
  not that of the inputs is refused, where 1.2.1 reshaped any array of
  ``N`` entries into a column and failed on any other. Inputs on which
  1.2.1 failed with an error from inside the computation are refused
  with a message that names the problem: :meth:`gpyreg.GP.quad` refuses
  a GP without training data or posterior factors, where 1.2.1 raised
  ``AttributeError`` or ``TypeError``, and a ``mu`` with fewer columns
  than input dimensions, where it raised ``IndexError``; and
  :meth:`gpyreg.slice_sample.SliceSampler.sample` refuses a ``thin`` or
  ``burn`` that is not a whole number, an infinite one included, with
  ``ValueError`` instead of raising ``TypeError`` from ``range``. A noise
  variance may be any number or 0-d array; ``quad`` takes a ``sigma`` of
  one column, one standard deviation per measure in every dimension, as
  1.2.1 and gplite take it, and refuses a ``sigma`` of any other width
  than one or the number of input dimensions; and ``SliceSampler.sample``
  takes a ``thin`` or ``burn`` that is a whole number of any type, a
  float such as 2.0 included, where 1.2.1 took integers only. A failed
  Cholesky decomposition reports ``LinAlgError`` in both noise
  representations, where the low-noise one raised ``TypeError``, and the
  checks of the shape of the inputs raise ``ValueError`` where they raised
  ``AssertionError``. **Upgrading:** each refusal of an input that 1.2.1
  took in silence can stop a script that ran under 1.2.1, and so can the
  new type of an exception where a script catches the old one; such a
  script changes as follows:

  - a name the model has not, which set nothing, is left out;
  - a prior whose ``sigma`` is infinite, zero or negative, or NaN beside
    a location, is given a positive ``sigma``, or is replaced by ``None``
    where no prior is meant (1.2.1 took the absolute value of a negative
    ``sigma`` in the log prior);
  - a bound pair given inverted, which 1.2.1 collapsed onto its lower
    bound, is given in order (equal bounds fix a hyperparameter);
  - ``update(hyp=...)`` is given one column per hyperparameter of the
    GP, where 1.2.1 stored a row of another width and read it by offset;
  - ``gp.update(X_new=X, y_new=y)`` on a GP whose hyperparameters were
    never set, which 1.2.1 completed with NaN posterior factors, gives
    the data to ``gp.fit(X, y)`` instead. A following ``gp.fit()`` found
    hyperparameters in 1.2.1 only where it drew no hyperparameter samples
    (``n_samples=0``) and started from a space-filling design (``init_N``
    above zero, as at the default); for such a fit,
    ``compute_posterior=False`` in the update keeps the pattern working;
  - ``quad`` with another mean function, whose integral 1.2.1 computed
    as if that mean were a constant equal to its first hyperparameter,
    has no replacement;
  - ``quad`` with a ``mu`` of more columns than input dimensions, whose
    extra columns 1.2.1 ignored or broadcast into a wrong integral, is
    given one column per dimension;
  - a noise variance given as a row of ``N`` is given as a column;
  - a script that catches the exception of a check whose type changed
    catches the new one: ``ValueError`` for a shape check (an ``X`` that
    is not two-dimensional, or whose number of columns is not the GP's
    ``D``), which raised ``AssertionError``, for ``quad`` on a GP without
    training data or posterior factors (``AttributeError`` or
    ``TypeError``) or with a ``mu`` of fewer columns than input
    dimensions (``IndexError``),
    and for a ``thin`` or ``burn`` that is not a whole number
    (``TypeError``); ``LinAlgError`` for a failed Cholesky decomposition
    in the low-noise representation (``TypeError``).

* The covariance kernels refuse ``compute(compute_diag=True,
  compute_grad=True)`` with ``ValueError``. **Upgrading:** that
  combination returned the diagonal beside the gradient of the full
  matrix, a pair that meant nothing; ask for the two separately.
* The undocumented Metropolis step of
  :class:`gpyreg.slice_sample.SliceSampler` is removed, with its options
  and the attributes ``metropolis_pdf``, ``metropolis_rnd`` and
  ``metropolis_flag``. **Upgrading:** its options never turned the step
  on (their key was misspelled), but setting the three attributes
  directly did run it; a script that sets them now samples without the
  step, with no error.
* ``uuinv`` returns NaN for a ``p`` outside [0, 1] in every case, not only
  in the general one, and its documentation states the mixture it
  implements: the weight outside the plausible box is spread over the
  two tails in proportion to their lengths.
* Documentation: the isotropic kernels have a page;
  ``AbstractKernel.compute`` documents the ``(N, 1)`` shape of the
  diagonal it returns; :class:`gpyreg.noise_functions.GaussianNoise`
  documents the ``np.spacing(1)`` nugget it keeps with
  ``constant_add=False``; the default of ``add_noise`` of
  :meth:`gpyreg.GP.predict_full`, that its diagonal is not clamped and
  can be negative on a nearly singular posterior; which variance
  :meth:`gpyreg.GP.predict` returns by default, and how it pools the
  variances of several hyperparameter samples; that the log predictive
  density always carries the observation noise and over several
  hyperparameter samples is the density of the Gaussian with the pooled
  mean and variance; that :meth:`gpyreg.GP.log_posterior` renormalizes
  each prior over its bounds; what NaN degrees of freedom of a prior
  mean, in :meth:`gpyreg.GP.set_priors` and in the ``df_base`` option of
  :meth:`gpyreg.GP.fit`; and, in the docstring of the sampler's effective
  sample size, that its autocorrelations are paired from lag 0.

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
