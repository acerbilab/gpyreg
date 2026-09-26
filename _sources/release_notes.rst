Release notes
=============

1.3.4 (unreleased)
------------------

* The warning with which a single-point :meth:`gpyreg.GP.update` falls
  back to a full recomputation of a posterior names the line that called
  ``update``, as it did up to 1.3.2; in 1.3.3 it named a line of gpyreg.
* :class:`gpyreg.GP` takes the keyword ``raise_on_cholesky_failure``,
  ``False`` by default. With ``True``, a Cholesky factorization of the
  training covariance that fails raises ``numpy.linalg.LinAlgError`` at
  its first attempt, in the posteriors and in the objective of
  :meth:`gpyreg.GP.fit` (:meth:`gpyreg.GP.log_likelihood` and
  :meth:`gpyreg.GP.log_posterior`), as in MATLAB BADS
  (``CholAttempts = 0``). By default such a factorization is tried again
  with the noise multiplied tenfold, up to ten times, and the posterior
  keeps the multiplier, which its predictions apply to the noise while
  :meth:`gpyreg.GP.get_hyperparameters` returns the noise as fitted. With
  the switch on, the space-filling design of ``fit`` ranks a starting
  point whose factorization fails last, as one of infinite value, and a
  failure in the optimization, in the sampling or at the fitted
  hyperparameters raises from ``fit``, which leaves the GP as it was. A
  copy or a pickle of the GP keeps the switch. Without it nothing changes,
  to the last bit.

1.3.3 (2026-09-24)
------------------

A point marked **Upgrading** says what a script written for 1.3.2 may have
to change.

* ``update()``, ``update(hyp=...)``, :meth:`gpyreg.GP.set_hyperparameters`,
  ``fit(X, y)`` and ``fit()`` run on a GP whose ``s2`` holds a number, as
  a caller may assign it, which the noise function adds at every input
  (an ``update`` given new data needs ``s2`` as an array, as in every
  release). The checks that a GP holds as many noise variances as inputs
  read the number of rows of ``s2`` and raised ``AttributeError`` or
  ``IndexError`` for such a value; a value without rows is not counted,
  and the calls run as they did in 1.3.1, to the last bit.
* A call of :meth:`gpyreg.GP.fit` or :meth:`gpyreg.GP.update`, and of
  :meth:`gpyreg.GP.set_hyperparameters` through ``update``, that raises
  leaves the GP as it was before the call, with the training data, bounds,
  priors and posteriors that it held; ``fit`` also refuses a column of
  inputs without spread, as :meth:`gpyreg.GP.get_recommended_bounds`
  does, before it changes anything. Both stored the data they were given
  first, and ``fit`` then the bounds filled from them, so that a call that
  raised after that left the GP holding the data of the call without
  posteriors that match them, from which :meth:`gpyreg.GP.predict`
  raised, or mixed the new inputs with the old posteriors: a fit that
  raised on such a column, on a starting point that holds NaN (which a
  fit without a space-filling design or an optimization keeps, and whose
  posterior ``update`` refuses), on a failed factorization of the
  training covariance, or in the sampler, after the optimization; and an
  update whose factorization of the training covariance failed, which
  left the posteriors replaced by empty ones or, where it failed in the
  full recomputation of a posterior to which a single-point update falls
  back, the other posteriors already extended. A fit that raised after
  filling its bounds also left those bounds, which a later fit that takes
  the bounds of the GP, as it does by default, kept. Every fit and update
  that completes is unchanged, to the last bit. **Upgrading:** a caller
  that reads a GP after a failed ``fit`` or ``update`` finds it as it was
  before the call, where 1.3.2 left it holding the new data (and, after a
  failed fit, the bounds the fit had set) with no usable posterior; a
  script that goes on with what the failed call stored gives it to the GP
  itself: the data through ``update`` or the next ``fit``, the bounds
  through :meth:`gpyreg.GP.set_bounds`.
* :meth:`gpyreg.GP.update`, and :meth:`gpyreg.GP.set_hyperparameters`
  through it, refuse hyperparameters that are NaN (not set), where the
  update would compute the posteriors from them, before it changes
  anything. The update stored the data it was given and replaced the
  posteriors with empty ones first, so that a GP given a NaN
  hyperparameter kept no posterior, and :meth:`gpyreg.GP.predict` then
  raised ``AttributeError``, and a GP without hyperparameters given data
  held them after the refusal. **Upgrading:** a script that gives data to
  a GP without hyperparameters through ``update`` and catches this
  refusal passes ``compute_posterior=False``, with which the update
  stores the data without computing a posterior.

1.3.2 (2026-09-24)
------------------

A point marked **Upgrading** says what a script written for 1.3.1 may have
to change.

* :meth:`gpyreg.GP.get_recommended_bounds`, through which
  :meth:`gpyreg.GP.fit` fills the bounds that the caller leaves unset,
  refuses with ``ValueError`` training inputs without spread in a column,
  as a single training point has none in any column, where a
  hyperparameter whose recommended bounds take their scale from that
  column has no finite lower bound from the caller, and the message names
  the columns and the hyperparameters concerned. The recommended bounds of
  a length scale of the kernel, and of the scale of
  :class:`gpyreg.mean_functions.NegativeQuadratic`, take their scale from
  the width of the inputs, and a column of width zero gave them the pair
  ``(-inf, -inf)``. With the lower bound of such a hyperparameter left
  unset or given as ``-inf``, the optimizer of ``fit`` ended with
  ``KeyError`` where the upper bound was unset or ``-inf`` as well, as it
  did with a lower bound of ``+inf``. Beside a finite upper bound or one
  of ``+inf``, the fit returned ``-inf`` for the logarithm of the scale,
  a length scale (or scale of the mean) of zero, from which its
  predictions were NaN, except on some fits to a single training point,
  which ended at the upper bound, or at NaN where that was ``+inf`` and
  then raised ``ValueError``; with hyperparameter samples, its slice
  sampler raised ``ValueError``. A finite lower bound from the caller is
  taken, and an upper bound left unset collapses onto it, as before: the
  fit runs as it did, to the last bit. **Upgrading:** a script that
  catches the ``KeyError`` catches ``ValueError``, and a script that fits
  on such inputs, or reads the recommended bounds for them, gives the
  hyperparameters that the message names a finite lower bound first.
* :meth:`gpyreg.GP.set_priors` leaves the GP as it was when it refuses its
  argument, and marks the GP as having priors only where a coordinate of
  some block has one. On a GP without priors, a refused call marked the
  GP as having priors, and so did a call whose blocks had no prior in any
  coordinate, such as ``("student_t", (nan, nan, nan))``, so that ``str``
  reported them as present and :meth:`gpyreg.GP.fit` added to its
  objective their log prior, which is zero except at a point off the value
  of a hyperparameter that equal bounds fix, where it is ``-inf``. A fit
  without a space-filling design (``init_N=0``) ranks its starting points,
  those it is given or those of the GP, before it moves them into the
  bounds: it ranked such a point last, and ranks it by its likelihood, and
  where that point comes first, the fit starts from it, moved onto the
  fixed value, and its result changes. **Upgrading:** a script
  that reads from ``str`` whether a GP set with such blocks alone has
  priors finds none, and one that compares the result of such a fit with
  values stored from 1.3.1 stores them again.
* :meth:`gpyreg.GP.get_priors` returns every prior that
  :meth:`gpyreg.GP.set_priors` takes in a form that ``set_priors`` writes
  back as it was, so ``set_priors(get_priors())`` changes nothing, the
  mark of the GP as having priors or none included. A Student's t block
  whose degrees of freedom mix zero with NaN or a number, such as
  ``[0, nan]`` or ``[0, 3]``, came back as ``None``, which dropped the
  prior; degrees of freedom infinite throughout, or zero on the
  coordinates that have a prior and NaN on the others, came back as the
  Gaussian family, which ``set_priors`` writes with zero; and a family set
  on a block with no prior in any coordinate came back as ``None``. Such a
  block, whose location and ``sigma`` are NaN throughout, comes back under
  the Gaussian or the Student's t family that its degrees of freedom name:
  as ``"gaussian"`` where it was set with ``"gaussian"`` or
  ``"smoothbox"``, or with ``"student_t"`` or ``"smoothbox_student_t"``
  and zero degrees of freedom throughout; as ``None`` where it was set
  with one of the latter two and NaN degrees of freedom throughout; and as
  ``"student_t"`` where it was set with one of them and other degrees of
  freedom. ``get_priors`` raises ``ValueError``, with ``set_priors``'
  message, for priors that ``set_priors`` refuses, as priors written into
  ``hyper_priors`` directly can be, and those of a GP pickled by gpyreg
  1.2.x or earlier, whose ``set_priors`` took any ``sigma``, where it
  returned a block that ``set_priors`` then refused (for a negative
  ``sigma``, among others), or ``None`` (for a NaN ``sigma`` beside a
  location). **Upgrading:** a script that compares what ``get_priors``
  returns finds a Student's t family with infinite degrees of freedom, or
  with degrees of freedom of zero where it has a prior and NaN elsewhere,
  under its own name, not as the Gaussian family, a Student's t block
  whose degrees of freedom mix zero with NaN or a number as
  ``"student_t"``, not ``None``, and a block without a prior in any
  coordinate, unless its degrees of freedom are NaN throughout, under the
  family just given, not ``None``; a script that writes priors into
  ``hyper_priors`` directly sets them through ``set_priors`` instead; and
  one that calls ``get_priors`` on a GP pickled by gpyreg 1.2.x or earlier
  whose priors ``set_priors`` refuses sets the priors of that GP again
  with ``set_priors`` first.
* :meth:`gpyreg.GP.set_bounds` refuses with ``ValueError`` a lower bound
  above the upper bound of the same hyperparameter, naming it, as
  :meth:`gpyreg.GP.get_recommended_bounds` and :meth:`gpyreg.GP.fit`
  already did, and leaves the bounds as they were. It stored the inverted
  pair, which the next ``fit`` refused where it took both its lower and
  its upper bounds from the GP, as it does by default, and ignored where
  it was given its own ``lower_bounds`` and ``upper_bounds``.
  **Upgrading:** a script that set an inverted pair, and never fitted or
  fitted with bounds of its own, gives the pair in order.
* :meth:`gpyreg.GP.fit` takes its option ``thin`` as a whole number of an
  integer or a float type, as :meth:`gpyreg.slice_sample.SliceSampler.sample`
  takes it, where a whole float such as 2.0 raised ``TypeError`` from the
  fit's own use of it. It refuses with ``ValueError``, before it changes
  anything, a ``thin`` that is not a whole number greater than zero: a
  fraction, which raised ``TypeError``, or the sampler's ``ValueError`` on
  the burn-in where the default ``burn``, ``thin`` times ``n_samples``,
  is a fraction; zero or a negative number, which raised another
  ``ValueError`` after the optimization; a bool, which ran as the integer
  it stands for; and, in a fit without hyperparameter samples, which does
  not use it, any such value, which it took. **Upgrading:** a script that
  passes ``thin=True`` passes ``1``; one that catches the ``TypeError`` of
  a fractional ``thin`` catches ``ValueError``; and one that gives a fit
  with ``n_samples=0`` a ``thin`` that is not a whole number greater than
  zero leaves it out.
* :meth:`gpyreg.GP.fit` takes its other counts, ``n_samples``, ``opts_N``
  and ``init_N``, as whole numbers of an integer or a float type, where a
  whole float such as 2.0 raised ``TypeError``, and refuses with
  ``ValueError``, before it changes anything, one that is not a whole
  number of at least zero: a fraction, which raised ``TypeError`` or
  another ``ValueError`` after the optimization; a negative number, with
  which ``opts_N`` and ``init_N`` ran as zero and ``n_samples`` raised
  after the optimization; a NaN ``init_N``, which ran as zero; and a bool,
  which ran as the integer it stands for, except ``init_N=True``, which
  raised ``TypeError``.
  :meth:`gpyreg.slice_sample.SliceSampler.sample` takes its number of
  samples ``N`` the same way, and refuses with ``ValueError`` one that is
  not a whole number greater than zero, where a whole float, a fraction or
  a bool raised ``TypeError`` and zero returned an empty chain; it refuses
  a bool as ``thin`` or ``burn`` as well, which ran as the integer it
  stands for. The counts of ``fit`` and of ``sample``, ``thin`` and
  ``burn`` included, take a 0-d array that holds a whole number as that
  number, and refuse one that holds anything else as they refuse what it
  holds. An array holding an integer ran as that integer, except as
  ``thin`` or ``burn`` of ``sample``, and so as ``burn`` of ``fit``, which
  ``sample`` refused; one holding a whole float raised ``TypeError``, or
  that refusal.
  **Upgrading:** a script that passes a negative ``opts_N`` or ``init_N``,
  or a NaN ``init_N``, passes ``0``; one that passes a bool as a count
  passes the integer; one that catches the ``TypeError`` of a count that
  is not a whole number, such as ``opts_N=1.5`` or ``init_N=32.5`` in
  ``fit`` or ``N=5.5`` or ``N=True`` in ``sample``, catches
  ``ValueError``; and one that asks ``sample`` for zero samples does not
  call it.
* :meth:`gpyreg.GP.fit` checks its option ``burn`` with its other counts,
  before it changes anything and whether or not it draws hyperparameter
  samples, by the rule of :meth:`gpyreg.slice_sample.SliceSampler.sample`:
  a whole number of at least zero, taken as the other counts are, or
  ``None``, which leaves the burn-in to the sampler. It refuses any other
  value with ``ValueError``. The fit passed ``burn`` to the sampler
  unchecked: with samples, the sampler refused such a value with
  ``ValueError`` after the optimization, raised ``TypeError`` for a
  string, and ran a bool as the integer it stands for; without samples,
  the fit took any value. **Upgrading:** a script that catches the
  ``TypeError`` of a string ``burn`` in a fit that draws samples catches
  ``ValueError``, and one that gives a fit with ``n_samples=0`` a
  ``burn`` that is neither ``None`` nor a whole number of at least zero
  leaves it out.
* Where the smallest noise variance at the training inputs is below 1e-6,
  as it can be after a fit on noiseless targets, the posterior holds the
  negative inverse of the training covariance (the low-noise
  representation). :meth:`gpyreg.GP.predict` and
  :meth:`gpyreg.GP.predict_full` form the predictive covariance there from
  a Cholesky factor of the training covariance, which the posterior keeps
  beside the inverse (the attribute ``L_factor`` of
  :class:`gpyreg.gaussian_process.Posterior`, one more ``N`` by ``N``
  matrix per hyperparameter sample). Formed from the inverse, as gplite
  forms it, the covariance carried a rounding error that grows as the
  noise shrinks: with 30 training inputs on [-2, 2], a squared
  exponential kernel of unit length and output scales and a noise
  standard deviation of 1e-6, the variances at the training inputs, of
  order 1e-12, were off by about 1e-4, many of them clamped to zero, and
  the covariance of ``predict_full`` had eigenvalues of order -1e-3. A
  single-point :meth:`gpyreg.GP.update` of such a posterior takes from
  the factor the predictive variance of the new point and the solve that
  it divides by that variance, and extends the factor. With the variance
  formed from the inverse it recomputed the posterior in full where
  rounding had clamped that variance, and otherwise extended the
  posterior with a wrong one, which moved the predictive mean away from
  that of the posterior computed in full: over the last ten of the 30
  inputs of the GP above, added one at a time, with targets of range 2
  and noise standard deviations from 1e-7 to 1e-4, by up to 1e-2, and by
  up to 0.3 with a length scale of 3, as the noise and the rounding of the
  BLAS build decided.
  :meth:`gpyreg.GP.random_function` draws from the kept factor instead of
  factoring the training covariance at each call: its draws from a
  posterior that no single-point update has extended are unchanged, and
  after such updates they change with the posterior. A posterior pickled
  by an earlier version has the factor computed again where it is
  needed. The Cholesky representation is unchanged, to the last bit.
  **Upgrading:** the predictions of a GP in the low-noise representation,
  and its posterior after single-point updates, change; a script that
  compares them with values stored from 1.3.1 stores them again. PyBADS
  reaches this representation: at its default options the lower bound of
  the noise of its GP is a variance of about 1.4e-7, below the 1e-6 at
  which the representation starts. It makes no single-point updates and
  reads only the hyperparameters of the posteriors, so for any fit whose
  noise variance ends below 1e-6, as on a noiseless target, its
  predictive means are unchanged and its predictive variances change,
  which can take its search elsewhere.
* :meth:`gpyreg.GP.quad` with ``compute_var=True`` forms the variance of
  an integral from the same Cholesky factor in the low-noise
  representation, where it formed it from the inverse and carried its
  rounding: with two training points 1e-3 apart, a squared exponential
  kernel of unit length and output scales, a noise standard deviation of
  1e-6 and a Gaussian measure of standard deviation 0.5 centred between
  the two points, a variance of 0.016 was off by 6e-11. The means of
  the integrals, and both in the Cholesky representation, are unchanged,
  to the last bit. **Upgrading:** the variances of integrals of a GP in
  the low-noise representation change.
* :meth:`gpyreg.GP.fit` on a GP without training data, given neither to
  it nor held from an earlier ``fit`` or ``update``, raises ``ValueError``
  that says so and names what is missing, before it changes anything. It
  raised ``AttributeError`` from a covariance function or from the check
  of the input shapes, or, given ``X`` alone, the ``ValueError`` of
  :meth:`gpyreg.GP.get_recommended_bounds` after storing ``X``.
  **Upgrading:** a script that catches the ``AttributeError`` catches
  ``ValueError``.
* :meth:`gpyreg.GP.update` and :meth:`gpyreg.GP.fit` refuse with
  ``ValueError``, before they change anything and with a message that
  gives the numbers, a call that would make the number of targets, or of
  noise variances, that the GP holds differ from its number of inputs.
  ``update`` refuses targets ``y_new`` or noise variances ``s2_new`` given
  without the inputs ``X_new`` to a GP that holds no inputs, or whose
  inputs have their targets, or their variances; ``X_new`` without
  ``y_new`` to a GP that holds targets; and ``X_new`` with ``y_new`` to a
  GP that holds inputs without targets. ``fit`` refuses ``X`` given
  without ``y``, or without ``s2``, where the targets, or the variances,
  that the GP holds are one per input it holds and not one per row of
  ``X``; it counts only the variances that the noise function reads (as
  :class:`gpyreg.noise_functions.GaussianNoise` with
  ``user_provided_add`` does), and keeps the others as before. On a GP
  without inputs, ``update`` raised ``AttributeError``. Elsewhere both
  stored the data, and the computation of the posterior, or the
  objective of the fit, then raised ``ValueError`` and left the GP
  without posteriors that match its data, so that the next
  :meth:`gpyreg.GP.predict` raised. They raised nothing, and left the
  numbers different, where no posterior was computed
  (``compute_posterior=False``, or a GP without targets), where only
  noise variances that the noise function does not take differed, as
  after ``s2_new`` given alone to data that hold their variances, and
  where NumPy broadcast a single value against the others: on a GP of a
  single training point, and where one input and its target were given
  to a GP that holds inputs without targets, whose posterior took that
  target at every input. Targets given alone, one per input, to a GP
  that holds inputs without targets, and noise variances given alone to
  one that holds data without variances, are taken as before: no call
  that ran to its end and left the GP with as many targets, and as many
  noise variances, as inputs, or none of them, is refused.
  **Upgrading:** a script that catches the ``AttributeError`` catches
  ``ValueError``, and one that relied on a call that left the numbers
  different (without a posterior, in variances that the noise function
  does not take, or through such a broadcast) gives the targets and the
  variances with their inputs, one per input.
* Documentation: the ``Raises`` section of :meth:`gpyreg.GP.fit` names
  the ``ValueError`` it passes on from
  :meth:`gpyreg.GP.get_recommended_bounds`, from the check of the shapes
  of its training data and from
  :class:`gpyreg.slice_sample.SliceSampler`, and the ``ValueError`` and
  ``LinAlgError`` it passes on from :meth:`gpyreg.GP.update`. Those of
  :meth:`gpyreg.GP.predict`, :meth:`gpyreg.GP.predict_full`,
  :meth:`gpyreg.GP.quad` and :meth:`gpyreg.GP.random_function` name the
  ``LinAlgError`` of the factorization that a posterior pickled by an
  earlier version in the low-noise representation needs. Those of
  ``update``, ``predict`` and ``predict_full`` name the ``ValueError`` of
  the check of the shapes of the data they are given. Those of ``fit``,
  ``update``, ``predict`` and ``predict_full`` name the ``TypeError`` of a
  noise variance that is neither an array, a number nor ``None``, and that
  of ``predict`` the ``ValueError`` of ``return_lpd=True`` without
  ``y_star``.

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
  already. PyVBMC 1.0.4 is such a caller: it starts the length scales of
  its GP from the recommendations for the points of highest density of its
  training set, and where those points share one value in a column, the
  start of that length scale is ``-inf``, or tens below zero where
  rounding leaves their standard deviation in it above zero, and from
  ``-inf`` the first GP fit of the run fails, where it completed with
  gpyreg 1.0.3 to 1.2.1; PyVBMC 1.5 takes the statistics of such a column
  from the whole training set.
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

  - a name the model has not, which set nothing, is left out; PyVBMC 1.0.4
    is such a caller: at its uncertainty level 1 (``uncertainty_handling``
    on, and a target that does not return its own noise estimate) it sets
    a prior on ``noise_provided_log_multiplier``, a hyperparameter its GP
    does not have, so that every such run stops at its first GP fit; it
    needs gpyreg 1.2.1 or earlier, or PyVBMC 1.5;
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
