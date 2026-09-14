Release notes
=============

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
