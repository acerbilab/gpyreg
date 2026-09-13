Release notes
=============

1.2.1 (unreleased)
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
