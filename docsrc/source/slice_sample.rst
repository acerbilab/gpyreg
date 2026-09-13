==============
Slice sampling
==============
-----------------------
``gpyreg.slice_sample``
-----------------------

Known issue: stepping out
=========================

In versions 1.1.0 and 1.2.0, multidimensional sampling with
``options={"step_out": True}`` can evaluate bracket endpoints using stale
coordinates from an earlier coordinate update. This can change the
stepping-out decisions. The default ``step_out=False`` path and ordinary
:meth:`gpyreg.GP.fit` calls do not enable these evaluations.

Keep ``step_out=False`` until this issue is resolved; see
`issue #44 <https://github.com/acerbilab/gpyreg/issues/44>`_ for the report
and repair status. The report identifies a coordinate-update defect;
its effect on sampling accuracy has not been quantified.

``SliceSampler``
----------------
.. autoclass:: gpyreg.slice_sample.SliceSampler
    :members:
    :undoc-members:
