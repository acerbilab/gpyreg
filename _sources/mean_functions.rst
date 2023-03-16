Mean functions
==============
-------------------------
``gpyreg.mean_functions``
-------------------------

Each mean function is implemented as its own class. An instance of one of these classes is passed to ``gpyreg.GP`` at initialization and defines the type of mean function for that Gaussian process.

``ConstantMean``
----------------
.. autoclass:: gpyreg.mean_functions.ConstantMean
    :members:
    :undoc-members:

``NegativeQuadratic``
---------------------
.. autoclass:: gpyreg.mean_functions.NegativeQuadratic
    :members:
    :undoc-members:

``ZeroMean``
------------
.. autoclass:: gpyreg.mean_functions.ZeroMean
    :members:
    :undoc-members:
