Noise functions
===============
--------------------------
``gpyreg.noise_functions``
--------------------------

Each noise function (so far, only :ref:`\`\`GaussianNoise\`\``) is implemented as its own class. An instance of one of these classes is passed to ``gpyreg.GP`` at initialization and defines the type of noise function for that Gaussian process.

``GaussianNoise``
-----------------
.. autoclass:: gpyreg.noise_functions.GaussianNoise
    :members:
    :undoc-members:
