from setuptools import setup, find_packages

setup(name='gpyreg',
      version='0.1.0',
      description='Lightweight Gaussian process regression package',
      packages=find_packages(),
      install_requires=['numpy',
                        'scipy',
                        'matplotlib',
                        'pytest',
                        'sphinx',
                        'numpydoc'],
     )
