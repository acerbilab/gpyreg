from setuptools import find_packages, setup

setup(
    name="gpyreg",
    version="0.1.0",
    description="Lightweight Gaussian process regression package",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "pytest",
        "sphinx",
        "numpydoc",
    ],
)
