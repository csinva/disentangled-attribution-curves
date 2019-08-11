from distutils.core import setup
import setuptools

with open("readme.md", "r") as fh:
    long_description = fh.read()

setup(
    name='dac',
    version='0.0.1',
    description="Disentangled attribution curves",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/csinva/sensible-local-interpretations",
    packages=setuptools.find_packages(),
    install_requires=[
        'matplotlib',
        'numpy',
        'scikit-learn',
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)