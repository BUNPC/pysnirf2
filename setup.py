from setuptools import setup
import os
import sys

setup(
    name='pysnirf2',
    # version='0.0.1',
    description='Compute vibrational levels, wavefunctions, and expectation values using the Numerov-Cooley algorithm.',
    long_description=long_description,
    author='Radovan Bast',
    author_email='radovan.bast@uit.no',
    url='https://github.com/bast/numerov',
    license='MPL-2.0',
    packages=['numerov'],
    install_requires=[
        'cached-property==1.5.2',
        'h5py==3.6.0',
        'numpy==1.19.5',
        'pip==21.3.1',
        'setuptools==40.8.0'
    ],
    scripts=['bin/cooley'],
    include_package_data=True,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.6',
    ],
)
