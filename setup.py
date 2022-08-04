#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Based on https://github.com/kennethreitz/setup.py
# To upload to PyPI
# >>> setup.py upload

import io
import os
import sys
from shutil import rmtree

from setuptools import find_packages, setup, Command

from snirf.__version__ import __version__ as VERSION

NAME = 'snirf'

about = {}
about['__version__'] = VERSION

here = os.path.abspath(os.path.dirname(__file__))

try:
    with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = ''


class UploadCommand(Command):
    """Support setup.py upload."""

    description = 'Build and publish the package.'
    user_options = []

    @staticmethod
    def status(s):
        """Prints things in bold."""
        print('\033[1m{0}\033[0m'.format(s))

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        try:
            self.status('Removing previous builds…')
            rmtree(os.path.join(here, 'dist'))
        except OSError:
            pass

        self.status('Building Source and Wheel (universal) distribution…')
        os.system('{0} setup.py sdist bdist_wheel --universal'.format(sys.executable))

        self.status('Uploading the package to PyPI via Twine…')
        os.system('twine upload dist/*')

        self.status('Pushing git tags…')
        os.system('git tag v{0}'.format(about['__version__']))
        os.system('git push --tags')

        sys.exit()


setup(
    name=NAME,
    version=about['__version__'],
    description='Interface and validator for SNIRF files',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author_email='sstucker@bu.edu',
    python_requires='>=3.6.0',
    install_requires=[
        'h5py>=3.1.0',
        'numpy',
        'setuptools',
        'pip',
        'termcolor',
        'colorama',
    ],
    url='https://github.com/BUNPC/pysnirf2',
    packages=find_packages(exclude=('tests', 'gen')),
    include_package_data=True,
    license='GPLv3',
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: Python :: Implementation :: PyPy'
    ],
    # $ setup.py publish support.
    cmdclass={
        'upload': UploadCommand,
    },
)
