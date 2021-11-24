from setuptools import setup, find_packages

setup(
    name='pysnirf2',
    description='Interface and validator for the SNIRF file format dynamically generated from the specification document.',
    author_email='sstucker@bu.edu',
    url='https://github.com/BUNPC/pysnirf2',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    include_package_data=True,
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    python_requires='>=3.0.*'
)
