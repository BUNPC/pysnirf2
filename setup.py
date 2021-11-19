from setuptools import setup

setup(
    name='pysnirf2',
    # version='0.0.1',
    description='Interface and validator for the SNIRF file format dynamically generated from the specification document.',
    author_email='sstucker@bu.edu',
    url='https://github.com/BUNPC/pysnirf2',
    packages=['src'],
    install_requires=[
        'cached-property==1.5.2',
        'h5py==3.6.0',
        'numpy==1.19.5',
        'pip==21.3.1',
        'setuptools==40.8.0'
    ],
    include_package_data=True,
    classifiers=[
        'Programming Language :: Python :: 3.6',
    ],
)