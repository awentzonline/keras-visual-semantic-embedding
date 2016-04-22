#!/usr/bin/env python
from distutils.core import setup
from setuptools import find_packages

setup(
    name='keras-visual-semantic-embedding',
    version='0.0.1',
    description='Keras implementation of visual-semantic embedding.',
    author='Adam Wentz',
    author_email='adam@adamwentz.com',
    url='https://github.com/awentzonline/keras-visual-semantic-embedding/',
    packages=find_packages(),
    install_requires=[
        'h5py>=2.5.0',
        'Keras>=1.0.1',
        'numpy>=1.10.4',
        'six>=1.10.0',
    ],
)
