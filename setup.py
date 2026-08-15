# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib


from os.path import dirname, realpath
from setuptools import setup, find_packages, Distribution
from opencood.version import __version__


def _read_requirements_file():
    """Return the elements in requirements.txt."""
    req_file_path = '%s/requirements.txt' % dirname(realpath(__file__))
    with open(req_file_path) as f:
        return [line.strip() for line in f]


setup(
    name='V2V-inference',
    version=__version__,
    packages=find_packages(),
    url='https://github.com/Flo222/V2V_inference.git',
    license='MIT',
    author='Runsheng Xu, Hao Xiang',
    author_email='rxx3386@ucla.edu',
    description='Structured V2V cooperative perception and communication experimentation framework; '
                'cooperative detection',
    long_description=open("README.md").read(),
    install_requires=_read_requirements_file(),
)
