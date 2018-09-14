#!/usr/bin/env python
from setuptools import setup, find_packages

module_name = 'bayesian_uncertainty'


def requirements_from_pip():
    return list(filter(lambda l: not l.startswith('#'), open('pip.txt').readlines()))


setup(name=module_name,
      url="https://github.com/learningtitans/bayesian_uncertainty",
      author="Pedro Tabacof",
      package_dir={'': 'src'},
      packages=find_packages('src'),
      version="0.0.1 ",
      install_requires=requirements_from_pip(),
      include_package_data=True,
      zip_safe=False)
