#! /usr/bin/env python

from setuptools import find_packages, setup

setup(
    name='skills',
    version='0.0.0',
    url='https://github.com/lobachevzky/skills',
    author='Ethan Brooks',
    author_email='ethanabrooks@gmail.com',
    packages=find_packages(),
    include_package_data=True,
    entry_points=dict(console_scripts=['skills=skills.main:cli']),
)
