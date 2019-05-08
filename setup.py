"""Author: Brandon Trabucco, Kavi Gupta, Copyright 2019"""


from setuptools import find_packages
from setuptools import setup


REQUIRED_PACKAGES = ['numpy', 'tensorflow-gpu', 'matplotlib']


setup(name='neural_lambda_calculus', version='0.1',
    install_requires=REQUIRED_PACKAGES,
    include_package_data=True,
    packages=[p for p in find_packages() if p.startswith('neural_lambda_calculus')],
    description='A Neural Lambda Calculus For Program Induction.')