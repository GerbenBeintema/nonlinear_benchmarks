from setuptools import setup, find_namespace_packages

with open('requirements.txt') as f:
    install_requires = [line for line in f]

packages = [a for a in find_namespace_packages(where='.') if a[:6]=='nonlinear_benchmarks']

setup(name = 'nonlinear_benchmarks',
      version = '0.0.3',
      description = 'The official dataload for http://www.nonlinearbenchmark.org/',
      author = 'Gerben Beintema',
      author_email = 'g.i.beintema@tue.nl',
      license = 'BSD 3-Clause License',
      python_requires = '>=3.6',
      packages=packages,
      install_requires = install_requires,
      extras_require = dict(
        docs = ['sphinx>=1.6','sphinx-rtd-theme>=0.5']
        )
    )
