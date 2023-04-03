from setuptools import setup, find_packages
# write a basic setup.py
setup(
    name='uncertainty_modelling',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'numpy',
        'scikit-learn',
        'statsmodels',
        'matplotlib',
        'seaborn'])
# run the following command in the terminal:
# pip install -e .