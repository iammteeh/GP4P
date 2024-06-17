from setuptools import setup, find_packages
import setuptools
# write a basic setup.py
setup(
    name='GP4P',
    version='RC1',
    description='Gaussian Process for Performance Estimation',
    python_requires='>=3.9.6',
    author = "Immanuel Thoke",
    author_email = "immanuel.thoke@gmail.com",
    packages=find_packages(),
    install_requires=[
        "botorch==0.9.2",
        "copulae==0.7.9",
        "jax==0.4.14",
        "matplotlib==3.8.0",
        "networkx==3.1",
        "numpy==1.22.4",
        "numpyro==0.13.2",
        "pandas==1.5.3",
        "pyro_ppl==1.8.6+4be5c2e",
        "python_sat==0.1.8.dev10",
        "scipy==1.13.1",
        "seaborn==0.13.2",
        "setuptools==68.2.2",
        "statsmodels==0.14.0",
        "torch==2.0.1"]
)
# run the following command in the terminal:
# pip install -e .