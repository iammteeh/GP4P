from setuptools import setup, find_packages
import setuptools
# write a basic setup.py
setup(
    name='GP4P',
    version='1.1.0',
    description='Gaussian Process Regression for UQ Performance Estimation of Configurable Software Systems',
    python_requires='>=3.9.6',
    author = "Immanuel Thoke",
    author_email = "immanuel.thoke@gmail.com",
    packages=find_packages(),
    install_requires=[
        "botorch==0.9.2",
        "copulae==0.7.9",
        "jax==0.4.30",
        "matplotlib==3.9.2",
        "networkx==3.2.1",
        "numpy==2.1.1",
        "numpyro==0.15.2",
        "pandas==2.2.2",
        "python_sat==1.8.dev13",
        "scipy==1.14.1",
        "seaborn==0.13.2",
        "setuptools==72.1.0",
        "statsmodels==0.14.2",
        "streamlit==1.38.0",
        "torch==2.3.0"]
)
# run the following command in the terminal:
# pip install -e .