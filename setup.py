from setuptools import setup, find_packages

setup(
    name="blenns",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'yfinance',
        'matplotlib',
        'scikit-learn',
        'tensorflow',
        'mplfinance',
        'Pillow'
    ],
)