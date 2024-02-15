from setuptools import setup, find_packages

setup(
    name='connectiviz',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'matplotlib',
        'seaborn',
        'scipy',
        'holoviews'
    ],
)