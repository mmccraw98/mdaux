from setuptools import setup, find_packages

# pip install -e . # install in editable mode

setup(
    name='mdaux',
    version='0.1.1',
    license='MIT',
    description='A package for analyzing molecular dynamics simulations.',
    author='Marshall R. McCraw',
    packages=find_packages(),
    install_requires=[
        'cycler>=0.12.1',
        'joblib>=1.3.2',
        'matplotlib>=3.7.3',
        'numpy>=1.24.4',
        'pandas>=2.0.3',
        'tqdm>=4.66.1'
    ]
)