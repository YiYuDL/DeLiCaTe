from setuptools import setup

setup(
    name='DeLiCaTe',
    version='1.0.0',
    packages=['DeLiCaTe'],
    url='https://github.com/YiYuDL/DeLiCaTe',
    author='Yi Yu',
    author_email='yuyi689@gmail.com',
    install_requires=[
        'flake8==3.8.4',
        'mypy==0.790',
        'pytest==5.3.2',
        'pytorch-lightning==0.8.4',
        'scikit-learn==0.21.3',
        'scipy==1.7.3',
        'textbrewer==0.2.1'
        'transformers==3.5.1',
        'rdkit==2019.03.1.0',
        'torch==1.7.0',
    ], 
)
