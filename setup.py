from setuptools import setup, find_packages

setup(
    name='blafisk',
    version='0.2.0',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'transformers>=4.57.0',
        # torch and torchvision installed separately - see README
        'chess>=1.11.0',
        'sentencepiece>=0.2.1',
        'protobuf>=6.33.0',
        'hf-xet>=1.1.0',
        'python-dotenv>=1.0.0',
    ],
)