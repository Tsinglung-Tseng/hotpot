from setuptools import setup, find_packages

setup(
    name="hotpot",
    version="0.1",
    namespace_packages=['hotpot'],
    packages=find_packages('src/python'),
    package_dir={'': 'src/python'},

    install_requires=[
        'jfs',
        'pathlib',
        'dxl-pygate',
        'typing'
    ],

    author="Tsinglung.Tseng",
    author_email="tsinglung.tseng@gmail.com",
    license="PSF"
)