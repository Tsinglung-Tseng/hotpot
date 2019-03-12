from setuptools import setup, find_packages

setup(
    name="hotpot",
    version="0.0.1",
    description='Hotpot of my absolute awesome libs.',
    url='https://github.com/Tsinglung-Tseng/hotpot',
    author='Tsinglung Tseng',
    author_email='tsinglung.tseng@gmail.com',
    license='MIT',
    package_dir={'': 'src/python'},
    install_requires=[
        'pathlib',
        'typing',
        'tables',
        'matplotlib',
        'numpy'
    ],
    scripts=[],
    zip_safe=False
)