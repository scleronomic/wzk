from setuptools import setup, find_packages

with open("docs/README.md", "r") as fh:
    long_description = fh.read()


# import os
# import subprocess
# from setuptools.command.install import install
# directory = os.path.split(__file__)[0]
# class InstallLocalPackage(install):
#     def run(self):
#         install.run(self)
#         print(directory)
#         subprocess.call(f"cd wzk/cpp/MinSphere; python setup.py develop", shell=True)
# cmdclass={'install': InstallLocalPackage}

setup(
    name="wzk",
    version="0.0.1",
    author="Johannes Tenhumberg",
    author_email="johannes.tenhumberg@gmail.com",
    description="WerkZeugKasten - collection of python convenience functions for common modules",
    long_description=long_description,
    url="https://github.com/scleronomic/WerkZeugKasten",
    packages=find_packages(),
    install_requires=['numpy',
                      'scipy',
                      'scikit-image',

                      'matplotlib',
                      'PyQt5',
                      'pyvista',
                      'setuptools',
                      'msgpack',
                      'pyOpt @ git+https://github.com/madebr/pyOpt@gfortran-11-fixes',  # TODO wait for merge of fix
                      ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
