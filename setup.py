from setuptools import setup, find_packages

setup(
    name="wzk",
    version="0.1.4",
    author="Johannes Tenhumberg",
    author_email="johannes.tenhumberg@gmail.com",
    description="WZK - WerkZeugKasten - Collection of different Python Tools",
    url="https://github.com/scleronomic/wzk",
    packages=find_packages(),
    install_requires=['numpy',
                      'scipy',
                      'scikit-image',

                      'matplotlib',
                      'pyvista',
                      'setuptools',
                      'msgpack',
                      # 'pyOpt @ git+https://github.com/madebr/pyOpt@master',
                      ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
