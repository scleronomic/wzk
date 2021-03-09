from setuptools import Extension, setup

ext = Extension(
    name='MinSphere',
    sources=['./MinSphere.cpp'],
    extra_compile_args=['-std=c++1y', '-ffast-math', '-Ofast', '-fpermissive'],  # TODO  '-std=c++1y' <-> '-std=c++14'
    # -lgmp
    include_dirs=[],
    library_dirs=[],
    libraries=[],
    language='c++',
)
#
setup(
    name='MinSphere',
    version='0.1.0',
    ext_modules=[ext],
)

# python setup.py develop

