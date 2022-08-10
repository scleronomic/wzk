import os
from setuptools import Extension, setup


HOMEBREW_PREFIX = '/opt/homebrew/Cellar'
cgal_include_dir = os.environ.get('CGAL_INCLUDE_DIR',  # either user-defined
                                  HOMEBREW_PREFIX + '/cgal/5.5/include')  # or installed via homebrew

boost_include_dir = os.environ.get("BOOST_INCLUDE_DIR",  # either user-defined
                                   HOMEBREW_PREFIX + 'boost/1.79.0_1/include')  # or installed via homebrew

ext = Extension(
    name='wzkMinSphere',
    sources=['./MinSphere.cpp'],
    extra_compile_args=['-std=c++1y', '-ffast-math', '-Ofast', '-fpermissive'],
    # -lgmp
    include_dirs=[cgal_include_dir, boost_include_dir],
    library_dirs=[],
    libraries=[],
    language='c++',
)
#
setup(
    name='wzkMinSphere',
    version='0.1.0',
    ext_modules=[ext],
)

# # '/opt/homebrew/Cellar/cgal/5.3/include'
# # '/opt/homebrew/Cellar/boost/1.76.0/include'
# pip install -e .