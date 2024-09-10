from setuptools import Extension, setup

EIGEN_INCLUDE_DIR = "/Users/jote/mambaforge/envs/py3.11/include/eigen3"


ext = Extension(
    name="wzkopenGJK",
    sources=["./topy.cpp", "./openGJK.cpp"],
    extra_compile_args=["-std=c++1y", "-ffast-math", "-Ofast", "-fpermissive",
                        "-DEIGEN_STACK_ALLOCATION_LIMIT=524288"],
    include_dirs=[EIGEN_INCLUDE_DIR],
    library_dirs=[],
    libraries=[],
    language="c++",
)

setup(
    name="wzkopenGJK",
    version="0.1.0",
    ext_modules=[ext],
)

# Compile via:
# pip install -e .

