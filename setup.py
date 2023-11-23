""" Setuptools module for building and packaging darwin. """


# Python Dependencies
from os import path
from setuptools import setup
from setuptools import find_packages


here = path.abspath(path.dirname(__file__))
PACKAGE_NAME = "Annealing"


def get_version(rel_path):
    """Function to read the __version__ from the init.
    This lets us define the version in a single location.

    Args:
        rel_path (str): Path from here defined above.

    Returns:
        Version string.

    """
    def read(x):
        with open(path.join(here, x), "r") as fp:
            return fp.read()

    for line in read(rel_path).splitlines():
        if line.startswith("__version__"):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")


# Get the long description from the README file
with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

# Get Requirements from text file
with open(path.join(here, "requirements.txt"), encoding="utf-8") as f:
    dependencies = f.read().splitlines()


setup(
    name=PACKAGE_NAME,  # Required
    version=get_version(f"{PACKAGE_NAME}/__init__.py"),
    description="Clean Extensible Interface for Simulated Annealing",
    long_description=long_description,
    long_description_content_type="text/markdown",  # Optional (see note above)
    url="https://github.com/Spill-Tea/Simulated-Annealing",
    author="Jason C Del Rio",
    author_email="spillthetea917@gmail.com",
    classifiers=[  # Optional
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "License :: OSI Approved :: MIT License",
    ],

    # Note that this is a string of words separated by whitespace, not a list.
    keywords="Annealing Optimization TSP",
    packages=find_packages(
        exclude=["tests", "dist", "build", "docs", "scripts", "templates", "notebooks", "__pycache__"]
    ),  # Required

    # https://packaging.python.org/guides/distributing-packages-using-setuptools/#python-requires
    python_requires=">=3.6, <4",
    install_requires=dependencies,
    extras_require={  # Optional
        "dev": ["pytest", "flake8", "pytest-cov", "wheel", "pdoc"],
        "test": ["pytest", "flake8", "pytest-cov", "wheel"],
    },

    project_urls={  # Optional
        "Bug Reports": "https://github.com/Spill-Tea/Simulated-Annealing/issues",
        "Source": "https://github.com/Spill-Tea/Simulated-Annealing",
    },
)
