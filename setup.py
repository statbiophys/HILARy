from __future__ import annotations

from setuptools import find_packages
from setuptools import setup

from hilary import __version__

setup(
    install_requires=[
        "setuptools>=56,<57",
        "logging>=0.4.9,<1",
        "numpy>=1.20.0,<2",
        "openpyxl>=3.1,<4",
        "pandas>=2.1,<3",
        "scipy>=1.6,<2",
        "structlog>=22.3.0,<23",
        "textdistance>=4.6,<5",
        "tqdm>=4.66,<5",
        "typer>=0.9,<1",
        "atriegc>=0.0.3,<1.0.0",
    ],
    name="hilary",
    version=__version__,
    url="https://github.com/statbiophys/HILARy/",
    author="Natanael Spisak, Gabriel Athènes",
    author_email="natanael.spisak@gmail.com, gabriel.athenes@polytechnique.edu",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "infer=hilary.__main__:app",
        ],
    },
)