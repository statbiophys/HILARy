from __future__ import annotations

from hilary import __version__
from setuptools import find_packages, setup

setup(
    install_requires=[
        "setuptools>=56,<57",
        "numpy>=1.20.0,<2",
        "openpyxl>=3.1,<4",
        "pandas>=2.1,<2.2",
        "scipy>=1.6,<2",
        "structlog>=22.3.0,<23",
        "textdistance>=4.6,<5",
        "tqdm>=4.66,<5",
        "typer>=0.9,<1",
        "atriegc>=0.0.3,<1.0.0",
        "scipy>1.11,<2",
    ],
    name="hilary",
    version="1.2.1",
    url="https://github.com/statbiophys/HILARy/",
    author="Gabriel Athènes,Natanael Spisak",
    author_email="gabriel.athenes@polytechnique.edu,natanael.spisak@gmail.com",
    packages=find_packages(),
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "infer-lineages=hilary.__main__:app",
        ],
    },
    classifiers=[
        "Intended Audience :: Developers",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
    ],
)
