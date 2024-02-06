from __future__ import annotations

from hilary import __version__
from setuptools import find_packages, setup

setup(
    install_requires=[
        "setuptools>=56,<57",
        "numpy>=1.20.0,<2",
        "openpyxl>=3.1,<4",
        "pandas>=2.1,<3",
        "scipy>=1.6,<2",
        "structlog>=22.3.0,<23",
        "textdistance>=4.6,<5",
        "tqdm>=4.66,<5",
        "typer>=0.9,<1",
        "atriegc>=0.0.3,<1.0.0",
        "scipy>1.11,<2",
    ],
    name="hilary",
    version="1.1.2",
    url="https://github.com/statbiophys/HILARy/",
    author="Natanael Spisak, Gabriel Athènes",
    author_email="natanael.spisak@gmail.com, gabriel.athenes@polytechnique.edu",
    packages=find_packages(),
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "infer=hilary.__main__:app",
        ],
    },
    classifiers=[
        "Intended Audience :: Developers",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
    ],
)
