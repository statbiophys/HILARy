"""Execute hilary with command line."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import typer
from core import *

app = typer.Typer(add_completion=False)


@app.command()
def main():
    """Save all dataframes from all modules in one only excel file."""
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "simu_path",
        type=Path,
        help="",
    )
    parser.add_argument(
        "result_path",
        type=Path,
        help="",
    )
    parser.add_argument(
        "data_negative",
        type=Path,
        help="",
    )
    args = parser.parse_args()
    data_simu = pd.read_csv(args.simu_path)
    data_negative = pd.read_csv(args.data_negative)
    table_out = alignement_free_clone(data_simu, data_negative, W_l=150, per=0.1)
    table_out.to_csv(args.result_path)


if __name__ == "__main__":
    main()
