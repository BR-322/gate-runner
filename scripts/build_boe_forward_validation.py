#!/usr/bin/env python3
"""Fetch the source-standard BoE GBP/USD spot and forward validation panel."""

from __future__ import annotations

import argparse
import gzip
import io
import urllib.parse
import urllib.request
from pathlib import Path


URL = "https://www.bankofengland.co.uk/boeapps/database/_iadb-fromshowcolumns.asp"
PARAMETERS = {
    "csv.x": "yes",
    "Datefrom": "01/Jan/2009",
    "Dateto": "31/Dec/2024",
    "SeriesCodes": "XUDLUSS,XUDLDS1,XUDLDS3,XUDLDS6,XUDLDSY",
    "CSVF": "TN",
    "UsingCodes": "Y",
    "VPD": "Y",
    "VFD": "N",
}
EXPECTED_HEADER = b"DATE,XUDLUSS,XUDLDS1,XUDLDS3,XUDLDS6,XUDLDSY"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    request = urllib.request.Request(
        f"{URL}?{urllib.parse.urlencode(PARAMETERS)}",
        headers={"User-Agent": "Gate-Runner-public-data-builder/0.3"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        raw = response.read()
    if not raw.startswith(EXPECTED_HEADER):
        raise ValueError("BoE download has an unexpected header")

    output = io.BytesIO()
    with gzip.GzipFile(fileobj=output, mode="wb", filename="", mtime=0) as handle:
        handle.write(raw)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(output.getvalue())
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
