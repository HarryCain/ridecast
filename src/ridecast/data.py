"""Download the original UCI archive, validate it, and record provenance."""
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import ssl
from urllib.request import urlopen
from zipfile import ZipFile

import certifi
import pandas as pd

SOURCE = "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip"
CITATION = "Fanaee-T, H. (2013). Bike Sharing [Dataset]. UCI. https://doi.org/10.24432/C5W894"


def download(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / "hour.csv"
    if destination.exists():
        return destination
    # Use a maintained CA bundle, including on Python.org macOS installations
    # whose optional system-certificate setup has not been run. Keep TLS verified.
    context = ssl.create_default_context(cafile=certifi.where())
    with urlopen(SOURCE, timeout=60, context=context) as response:
        archive = response.read()
    with ZipFile(BytesIO(archive)) as bundle:
        members = [name for name in bundle.namelist() if name.split("/")[-1] == "hour.csv"]
        if len(members) != 1:
            raise ValueError("UCI archive must contain exactly one hour.csv")
        content = bundle.read(members[0])
    # Validate before caching; never extract untrusted archive paths.
    validate(pd.read_csv(BytesIO(content)))
    destination.write_bytes(content)
    (directory / "provenance.json").write_text(json.dumps({
        "source": SOURCE, "citation": CITATION, "license": "CC BY 4.0",
        "archive_sha256": sha256(archive).hexdigest(),
        "csv_sha256": sha256(content).hexdigest(),
    }, indent=2) + "\n")
    return destination


def validate(raw: pd.DataFrame) -> None:
    required = {"dteday", "hr", "cnt", "casual", "registered", "holiday", "workingday",
                "temp", "hum", "windspeed", "weathersit"}
    if missing := required - set(raw.columns):
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if raw.empty or raw[list(required)].isna().any().any():
        raise ValueError("Source rows must be nonempty and complete")
    if not raw.hr.between(0, 23).all() or not (raw.hr % 1 == 0).all():
        raise ValueError("Hour must be an integer from 0 to 23")
    if not (raw.cnt >= 0).all() or not (raw.cnt == raw.casual + raw.registered).all():
        raise ValueError("Rental counts must be nonnegative and reconcile")
    stamp = pd.to_datetime(raw.dteday) + pd.to_timedelta(raw.hr, unit="h")
    if stamp.duplicated().any():
        raise ValueError("Duplicate local timestamps are ambiguous")
    for name in ["temp", "hum", "windspeed"]:
        if not raw[name].between(0, 1).all():
            raise ValueError(f"{name} must be normalized to [0, 1]")


def load(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    validate(raw)
    raw["timestamp"] = pd.to_datetime(raw.dteday) + pd.to_timedelta(raw.hr, unit="h")
    return raw.set_index("timestamp").sort_index()
