"""Download the Zenodo "Classification of IT Support Tickets" data files used by triago.zenodo_bench.

Benitez Pereira (2022), doi:10.5281/zenodo.7648117, CC BY 4.0. Only the CSVs are fetched
(the record also ships code and pickle files, which are not needed and not loaded).

    python scripts/fetch_zenodo.py
"""
import hashlib
from pathlib import Path

import requests

URL = "https://zenodo.org/records/7648117/files/{}?download=1"
SHA256 = {
    "X_train.csv": "789712f97fb472544eeb82d8f5a142ed46e0bf254c6c34ac53a9f1575a9d460e",
    "y_train.csv": "cb851ff4d5e792886504b3040bcc85a5dbf94514d31293eddd6d69fcb7278d53",
    "X_test.csv": "4b808bc32c57f063960a3a4e7a4a30769c96be8143649aadc86835f4cb9763e5",
    "y_test.csv": "2b10db5e2f071e649efb3d40dd007ec62bc661f35b4afaf44117a00865760f94",
}
OUT = Path(__file__).resolve().parent.parent / "data" / "external" / "zenodo_it_tickets"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, digest in SHA256.items():
        dest = OUT / name
        if dest.exists() and hashlib.sha256(dest.read_bytes()).hexdigest() == digest:
            print(f"ok       {name}")
            continue
        r = requests.get(URL.format(name), timeout=60)
        r.raise_for_status()
        if hashlib.sha256(r.content).hexdigest() != digest:
            raise SystemExit(f"checksum mismatch for {name}")
        dest.write_bytes(r.content)
        print(f"fetched  {name}")


if __name__ == "__main__":
    main()
