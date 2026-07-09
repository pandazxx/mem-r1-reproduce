"""Download the LoCoMo benchmark data to data/locomo10.json."""

import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from memory_r1.locomo import LOCOMO_URL, load_locomo

DEST = Path(__file__).resolve().parents[1] / "data" / "locomo10.json"


def main() -> None:
    if DEST.exists():
        print(f"{DEST} already exists, skipping download")
    else:
        DEST.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {LOCOMO_URL}")
        urllib.request.urlretrieve(LOCOMO_URL, DEST)
    conversations = load_locomo(DEST)
    n_qa = sum(len(c.qa) for c in conversations)
    print(f"OK: {len(conversations)} conversations, {n_qa} QA pairs")


if __name__ == "__main__":
    main()
