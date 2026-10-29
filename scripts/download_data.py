"""Download the public datasets used in the experiments.

Everything is well under 5 GB. Idempotent: existing complete files are skipped.
On failure, prints the manual URL and the exact target path. The speech and DEMAND
archives are unpacked into data/speech/extracted/ and data/demand/extracted/, where
the experiments read them.

  uv run python scripts/download_data.py [ligo|speech|demand|ottawa|all]
"""
import os
import sys
import tarfile
import zipfile
import requests
from tqdm import tqdm

DATA = os.path.join(os.path.dirname(__file__), "..", "data")

MANUAL = []


def fetch(url, dest, desc=None, min_bytes=10_000):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.exists(dest) and os.path.getsize(dest) > min_bytes:
        print(f"[skip] {os.path.basename(dest)} exists "
              f"({os.path.getsize(dest)/1e6:.0f} MB)")
        return True
    try:
        with requests.get(url, stream=True, timeout=60,
                          headers={"User-Agent": "zerospec-research/0.1"}) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            tmp = dest + ".part"
            with open(tmp, "wb") as f, tqdm(total=total, unit="B", unit_scale=True,
                                            desc=desc or os.path.basename(dest)) as bar:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
                    bar.update(len(chunk))
        os.replace(tmp, dest)
        print(f"[ok]   {os.path.basename(dest)} ({os.path.getsize(dest)/1e6:.0f} MB)")
        return True
    except Exception as e:
        print(f"[FAIL] {url}\n       -> {e}")
        MANUAL.append((url, dest))
        return False


def unpack(archive, dest_dir, probe):
    """Unpack archive into dest_dir unless probe, a path inside the result, exists."""
    if os.path.exists(probe) or not os.path.exists(archive):
        return
    print(f"[unpack] {os.path.basename(archive)} -> {dest_dir}")
    os.makedirs(dest_dir, exist_ok=True)
    if archive.endswith(".zip"):
        with zipfile.ZipFile(archive) as z:
            z.extractall(dest_dir)
    else:
        with tarfile.open(archive) as t:
            t.extractall(dest_dir, filter="data")


def ligo():
    """GW150914: H1 and L1, 4096 s, 4 kHz HDF5 (~115 MB each)."""
    from gwosc.locate import get_urls
    gps = 1126259462
    for det in ("H1", "L1"):
        try:
            urls = [u for u in get_urls(det, gps, gps) if u.endswith(".hdf5")]
            url = next((u for u in urls if "4KHZ" in u.upper() and "4096" in u), urls[-1])
            fetch(url, os.path.join(DATA, "ligo", os.path.basename(url)))
        except Exception as e:
            print(f"[FAIL] gwosc locate {det}: {e}")
            MANUAL.append((f"https://gwosc.org/events/GW150914/ ({det} 4096s HDF5)",
                           os.path.join(DATA, "ligo", f"{det}_GW150914_4096s.hdf5")))


def speech():
    """LibriSpeech test-clean (346 MB)."""
    archive = os.path.join(DATA, "speech", "test-clean.tar.gz")
    fetch("https://www.openslr.org/resources/12/test-clean.tar.gz", archive)
    out = os.path.join(DATA, "speech", "extracted")
    unpack(archive, out, os.path.join(out, "LibriSpeech", "test-clean"))


def demand():
    """DEMAND noise: two contrasting environments at 16 kHz (~250 MB each)."""
    out = os.path.join(DATA, "demand", "extracted")
    for env in ("DKITCHEN", "PCAFETER"):
        archive = os.path.join(DATA, "demand", f"{env}_16k.zip")
        fetch(f"https://zenodo.org/records/1227121/files/{env}_16k.zip?download=1", archive)
        unpack(archive, out, os.path.join(out, env, "ch01.wav"))


def ottawa():
    """University of Ottawa variable-speed bearing dataset (Mendeley v43hmbwxpm)."""
    url = "https://data.mendeley.com/public-files/datasets/v43hmbwxpm/files"
    # Mendeley bulk download endpoint (version 2)
    bulk = "https://data.mendeley.com/api/datasets-v2/datasets/v43hmbwxpm/zip/download?version=2"
    ok = fetch(bulk, os.path.join(DATA, "ottawa", "ottawa_bearing_v2.zip"))
    if not ok:
        print("       Mendeley bulk endpoint may have moved; download manually:")
        print("       https://data.mendeley.com/datasets/v43hmbwxpm/2")
    print("       The bearing experiment reads data/ottawa/v43hmbwxpm-2/"
          "<numbered folders>/*.mat; unpack the archive there.")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "ligo"):
        ligo()
    if which in ("all", "speech"):
        speech()
    if which in ("all", "demand"):
        demand()
    if which in ("all", "ottawa"):
        ottawa()
    if MANUAL:
        print("\n=== MANUAL DOWNLOADS NEEDED ===")
        for url, dest in MANUAL:
            print(f"  URL : {url}\n  SAVE TO: {os.path.abspath(dest)}\n")
    else:
        print("\nAll requested downloads complete.")
