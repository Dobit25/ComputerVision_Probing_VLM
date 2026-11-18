"""Download immutable assets, verify bytes, and retain a provenance record."""

import hashlib
import subprocess
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .io_utils import write_json


def download_ranges(url, path, workers=4, chunk_size=8 * 1024 * 1024):
    """Resumable byte ranges, validated before joining; final checksum remains mandatory."""
    def request(start, end):
        separator = "&" if "?" in url else "?"
        req = urllib.request.Request(f"{url}{separator}vlm_probe_part={start}",
                                     headers={"Range": f"bytes={start}-{end}"})
        return urllib.request.urlopen(req, timeout=60)

    with request(0, 0) as response:
        match = re.fullmatch(r"bytes 0-0/(\d+)", response.headers.get("Content-Range", ""))
        if response.status != 206 or not match:
            raise RuntimeError("Server does not support verifiable byte ranges.")
        total = int(match.group(1))
        response.read(1)
    parts = path.with_name(path.name + ".parts")
    parts.mkdir(parents=True, exist_ok=True)

    def fetch(start):
        end = min(start + chunk_size, total) - 1
        part = parts / f"{start:012d}"
        if part.exists() and part.stat().st_size == end - start + 1:
            return part
        for attempt in range(4):
            try:
                with request(start, end) as response, part.open("wb") as stream:
                    if response.status != 206 or response.headers.get("Content-Range") != f"bytes {start}-{end}/{total}":
                        raise RuntimeError("Download range response mismatch.")
                    while block := response.read(1024 * 1024):
                        stream.write(block)
                if part.stat().st_size != end - start + 1:
                    raise RuntimeError("Truncated range download.")
                return part
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(2 * (attempt + 1))

    starts = list(range(0, total, chunk_size))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(fetch, start) for start in starts]
        for completed, future in enumerate(as_completed(futures), 1):
            future.result()
            if completed % 16 == 0 or completed == len(starts):
                print(f"{path.name}: {completed}/{len(starts)} chunks verified", flush=True)
    with path.open("wb") as output:
        for start in starts:
            with (parts / f"{start:012d}").open("rb") as stream:
                while block := stream.read(1024 * 1024):
                    output.write(block)


def digest_file(path, algorithm="sha256"):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, algorithm).hexdigest()


def obtain(path, urls, expected, algorithm="sha256"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if digest_file(path, algorithm) != expected:
            raise ValueError(f"Existing asset checksum mismatch: {path}")
        return path
    errors = []
    partial = path.with_name(path.name + ".partial")
    for url in urls:
        print(f"Downloading {path.name} from {url}", flush=True)
        result = subprocess.run([
            "curl.exe", "--fail", "--location", "--silent", "--show-error",
            "--connect-timeout", "20", "--speed-limit", "1024", "--speed-time", "90",
            "--retry", "2", "--output", str(partial), url,
        ], capture_output=True, text=True)
        if result.returncode != 0:
            try:
                print(f"Retrying {path.name} with resumable byte ranges...", flush=True)
                download_ranges(url, partial)
                result.returncode = 0
            except Exception as error:
                errors.append(f"{url}: range download failed: {error}")
        if result.returncode == 0 and digest_file(partial, algorithm) == expected:
            partial.replace(path)
            write_json(path.with_name(path.name + ".source.json"),
                       {"url": url, "checksum_algorithm": algorithm, "checksum": expected,
                        "sha256": digest_file(path), "size_bytes": path.stat().st_size})
            print(f"Verified {path.name} ({path.stat().st_size} bytes)", flush=True)
            return path
        errors.append(f"{url}: {result.stderr.strip() or 'checksum mismatch'}")
    raise RuntimeError("No valid download: " + " | ".join(errors))
