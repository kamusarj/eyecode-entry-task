"""Fetch the exact raw-file list/revision used in the recorded HoMMI run."""
import argparse
import json
from pathlib import Path

from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "data/hommi_raw"))
    args = ap.parse_args()
    manifest = json.loads((ROOT / "configs/hommi_download_manifest.json").read_text())
    for i, name in enumerate(manifest["files"], 1):
        hf_hub_download(manifest["repo_id"], name, repo_type="dataset",
                        revision=manifest["revision"], local_dir=args.out)
        if i % 20 == 0:
            print(f"{i}/{len(manifest['files'])} files", flush=True)


if __name__ == "__main__":
    main()
