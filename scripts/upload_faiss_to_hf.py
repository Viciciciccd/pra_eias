"""Upload the MedCPT FAISS index + source texts to the Hugging Face Hub.

This is a one-off publishing script.  It creates (if missing) the dataset repo
``process-reward-agents/medcpt-faiss-index`` as **private** and uploads the
four FAISS indices and their paired ``{source}_texts.json`` files.

Usage::

    python scripts/upload_faiss_to_hf.py \
        --src "$PRA_RETRIEVER_INDEX" \
        --repo process-reward-agents/medcpt-faiss-index \
        --private

The HF auth token is picked up from ``huggingface-cli login`` / ``hf auth login``
(or the ``HF_TOKEN`` env var).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi, create_repo


SOURCES = ["cpg", "recop", "textbooks", "statpearls"]


README = """\
# MedCPT FAISS index for `pra`

FAISS (`IndexFlatIP`, dim=768) indices built from MedCPT-Article-Encoder
embeddings, together with the paired `{source}_texts.json` payloads used by
[`process-reward-agents/pra`](https://github.com/) at retrieval time.

## Contents

| File                  | Description                                      |
| --------------------- | ------------------------------------------------ |
| `cpg.index`           | FAISS index for the clinical-practice-guideline corpus |
| `recop.index`         | FAISS index for the RECOP corpus                 |
| `textbooks.index`     | FAISS index for the Textbooks corpus             |
| `statpearls.index`    | FAISS index for the StatPearls corpus            |
| `{source}_texts.json` | Ordered list of `{"text", "source"}` entries that share the row order with `{source}.index` |

Encoder: [`ncbi/MedCPT-Article-Encoder`](https://huggingface.co/ncbi/MedCPT-Article-Encoder).
Query-time encoder: [`ncbi/MedCPT-Query-Encoder`](https://huggingface.co/ncbi/MedCPT-Query-Encoder).

## Usage (from `pra`)

```bash
pip install -e ".[training]"
pra-download-index                       # snapshot into $PRA_DATA_ROOT/faiss_index
export PRA_RETRIEVER_INDEX="$PRA_DATA_ROOT/faiss_index"
```

Or directly via `huggingface_hub`:

```python
from huggingface_hub import snapshot_download
path = snapshot_download(
    repo_id="process-reward-agents/medcpt-faiss-index",
    repo_type="dataset",
    local_dir="./faiss_index",
)
```

## License / access

Private while under review.  Contact the `process-reward-agents` org for access.
"""


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--src", required=True, help="Directory that contains {source}.index and {source}_texts.json")
    p.add_argument("--repo", default="process-reward-agents/medcpt-faiss-index",
                   help="HF repo id (default: process-reward-agents/medcpt-faiss-index)")
    p.add_argument("--private", action="store_true", default=True,
                   help="Create/keep the repo private (default: True)")
    p.add_argument("--public", dest="private", action="store_false",
                   help="Create the repo as public instead of private")
    p.add_argument("--token", default=None, help="HF token (falls back to cached login / HF_TOKEN env)")
    p.add_argument("--sources", nargs="+", default=SOURCES, help="Corpora to upload")
    p.add_argument("--dry_run", action="store_true", help="Print planned actions and exit")
    args = p.parse_args()

    src = Path(args.src).expanduser().resolve()
    if not src.is_dir():
        print(f"[upload] ERROR: --src not a directory: {src}", file=sys.stderr)
        return 2

    files = []
    for s in args.sources:
        idx = src / f"{s}.index"
        txt = src / f"{s}_texts.json"
        for f in (idx, txt):
            if not f.is_file():
                print(f"[upload] ERROR: missing expected file: {f}", file=sys.stderr)
                return 2
            files.append(f)

    total_bytes = sum(f.stat().st_size for f in files)
    print(f"[upload] source dir    : {src}")
    print(f"[upload] repo id       : {args.repo}  (private={args.private})")
    print(f"[upload] files ({len(files)}):")
    for f in files:
        print(f"         {f.name:32s} {f.stat().st_size / (1024 ** 3):6.2f} GiB")
    print(f"[upload] total         : {total_bytes / (1024 ** 3):.2f} GiB")

    if args.dry_run:
        print("[upload] --dry_run set; exiting without contacting HF")
        return 0

    token = args.token or os.environ.get("HF_TOKEN")

    create_repo(
        repo_id=args.repo,
        repo_type="dataset",
        private=args.private,
        exist_ok=True,
        token=token,
    )
    print(f"[upload] repo ready: https://huggingface.co/datasets/{args.repo}")

    api = HfApi(token=token)
    readme_path = src / "_PRA_README.md"
    readme_path.write_text(README)
    try:
        api.upload_file(
            path_or_fileobj=str(readme_path),
            path_in_repo="README.md",
            repo_id=args.repo,
            repo_type="dataset",
            commit_message="Add README",
        )
    finally:
        readme_path.unlink(missing_ok=True)

    allow = [f"{s}.index" for s in args.sources] + [f"{s}_texts.json" for s in args.sources]
    print(f"[upload] uploading {len(allow)} files via upload_large_folder ...")
    api.upload_large_folder(
        repo_id=args.repo,
        repo_type="dataset",
        folder_path=str(src),
        allow_patterns=allow,
        print_report=True,
    )
    print(f"[upload] done -> https://huggingface.co/datasets/{args.repo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
