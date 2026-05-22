from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


DEFAULT_REPO = "process-reward-agents/medcpt-faiss-index"
DEFAULT_SOURCES = ["cpg", "recop", "textbooks", "statpearls"]


def default_dest() -> Path:
    root = os.environ["PRA_DATA_ROOT"]
    return Path(os.path.expandvars(root)) / "faiss_index"


def download_index(
    dest: str | os.PathLike | None = None,
    repo_id: str = DEFAULT_REPO,
    sources: list[str] | None = None,
    token: str | None = None,
    local_dir_use_symlinks: bool = False,
) -> Path:

    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise RuntimeError(
            "huggingface_hub is required to download the FAISS index. "
            "Install with `pip install -e \".[training]\"` or "
            "`pip install huggingface_hub`."
        ) from e

    dest_path = Path(os.fspath(dest) if dest is not None else default_dest())
    dest_path.mkdir(parents=True, exist_ok=True)

    sources = sources or DEFAULT_SOURCES
    allow = [f"{s}.index" for s in sources] + [f"{s}_texts.json" for s in sources]

    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(dest_path),
        local_dir_use_symlinks=local_dir_use_symlinks,
        allow_patterns=allow,
        token=token or os.environ.get("HF_TOKEN"),
    )
    return dest_path


def main_cli() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--dest", default=None,
                   help="Destination directory (default: $PRA_DATA_ROOT/faiss_index)")
    p.add_argument("--repo", default=DEFAULT_REPO, help=f"HF dataset repo id (default: {DEFAULT_REPO})")
    p.add_argument("--sources", nargs="+", default=DEFAULT_SOURCES,
                   choices=DEFAULT_SOURCES, help="Subset of corpora to fetch (default: all)")
    p.add_argument("--token", default=None, help="HF token (falls back to cached login / HF_TOKEN)")
    args = p.parse_args()

    try:
        out = download_index(
            dest=args.dest, repo_id=args.repo, sources=args.sources, token=args.token,
        )
    except Exception as e:
        print(f"[download_index] ERROR: {e}", file=sys.stderr)
        return 2

    print(f"[download_index] ready at: {out}")
    print(f"[download_index] files:")
    for f in sorted(out.iterdir()):
        if f.is_file():
            print(f"                 {f.name:32s} {f.stat().st_size / (1024 ** 3):6.2f} GiB")
    print()
    print(f"    export PRA_RETRIEVER_INDEX={out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main_cli())
