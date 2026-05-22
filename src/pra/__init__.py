import os as _os

_os.environ.setdefault("PRA_DATA_ROOT", "./data")
_os.environ.setdefault("PRA_OUTPUT_ROOT", "./outputs")
_os.environ.setdefault("PRA_RETRIEVER_INDEX", "./data/faiss_index")

__all__: list[str] = []