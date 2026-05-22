from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional

import faiss
import numpy as np
import torch
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer


DEFAULT_SOURCES = ("cpg", "recop", "textbooks", "statpearls")
STEP_PATTERN = re.compile(
    r"(?ims)^\s*steps?\s*(\d+)\s*:\s*([\s\S]*?)(?=^\s*steps?\s*\d+\s*:|\Z)"
)


class InMemorySources:
    def __init__(self, index_path: str, sources=DEFAULT_SOURCES):
        self.data = {}
        for source in sources:
            with open(os.path.join(index_path, f"{source}_texts.json")) as f:
                self.data[source] = json.load(f)

    def get(self, source: str, idx: int):
        return self.data[source][idx]


class MedCPTRetriever:
    def __init__(
        self,
        device: str,
        index_path: str,
        shared_sources: Optional[InMemorySources] = None,
        shared_faiss_indices: Optional[dict] = None,
        sources=DEFAULT_SOURCES,
    ):
        self.device = device
        self.index_path = index_path
        self.sources = tuple(sources)

        self.model_q = AutoModel.from_pretrained("ncbi/MedCPT-Query-Encoder").to(device)
        self.model_q.eval()
        self.tokenizer_q = AutoTokenizer.from_pretrained("ncbi/MedCPT-Query-Encoder")
        self.model_c = AutoModelForSequenceClassification.from_pretrained("ncbi/MedCPT-Cross-Encoder").to(device)
        self.model_c.eval()
        self.tokenizer_c = AutoTokenizer.from_pretrained("ncbi/MedCPT-Cross-Encoder")

        self.in_mem_sources = shared_sources or InMemorySources(index_path, sources=self.sources)
        if shared_faiss_indices is not None:
            self.faiss_indices = shared_faiss_indices
        else:
            self.faiss_indices = {
                s: faiss.read_index(os.path.join(index_path, f"{s}.index")) for s in self.sources
            }

    def encode(self, query: str) -> np.ndarray:
        encoded = self.tokenizer_q(
            query, truncation=True, padding=True, return_tensors="pt", max_length=512
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        with torch.inference_mode():
            embeds = self.model_q(**encoded).last_hidden_state[:, 0, :]
        return np.asarray(embeds.cpu().numpy()[0], dtype="float32")

    def faiss_retrieve(self, query: str, allowed_sources: List[str], topk: int) -> List[str]:
        query_embedding = np.array([self.encode(query)], dtype="float32")
        evidence = []
        for source in allowed_sources:
            scores, indices = self.faiss_indices[source].search(query_embedding, topk)
            hits = []
            for idx, score in zip(indices[0], scores[0]):
                row = self.in_mem_sources.get(source, idx).copy()
                row["score"] = float(score)
                hits.append(row)
            hits.sort(key=lambda r: r["score"], reverse=True)
            evidence.extend(r["text"] for r in hits[:topk])
        return evidence

    def rerank(self, query: str, docs: List[str]) -> List[str]:
        pairs = [[query, d] for d in docs]
        with torch.inference_mode():
            enc = self.tokenizer_c(
                pairs, truncation=True, padding=True, return_tensors="pt", max_length=512
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            logits = self.model_c(**enc).logits.squeeze(dim=1).detach().cpu()
        order = torch.argsort(logits, descending=True)
        return [docs[i] for i in order]

    def retrieve_and_rerank(
        self, query: str, allowed_sources: List[str], topk: int, rerank_topk: int
    ) -> List[str]:
        docs = self.faiss_retrieve(query, allowed_sources, topk)
        return self.rerank(query, docs)[:rerank_topk]


def _options_str(options) -> str:
    return "\n".join(f"{k}: {v}" for k, v in options.items()) if options else ""


def _build_stepwise_retrievals(
    retriever: MedCPTRetriever,
    item: dict,
    steps,
    *,
    version: str,
    sources: List[str],
    topk: int,
    rerank_topk: int,
) -> List[dict]:
    question = item["question"]
    options_str = _options_str(item["options"])
    out = []
    for k in range(len(steps)):
        if version == "window":
            start = max(0, k - 1)
            window = steps[start : k + 1]
        else:  # cumulative
            window = steps[: k + 1]
        body = "\n".join(f"Step {n}: {s}" for n, s in window)
        query = f"{question}\n\n{options_str}\n\n{body}"
        docs = retriever.retrieve_and_rerank(query, sources, topk=topk, rerank_topk=rerank_topk)
        out.append({"retrieval_query": query, "related_docs": docs})
    return out


def _process_item(args):
    idx, item, retriever, opts = args
    model_outputs = item["model_outputs"]
    moi = opts["model_output_idx"]
    if not (0 <= moi < len(model_outputs)):
        return idx, None, 0

    model_output = model_outputs[moi]
    steps = [(int(n), body.strip()) for n, body in STEP_PATTERN.findall(model_output)]
    retrievals = _build_stepwise_retrievals(
        retriever, item, steps,
        version=opts["version"], sources=opts["sources"],
        topk=opts["topk"], rerank_topk=opts["rerank_topk"],
    )

    filtered = item.copy()
    filtered["model_outputs_steps"] = [
        {"model_output": model_output, "steps": steps, "stepwise_retrievals": retrievals}
    ]
    filtered["question_id"] = f"{idx}_{moi}"
    return idx, filtered, len(steps)


def run(
    input_path: str,
    output_path: str,
    index_path: str,
    *,
    version: str = "window",
    sources: Optional[List[str]] = None,
    topk: int = 200,
    rerank_topk: int = 64,
    model_output_idx: int = 0,
    workers_per_gpu: int = 4,
) -> None:
    if version not in ("cumulative", "window"):
        raise ValueError(f"Unknown version {version!r}; must be 'cumulative' or 'window'")

    sources = list(sources or DEFAULT_SOURCES)
    devices = (
        [f"cuda:{i}" for i in range(torch.cuda.device_count())]
        if torch.cuda.is_available()
        else ["cpu"]
    )
    print(f"[retrieve] version={version}  devices={devices}  sources={sources}")

    with open(input_path) as f:
        items = json.load(f)
    print(f"[retrieve] loaded {len(items)} items from {input_path}")

    print("[retrieve] loading shared sources and FAISS indices...")
    shared_sources = InMemorySources(index_path, sources=sources)
    shared_faiss_indices = {
        s: faiss.read_index(os.path.join(index_path, f"{s}.index")) for s in sources
    }

    retrievers = []
    for dev in devices:
        print(f"[retrieve] building retriever on {dev}")
        retrievers.append(
            MedCPTRetriever(
                dev,
                index_path,
                shared_sources=shared_sources,
                shared_faiss_indices=shared_faiss_indices,
                sources=sources,
            )
        )

    opts = {
        "version": version,
        "sources": sources,
        "topk": topk,
        "rerank_topk": rerank_topk,
        "model_output_idx": model_output_idx,
    }
    task_args = [(idx, item, retrievers[idx % len(retrievers)], opts) for idx, item in enumerate(items)]

    total_workers = workers_per_gpu * len(devices)
    print(f"[retrieve] running with {total_workers} workers across {len(devices)} device(s)")

    from tqdm.auto import tqdm

    step_lengths = []
    collected = []
    with ThreadPoolExecutor(max_workers=total_workers) as pool:
        futures = {pool.submit(_process_item, a): a[0] for a in task_args}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="retrieve"):
            try:
                idx, result, step_count = fut.result()
                step_lengths.append(step_count)
                if result is not None:
                    collected.append((idx, result))
            except Exception as e:
                print(f"[retrieve] error item {futures[fut]}: {e}", file=sys.stderr)

    collected.sort(key=lambda x: x[0])
    filtered = [r for _, r in collected]
    print(f"[retrieve] step-count histogram: {collections.Counter(step_lengths)}")
    print(f"[retrieve] kept {len(filtered)}/{len(items)} items")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(filtered, f, indent=2)
    print(f"[retrieve] wrote {output_path}")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MedCPT retrieval for training data")
    p.add_argument("--input", required=True, help="Input JSON (train_results_*.json)")
    p.add_argument("--output", required=True, help="Output JSON path")
    p.add_argument("--index_path", required=True, help="Directory with *.index and *_texts.json files")
    p.add_argument("--version", default="window", choices=["cumulative", "window"],
                   help="cumulative=all-steps-so-far, window=last-2-steps (default)")
    p.add_argument("--sources", nargs="+", default=list(DEFAULT_SOURCES))
    p.add_argument("--topk", type=int, default=200)
    p.add_argument("--rerank_topk", type=int, default=64)
    p.add_argument("--model_output_idx", type=int, default=0,
                   help="Which model_output to process (default: 0)")
    p.add_argument("--workers_per_gpu", type=int, default=4)
    return p.parse_args(argv)


def main_cli() -> None:
    args = parse_args()
    run(
        args.input,
        args.output,
        args.index_path,
        version=args.version,
        sources=args.sources,
        topk=args.topk,
        rerank_topk=args.rerank_topk,
        model_output_idx=args.model_output_idx,
        workers_per_gpu=args.workers_per_gpu,
    )


if __name__ == "__main__":
    main_cli()
