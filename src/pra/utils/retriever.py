import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional

import faiss
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer


DEFAULT_TOPK = 200
DEFAULT_RERANK_TOPK = 64
DEFAULT_DEVICE = "cuda:1"
DEFAULT_SOURCES = ["cpg", "recop", "textbooks", "statpearls"]


class InMemorySources:
    def __init__(self, index_path: str, sources: Optional[List[str]] = None):
        if not (index_path or "").strip():
            raise ValueError("retriever index_path is empty; set --retriever_index_path")
        self.data = {}
        for source in sources or DEFAULT_SOURCES:
            with open(f"{index_path}/{source}_texts.json") as f:
                self.data[source] = json.load(f)

    def get(self, source, idx):
        return self.data[source][idx]


class MedCPTRetriever:
    def __init__(
        self,
        device: str = DEFAULT_DEVICE,
        rerank_devices: Optional[List[str]] = None,
        index_path: str = "",
        sources: Optional[List[str]] = None,
    ):
        if not (index_path or "").strip():
            raise ValueError("MedCPTRetriever requires a non-empty index_path")
        self.device = device
        self.index_path = index_path
        self.sources = sources or DEFAULT_SOURCES
        self.rerank_devices = rerank_devices or [device]

        self.model_q = AutoModel.from_pretrained("ncbi/MedCPT-Query-Encoder").to(device)
        self.model_q.eval()
        self.tokenizer_q = AutoTokenizer.from_pretrained("ncbi/MedCPT-Query-Encoder")

        self.models_c = {}
        for dev in self.rerank_devices:
            model = AutoModelForSequenceClassification.from_pretrained("ncbi/MedCPT-Cross-Encoder").to(dev)
            model.eval()
            self.models_c[dev] = model
        self.model_c = self.models_c[self.rerank_devices[0]]
        self.tokenizer_c = AutoTokenizer.from_pretrained("ncbi/MedCPT-Cross-Encoder")

        self.in_mem_sources = InMemorySources(index_path=index_path, sources=self.sources)
        self.faiss_indices = {
            source: faiss.read_index(f"{self.index_path}/{source}.index")
            for source in self.sources
        }

    def batch_encode(self, queries: List[str], batch_size: int = 128, show_progress: bool = False) -> np.ndarray:
        """Encode queries in mini-batches."""
        all_embeds = []
        batches = list(range(0, len(queries), batch_size))

        start_time = time.time()
        pbar = tqdm(batches, desc="    Encoding queries", unit="batch") if show_progress else None

        for batch_idx, i in enumerate(batches):
            batch = queries[i : i + batch_size]
            encoded = self.tokenizer_q(
                batch, truncation=True, padding=True, return_tensors="pt", max_length=512
            )
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            with torch.inference_mode():
                embeds = self.model_q(**encoded).last_hidden_state[:, 0, :]
            all_embeds.append(embeds.cpu().numpy())

            if pbar is not None:
                pbar.update(1)
                elapsed = time.time() - start_time
                queries_done = min((batch_idx + 1) * batch_size, len(queries))
                speed = queries_done / elapsed if elapsed > 0 else 0
                pbar.set_postfix({"speed": f"{speed:.1f}q/s", "queries": queries_done})

        if pbar is not None:
            pbar.close()

        return np.concatenate(all_embeds, axis=0)

    def _search_single_source(self, source: str, query_embeddings: np.ndarray, topk: int) -> List[List[dict]]:
        """Search a single FAISS index for multiple queries."""
        index = self.faiss_indices[source]
        scores, indices = index.search(query_embeddings, topk)

        results_per_query = []
        for q_idx in range(len(query_embeddings)):
            results = []
            for idx, score in zip(indices[q_idx], scores[q_idx]):
                result = self.in_mem_sources.get(source, idx).copy()
                result["score"] = float(score)
                results.append(result)
            results.sort(key=lambda x: x["score"], reverse=True)
            results_per_query.append(results[:topk])
        return results_per_query

    def batch_faiss_retrieve(
        self,
        queries: List[str],
        allowed_sources: List[str],
        topk: int,
        encode_batch_size: int = 128,
        show_progress: bool = False,
    ) -> List[List[str]]:
        """Batch retrieval for multiple queries with parallel source search."""
        query_embeddings = self.batch_encode(
            queries, batch_size=encode_batch_size, show_progress=show_progress
        ).astype("float32")

        source_results = {}
        with ThreadPoolExecutor(max_workers=len(allowed_sources)) as executor:
            futures = {
                executor.submit(self._search_single_source, source, query_embeddings, topk): source
                for source in allowed_sources
            }
            for future in as_completed(futures):
                source_results[futures[future]] = future.result()

        all_results = []
        for q_idx in range(len(queries)):
            combined = []
            for source in allowed_sources:
                combined.extend(r["text"] for r in source_results[source][q_idx])
            all_results.append(combined)
        return all_results

    def _rerank_single_with_tokenizer(self, tokenizer, query: str, docs: List[str], q_idx: int) -> tuple:
        """Rerank documents for a single query; uses a dedicated tokenizer to avoid thread-safety issues."""
        if not docs:
            return q_idx, []

        pairs = [[query, doc] for doc in docs]
        encoded = tokenizer(
            pairs, truncation=True, padding=True, return_tensors="pt", max_length=512
        )
        with torch.inference_mode():
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            logits = self.model_c(**encoded).logits.squeeze(dim=1).cpu()

        doc_scores = sorted(zip(docs, logits.tolist()), key=lambda x: x[1], reverse=True)
        return q_idx, [doc for doc, _ in doc_scores]

    def batch_rerank(
        self,
        queries: List[str],
        doc_lists: List[List[str]],
        show_progress: bool = False,
        num_workers: int = 8,
    ) -> List[List[str]]:
        """Rerank documents using ThreadPoolExecutor for parallel query processing."""
        if not queries:
            return []

        results: List[Optional[List[str]]] = [None] * len(queries)
        start_time = time.time()

        # Per-worker tokenizers because HF fast tokenizers are not always thread-safe.
        tokenizers = [
            AutoTokenizer.from_pretrained("ncbi/MedCPT-Cross-Encoder")
            for _ in range(num_workers)
        ]

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(
                    self._rerank_single_with_tokenizer,
                    tokenizers[q_idx % num_workers],
                    query,
                    docs,
                    q_idx,
                ): q_idx
                for q_idx, (query, docs) in enumerate(zip(queries, doc_lists))
            }

            iterator = as_completed(futures)
            if show_progress:
                iterator = tqdm(iterator, total=len(queries), desc="    Reranking", unit="q")
            for future in iterator:
                q_idx, ranked_docs = future.result()
                results[q_idx] = ranked_docs
                if show_progress:
                    elapsed = time.time() - start_time
                    done = sum(1 for r in results if r is not None)
                    speed = done / elapsed if elapsed > 0 else 0
                    iterator.set_postfix({"speed": f"{speed:.1f}q/s"})
            if show_progress:
                iterator.close()

        return results


_retriever_instance: Optional[MedCPTRetriever] = None


def get_retriever(
    device: str = DEFAULT_DEVICE,
    rerank_devices: Optional[List[str]] = None,
    index_path: Optional[str] = None,
    sources: Optional[List[str]] = None,
) -> MedCPTRetriever:
    global _retriever_instance
    if _retriever_instance is None:
        if not (index_path or "").strip():
            raise ValueError(
                "get_retriever(index_path=...) is required on first call (no baked-in dataset path)"
            )
        _retriever_instance = MedCPTRetriever(
            device=device,
            rerank_devices=rerank_devices,
            index_path=index_path,
            sources=sources,
        )
    return _retriever_instance


def batch_retrieve(
    queries: List[str],
    allowed_sources: Optional[List[str]] = None,
    topk: int = DEFAULT_TOPK,
    rerank_topk: int = DEFAULT_RERANK_TOPK,
    device: str = DEFAULT_DEVICE,
    encode_batch_size: int = 128,
    rerank_workers: int = 8,
    use_reranker: bool = True,
    show_progress: bool = False,
) -> List[List[str]]:
    """Batch retrieve and rerank documents for multiple queries."""
    if not queries:
        return []
    retriever = get_retriever(device=device)
    allowed_sources = allowed_sources or retriever.sources

    if show_progress:
        print(f"  [Retrieval] {len(queries)} queries (rerank={'ON' if use_reranker else 'OFF'})")

    t0 = time.time()
    doc_lists = retriever.batch_faiss_retrieve(
        queries, allowed_sources, topk=topk,
        encode_batch_size=encode_batch_size, show_progress=show_progress,
    )
    if show_progress:
        dt = time.time() - t0
        speed = len(queries) / dt if dt > 0 else 0
        print(f"    FAISS done: {len(queries)} queries in {dt:.2f}s ({speed:.2f}q/s)")

    if not use_reranker:
        return [docs[:rerank_topk] for docs in doc_lists]

    t0 = time.time()
    reranked_lists = retriever.batch_rerank(
        queries, doc_lists, show_progress=show_progress, num_workers=rerank_workers
    )
    if show_progress:
        dt = time.time() - t0
        speed = len(queries) / dt if dt > 0 else 0
        print(f"    Rerank done: {len(queries)} queries in {dt:.2f}s ({speed:.2f}q/s)")

    return [docs[:rerank_topk] for docs in reranked_lists]
