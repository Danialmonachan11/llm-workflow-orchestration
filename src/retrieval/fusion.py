"""Result fusion for hybrid (dense + sparse, single- or multi-backend)
retrieval: Reciprocal Rank Fusion (RRF).

RRF combines several independently-ranked result lists (e.g. a dense
vector search from Chroma, a dense vector search from Qdrant, and a
sparse BM25 keyword search) into one ranking, without needing the raw
scores from each method to be on comparable scales -- cosine similarity,
Qdrant's distance score, and a BM25 term-frequency score are not directly
comparable numbers, but their *rank positions* are. Standard formula:

    RRF(d) = sum over each result list L that contains d of  1 / (k + rank_L(d))

k=60 is the constant from the original RRF paper (Cormack et al., 2009)
and is what most hybrid-search implementations default to.
"""

from typing import List, Dict
from ..vectordb.base import SearchResult, Document


def _result_key(result: SearchResult) -> str:
    """Stable dedup key for a search result across different backends.

    Always the content string, never the backend-assigned id. IDs are not
    a safe dedup key here: a BM25 candidate is built straight from the raw
    seed Document objects and never gets one assigned, while Chroma/Qdrant
    mint their own backend-specific id for the same logical document -- so
    two backends that both matched the identical passage would fuse under
    two different keys and survive as visible duplicates in the final
    ranked/reranked output (caught by actually running hybrid search
    end-to-end, not by reasoning about the code in the abstract: the same
    GPT passage showed up twice, back to back, with an identical reranker
    score, once this path was exercised for real). Content is the one
    identity that is genuinely consistent across every source.
    """
    return result.document.content


def reciprocal_rank_fusion(
    result_lists: List[List[SearchResult]],
    k: int = 60,
) -> List[SearchResult]:
    """Fuse multiple ranked result lists into one RRF-ranked list.

    Each input list should already be sorted best-first (as every
    VectorDBBase.search() implementation and the BM25 search in this repo
    both return). The returned list is deduplicated (a document appearing
    in more than one input list is merged into a single entry, its fused
    score being the sum of its per-list RRF contributions) and sorted by
    fused score, best first. Each returned SearchResult's `.score` field
    is overwritten with the fused RRF score -- it is not comparable to
    the original per-backend similarity/BM25 score, by design (that's the
    whole point of RRF: rank position, not raw score, is what fuses
    cleanly across heterogeneous retrieval methods).
    """
    fused_scores: Dict[str, float] = {}
    representative: Dict[str, SearchResult] = {}

    for result_list in result_lists:
        for rank, result in enumerate(result_list, start=1):
            key = _result_key(result)
            fused_scores[key] = fused_scores.get(key, 0.0) + 1.0 / (k + rank)
            # Keep the first-seen (i.e. highest-ranked in whichever list
            # first contained it) representative document/metadata.
            if key not in representative:
                representative[key] = result

    ranked_keys = sorted(fused_scores.keys(), key=lambda key: fused_scores[key], reverse=True)

    fused_results: List[SearchResult] = []
    for key in ranked_keys:
        original = representative[key]
        fused_results.append(
            SearchResult(
                document=original.document,
                score=fused_scores[key],
                distance=original.distance,
            )
        )
    return fused_results
