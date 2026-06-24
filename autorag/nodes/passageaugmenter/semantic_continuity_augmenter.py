from typing import Dict, List, Union

import pandas as pd

from autorag.embedding.base import EmbeddingModel
from autorag.evaluation.metric.util import calculate_cosine_similarity
from autorag.nodes.passageaugmenter.base import BasePassageAugmenter
from autorag.utils.util import (
	filter_dict_keys,
	fetch_contents,
	embedding_query_content,
	result_to_dataframe,
	empty_cuda_cache,
)


def should_expand_neighbor(
	neighbor_score: float,
	anchor_score: float,
	relative_threshold: float,
	continuity_penalty: float,
	hop: int,
) -> bool:
	"""
	SCAR expansion decision rule.

	A neighboring chunk is kept only if its query-relevance, after a
	structural continuity penalty that grows with the hop distance from the
	anchor chunk, stays within a relative fraction of the anchor chunk's own
	query-relevance. Because the threshold is *relative* to the anchor score
	(rather than an absolute cutoff), the rule is approximately scale-invariant
	and transfers across embedding models without recalibration.

	:param neighbor_score: query-relevance of the candidate neighbor chunk.
	:param anchor_score: query-relevance of the retrieved (anchor) chunk.
	:param relative_threshold: fraction of the anchor score the neighbor must
	    reach to be expanded (e.g. 0.75).
	:param continuity_penalty: per-hop structural penalty subtracted from the
	    neighbor score; discourages drifting far from the anchor.
	:param hop: 1-based distance of the neighbor from the anchor.
	:return: True if the neighbor should be expanded.
	"""
	adjusted = neighbor_score - continuity_penalty * hop
	return adjusted >= relative_threshold * anchor_score


def _neighbor_hops(
	anchor_id: str, corpus_df: pd.DataFrame, key: str, max_hops: int
) -> List[str]:
	"""Walk prev_id/next_id metadata, returning ordered neighbor ids by hop."""
	hops = []
	current_id = anchor_id
	for _ in range(max_hops):
		rows = corpus_df.loc[corpus_df["doc_id"] == current_id, "metadata"].values
		if len(rows) == 0:
			break
		current_id = rows[0].get(key)
		if current_id is None:
			break
		hops.append(current_id)
	return hops


def collect_candidate_ids(
	anchor_ids: List[str],
	corpus_df: pd.DataFrame,
	mode: str,
	max_hops: int,
) -> List[str]:
	"""
	Gather every id that SCAR may need to score for a single query: the
	anchors plus all reachable neighbors within ``max_hops`` in the allowed
	directions. Order is preserved and duplicates removed so the candidates
	can be embedded once and reused.
	"""
	ordered: List[str] = []
	for anchor_id in anchor_ids:
		ids = [anchor_id]
		if mode in ("prev", "both"):
			ids += _neighbor_hops(anchor_id, corpus_df, "prev_id", max_hops)
		if mode in ("next", "both"):
			ids += _neighbor_hops(anchor_id, corpus_df, "next_id", max_hops)
		for id_ in ids:
			if id_ not in ordered:
				ordered.append(id_)
	return ordered


def scar_expand_query(
	anchor_ids: List[str],
	corpus_df: pd.DataFrame,
	score_lookup: Dict[str, float],
	mode: str,
	max_hops: int,
	relative_threshold: float,
	continuity_penalty: float,
) -> List[str]:
	"""
	Apply the SCAR policy to one query's retrieved chunks.

	For each anchor chunk we walk outward in the allowed directions, expanding
	a neighbor only while :func:`should_expand_neighbor` holds. Expansion in a
	direction stops at the first rejected hop (continuity is contiguous). The
	result preserves contextual order (prev neighbors, anchor, next neighbors)
	and is de-duplicated across anchors to keep the context compact.
	"""
	if mode not in ("prev", "next", "both"):
		raise ValueError(f"mode must be 'prev', 'next', or 'both', but got {mode}")

	augmented: List[str] = []

	def accept_direction(anchor_id: str, anchor_score: float, key: str) -> List[str]:
		kept = []
		for hop, neighbor_id in enumerate(
			_neighbor_hops(anchor_id, corpus_df, key, max_hops), start=1
		):
			neighbor_score = score_lookup.get(neighbor_id, 0.0)
			if not should_expand_neighbor(
				neighbor_score,
				anchor_score,
				relative_threshold,
				continuity_penalty,
				hop,
			):
				break
			kept.append(neighbor_id)
		return kept

	for anchor_id in anchor_ids:
		anchor_score = score_lookup.get(anchor_id, 0.0)
		current_ids = [anchor_id]
		if mode in ("prev", "both"):
			current_ids = (
				accept_direction(anchor_id, anchor_score, "prev_id")[::-1] + current_ids
			)
		if mode in ("next", "both"):
			current_ids += accept_direction(anchor_id, anchor_score, "next_id")
		for id_ in current_ids:
			if id_ not in augmented:
				augmented.append(id_)

	return augmented


class SemanticContinuityAugmenter(BasePassageAugmenter):
	"""
	Semantic Continuity-Aware Retrieval (SCAR) passage augmenter.

	Adapted from "SCAR: Semantic Continuity-Aware Retrieval for Efficient
	Context Expansion in RAG" (arXiv:2606.16661). Unlike
	:class:`PrevNextPassageAugmenter`, which expands a fixed number of
	neighbors for every retrieved chunk, SCAR *selectively* expands neighbors
	based on a relative expansion threshold tied to each anchor chunk's own
	query-relevance. This repairs boundary fragmentation while keeping the
	added token overhead small.
	"""

	def __init__(
		self,
		project_dir: str,
		embedding_model: Union[str, dict] = "openai",
		*args,
		**kwargs,
	):
		"""
		Initialize the SemanticContinuityAugmenter module.

		:param project_dir: The project directory.
		:param embedding_model: The embedding model name used to compute
		    query-neighbor relevance. Default is openai.
		"""
		super().__init__(project_dir, *args, **kwargs)
		slim_corpus_df = self.corpus_df[["doc_id", "metadata"]]
		slim_corpus_df.loc[:, "metadata"] = slim_corpus_df["metadata"].apply(
			filter_dict_keys, keys=["prev_id", "next_id"]
		)
		self.slim_corpus_df = slim_corpus_df

		self.embedding_model = EmbeddingModel.load(embedding_model)()

	def __del__(self):
		del self.embedding_model
		empty_cuda_cache()
		super().__del__()

	@result_to_dataframe(["retrieved_contents", "retrieved_ids", "retrieve_scores"])
	def pure(self, previous_result: pd.DataFrame, *args, **kwargs):
		"""
		Run the passage augmenter node - SemanticContinuityAugmenter module.

		:param previous_result: The previous result DataFrame.
		:param top_k: The top_k value to select the final results.
		:param mode: Expansion direction - 'prev', 'next', or 'both'. Default 'both'.
		:param max_hops: Maximum neighbors expanded per direction. Default 3.
		:param relative_threshold: Fraction of the anchor relevance a neighbor
		    must reach to be expanded. Default 0.75.
		:param continuity_penalty: Per-hop structural penalty. Default 0.05.
		:return: DataFrame with retrieved_contents, retrieved_ids, retrieve_scores.
		"""
		top_k = kwargs.pop("top_k")
		mode = kwargs.pop("mode", "both")
		max_hops = kwargs.pop("max_hops", 3)
		relative_threshold = kwargs.pop("relative_threshold", 0.75)
		continuity_penalty = kwargs.pop("continuity_penalty", 0.05)

		ids = self.cast_to_run(previous_result)
		assert "query" in previous_result.columns, (
			"previous_result must have query column."
		)
		queries = previous_result["query"].tolist()

		# Embed once over the full candidate frontier (anchors + reachable
		# neighbors) per query, then reuse the scores for both the expansion
		# decision and the final ranking.
		candidate_ids = [
			collect_candidate_ids(anchor_ids, self.slim_corpus_df, mode, max_hops)
			for anchor_ids in ids
		]
		candidate_contents = fetch_contents(self.corpus_df, candidate_ids)
		query_embeddings, content_embeddings = embedding_query_content(
			queries, candidate_contents, self.embedding_model, batch=128
		)
		score_lookups = [
			{
				cand_id: calculate_cosine_similarity(query_embedding, content_embedding)
				for cand_id, content_embedding in zip(cand_ids, content_embs)
			}
			for query_embedding, cand_ids, content_embs in zip(
				query_embeddings, candidate_ids, content_embeddings
			)
		]

		augmented_ids = self._pure(
			ids,
			score_lookups,
			mode,
			max_hops,
			relative_threshold,
			continuity_penalty,
		)

		augmented_contents = fetch_contents(self.corpus_df, augmented_ids)
		augmented_scores = [
			[float(score_lookup.get(id_, 0.0)) for id_ in id_list]
			for id_list, score_lookup in zip(augmented_ids, score_lookups)
		]
		return self.sort_by_scores(
			augmented_contents, augmented_ids, augmented_scores, top_k
		)

	def _pure(
		self,
		ids_list: List[List[str]],
		score_lookups: List[Dict[str, float]],
		mode: str = "both",
		max_hops: int = 3,
		relative_threshold: float = 0.75,
		continuity_penalty: float = 0.05,
	) -> List[List[str]]:
		"""
		Apply the SCAR expansion policy to every query's retrieved ids.

		:param ids_list: per-query lists of retrieved (anchor) ids.
		:param score_lookups: per-query {id: query-relevance} maps covering the
		    anchors and their candidate neighbors.
		:return: per-query lists of augmented ids.
		"""
		return [
			scar_expand_query(
				anchor_ids,
				self.slim_corpus_df,
				score_lookup,
				mode,
				max_hops,
				relative_threshold,
				continuity_penalty,
			)
			for anchor_ids, score_lookup in zip(ids_list, score_lookups)
		]
