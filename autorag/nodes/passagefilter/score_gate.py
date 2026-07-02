from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from autorag.nodes.passagefilter.base import BasePassageFilter
from autorag.utils.util import convert_inputs_to_list, result_to_dataframe


def adaptive_score_gate(
	bi_scores: List[float],
	cross_scores: Optional[List[float]] = None,
	z: float = 0.0,
	max_keep: Optional[int] = None,
	reverse: bool = False,
) -> List[int]:
	"""
	Return the indices to retain via dual-score statistical gating.

	Instead of injecting a fixed top-K, this selects a query-adaptive
	number of chunks: a chunk is retained when its bi-encoder score clears
	a statistical gate ``mean + z * std`` computed over the query's own
	score distribution. When cross-encoder (reranker) scores are supplied,
	a chunk the bi-encoder gate rejects is *rescued* if its cross-encoder
	score clears the cross-encoder gate -- the core insight that
	cross-encoder affirmation recovers chunks bi-encoder retrieval ranks
	poorly due to vocabulary mismatch.

	Adapted from ScoreGate (Adaptive Chunk Selection for Retrieval-Augmented
	Generation via Dual-Score Statistical Fusion).

	:param bi_scores: Bi-encoder (retrieval) similarity scores for one query.
	:param cross_scores: Optional cross-encoder reranker scores for the same
	    chunks, in the same order. Higher always means more relevant.
	:param z: Number of standard deviations above the mean for the gate.
	    ``z=0`` keeps above-average chunks; larger ``z`` keeps fewer.
	:param max_keep: Optional hard cap on retained chunks (best bi-encoder
	    scores win ties for the cap).
	:param reverse: If True, the lower the bi-encoder score the better.
	:return: Sorted indices to retain (always at least one).
	"""
	bi = np.asarray(bi_scores, dtype=float)
	if bi.size == 0:
		return []

	bi_gate = float(bi.mean()) + z * float(bi.std())
	if reverse:
		# lower is better -> mirror the gate below the mean
		bi_gate = float(bi.mean()) - z * float(bi.std())
		keep = set(np.where(bi <= bi_gate)[0].tolist())
	else:
		keep = set(np.where(bi >= bi_gate)[0].tolist())

	if cross_scores is not None:
		cross = np.asarray(cross_scores, dtype=float)
		if cross.size == bi.size and cross.size > 0:
			# cross-encoder affirmation: higher reranker score is better
			cross_gate = float(cross.mean()) + z * float(cross.std())
			rescued = np.where(cross >= cross_gate)[0].tolist()
			keep.update(rescued)

	if not keep:
		# never drop everything: fall back to the single best chunk
		best = int(np.argmin(bi)) if reverse else int(np.argmax(bi))
		keep = {best}

	if max_keep is not None and len(keep) > max_keep:
		order = sorted(keep, key=lambda i: bi[i], reverse=not reverse)
		keep = set(order[:max_keep])

	return sorted(keep)


class ScoreGateFilter(BasePassageFilter):
	"""
	Adaptive chunk-selection passage filter using dual-score statistical
	fusion over the bi-encoder retrieval scores already surfaced by the
	pipeline (and, optionally, cross-encoder reranker scores).

	Drops in alongside the cutoff filters: same ``(contents, ids, scores)``
	in, filtered subset out, with no upstream/downstream data-shape change.
	"""

	@result_to_dataframe(["retrieved_contents", "retrieved_ids", "retrieve_scores"])
	def pure(self, previous_result: pd.DataFrame, *args, **kwargs):
		_, contents, scores, ids = self.cast_to_run(previous_result)
		return self._pure(contents, scores, ids, *args, **kwargs)

	def _pure(
		self,
		contents_list: List[List[str]],
		scores_list: List[List[float]],
		ids_list: List[List[str]],
		z: float = 0.0,
		max_keep: Optional[int] = None,
		reverse: bool = False,
		cross_scores_list: Optional[List[List[float]]] = None,
	) -> Tuple[List[List[str]], List[List[str]], List[List[float]]]:
		"""
		Select a query-adaptive number of chunks via statistical gating.
		Keeps at least one chunk per query. This is a filter and does not
		override scores.

		:param contents_list: List of content strings for each query.
		:param scores_list: Bi-encoder (retrieval) scores for each content.
		:param ids_list: List of ids for each content.
		:param z: Standard deviations above the mean for the score gate.
		    Default 0.0 (keep above-average chunks).
		:param max_keep: Optional hard cap on retained chunks per query.
		:param reverse: If True, the lower the score the better. Default False.
		:param cross_scores_list: Optional cross-encoder reranker scores per
		    query, enabling the dual-score rescue path.
		:return: Filtered lists of contents, ids, and scores.
		"""
		remain_indices = [
			self.__row_pure(
				scores,
				cross_scores_list[i] if cross_scores_list is not None else None,
				z,
				max_keep,
				reverse,
			)
			for i, scores in enumerate(scores_list)
		]

		remain_content_list = list(
			map(lambda c, idx: [c[i] for i in idx], contents_list, remain_indices)
		)
		remain_scores_list = list(
			map(lambda s, idx: [s[i] for i in idx], scores_list, remain_indices)
		)
		remain_ids_list = list(
			map(lambda _id, idx: [_id[i] for i in idx], ids_list, remain_indices)
		)

		return remain_content_list, remain_ids_list, remain_scores_list

	@convert_inputs_to_list
	def __row_pure(
		self,
		scores_list: List[float],
		cross_scores: Optional[List[float]],
		z: float,
		max_keep: Optional[int],
		reverse: bool,
	) -> List[int]:
		assert isinstance(scores_list, list), "scores_list must be a list."
		return adaptive_score_gate(scores_list, cross_scores, z, max_keep, reverse)
