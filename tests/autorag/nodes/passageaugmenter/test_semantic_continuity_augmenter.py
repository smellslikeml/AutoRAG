from unittest.mock import patch

import pytest
from llama_index.embeddings.openai import OpenAIEmbedding

# Import through the EXISTING dispatcher to prove the SCAR module is wired
# into the passage_augmenter node, not just importable on its own.
from autorag.support import get_support_modules
from autorag.nodes.passageaugmenter import SemanticContinuityAugmenter
from autorag.nodes.passageaugmenter.semantic_continuity_augmenter import (
	should_expand_neighbor,
	collect_candidate_ids,
	scar_expand_query,
)

from tests.autorag.nodes.passageaugmenter.test_base_passage_augmenter import (
	ids_list,
	project_dir,
	previous_result,
	corpus_data,
	doc_id_list,
)
from tests.mock import mock_get_text_embedding_batch


def test_scar_registered_in_support():
	# The wiring edit in autorag/support.py must resolve both the snake_case
	# and class-name aliases to the new module.
	assert get_support_modules("scar_augmenter") is SemanticContinuityAugmenter
	assert (
		get_support_modules("SemanticContinuityAugmenter")
		is SemanticContinuityAugmenter
	)


def test_should_expand_neighbor_relative_rule():
	# Neighbor as relevant as the anchor (no penalty) -> expand.
	assert should_expand_neighbor(
		0.8, 0.8, relative_threshold=0.75, continuity_penalty=0.0, hop=1
	)
	# Neighbor well below the relative threshold -> reject.
	assert not should_expand_neighbor(
		0.4, 0.8, relative_threshold=0.75, continuity_penalty=0.0, hop=1
	)
	# Same scores but the per-hop penalty pushes a far neighbor below threshold.
	assert should_expand_neighbor(
		0.8, 0.8, relative_threshold=0.75, continuity_penalty=0.05, hop=1
	)
	assert not should_expand_neighbor(
		0.8, 0.8, relative_threshold=0.75, continuity_penalty=0.05, hop=5
	)


def test_scar_expands_only_continuous_neighbors():
	augmenter = SemanticContinuityAugmenter(project_dir=project_dir)
	anchor = doc_id_list[1]
	# next neighbor is highly relevant, the one after is not -> stop after one hop.
	score_lookup = {
		doc_id_list[1]: 0.9,
		doc_id_list[2]: 0.88,
		doc_id_list[3]: 0.10,
	}
	result = augmenter._pure(
		[[anchor]],
		[score_lookup],
		mode="next",
		max_hops=3,
		relative_threshold=0.75,
		continuity_penalty=0.0,
	)
	assert result == [[doc_id_list[1], doc_id_list[2]]]


def test_scar_no_expansion_when_neighbors_irrelevant():
	augmenter = SemanticContinuityAugmenter(project_dir=project_dir)
	anchor = doc_id_list[1]
	score_lookup = {doc_id_list[1]: 0.9, doc_id_list[2]: 0.1, doc_id_list[0]: 0.1}
	result = augmenter._pure(
		[[anchor]],
		[score_lookup],
		mode="both",
		relative_threshold=0.75,
		continuity_penalty=0.0,
	)
	# Irrelevant neighbors are dropped: SCAR keeps the context compact.
	assert result == [[anchor]]


def test_collect_candidate_ids_dedupes():
	augmenter = SemanticContinuityAugmenter(project_dir=project_dir)
	candidates = collect_candidate_ids(
		[doc_id_list[1]], augmenter.slim_corpus_df, mode="both", max_hops=1
	)
	# anchor + prev + next, no duplicates, anchor present.
	assert doc_id_list[1] in candidates
	assert len(candidates) == len(set(candidates))


def test_scar_expand_query_pure_function():
	# The policy core is usable without an embedding model.
	augmenter = SemanticContinuityAugmenter(project_dir=project_dir)
	result = scar_expand_query(
		[doc_id_list[1]],
		augmenter.slim_corpus_df,
		{doc_id_list[1]: 1.0, doc_id_list[2]: 0.95},
		mode="next",
		max_hops=2,
		relative_threshold=0.75,
		continuity_penalty=0.0,
	)
	assert result[0] == doc_id_list[1]
	assert doc_id_list[2] in result


@patch.object(
	OpenAIEmbedding,
	"get_text_embedding_batch",
	mock_get_text_embedding_batch,
)
def test_scar_augmenter_node():
	result_df = SemanticContinuityAugmenter.run_evaluator(
		project_dir=project_dir,
		previous_result=previous_result,
		mode="both",
		top_k=2,
	)
	contents = result_df["retrieved_contents"].tolist()
	ids = result_df["retrieved_ids"].tolist()
	scores = result_df["retrieve_scores"].tolist()
	assert len(contents) == len(ids) == len(scores) == 2
	for content_list, id_list, score_list in zip(contents, ids, scores):
		assert len(content_list) == len(id_list) == len(score_list)
		assert len(id_list) <= 2  # top_k respected
		for i, (content, _id, score) in enumerate(
			zip(content_list, id_list, score_list)
		):
			assert isinstance(content, str)
			assert isinstance(_id, str)
			assert isinstance(score, float)
			assert _id in corpus_data["doc_id"].tolist()
			if i >= 1:
				assert score_list[i - 1] >= score_list[i]
