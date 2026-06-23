import numpy as np
import pytest

from autorag.nodes.passagefilter import ScoreGateFilter
from autorag.nodes.passagefilter.score_gate import adaptive_score_gate
from autorag.support import get_support_modules
from tests.autorag.nodes.passagefilter.test_passage_filter_base import (
	contents_example,
	scores_example,
	ids_example,
	base_passage_filter_test,
	project_dir,
	previous_result,
	base_passage_filter_node_test,
)


@pytest.fixture
def score_gate_instance():
	return ScoreGateFilter(project_dir=project_dir, previous_result=previous_result)


def test_support_registry_resolves_score_gate():
	# the wiring edit in autorag/support.py must surface the new filter
	assert get_support_modules("score_gate") is ScoreGateFilter


def test_adaptive_cardinality_keeps_above_average():
	# scores [0.1, 0.8, 0.1, 0.5] -> mean 0.375 -> keep {0.8, 0.5}
	remain = adaptive_score_gate([0.1, 0.8, 0.1, 0.5])
	assert remain == [1, 3]


def test_cross_encoder_rescue():
	# bi-encoder ranks index 0 poorly, but the cross-encoder affirms it
	bi = [0.1, 0.8, 0.1, 0.5]
	cross = [0.9, 0.1, 0.1, 0.1]
	remain = adaptive_score_gate(bi, cross_scores=cross)
	assert 0 in remain  # rescued by cross-encoder affirmation
	assert set(remain) == {0, 1, 3}


def test_never_drops_everything():
	remain = adaptive_score_gate([0.5, 0.5, 0.5])
	assert len(remain) >= 1


def test_max_keep_caps_retained():
	remain = adaptive_score_gate([0.1, 0.8, 0.6, 0.5], max_keep=1)
	assert remain == [1]


def test_score_gate_pure(score_gate_instance):
	contents, ids, scores = score_gate_instance._pure(
		contents_example, scores_example, ids_example
	)
	base_passage_filter_test(contents, ids, scores)
	# adaptive selection retains fewer chunks than the fixed top-K input
	assert all(len(s) < 4 for s in scores)


def test_score_gate_pure_numpy(score_gate_instance):
	numpy_scores = np.array([[0.1, 0.8, 0.1, 0.5], [0.1, 0.2, 0.7, 0.3]])
	contents, ids, scores = score_gate_instance._pure(
		contents_example, numpy_scores, ids_example
	)
	base_passage_filter_test(contents, ids, scores)


def test_score_gate_node():
	result_df = ScoreGateFilter.run_evaluator(
		project_dir=project_dir, previous_result=previous_result, z=0.0
	)
	base_passage_filter_node_test(result_df)
