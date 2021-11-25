import pytest

wwc = pytest.importorskip("vs_knn.weighted_word_count")


def test_wcount_array(single_row, single_weight):
    unique_values, weights = wwc.weighted_word_count(single_row, single_weight)
    assert(sum(weights) == len(unique_values))


def test_wcount_item5w1(random_bin_matrix, weight_array_ones):
    unique_values, weights = wwc.weighted_word_count(random_bin_matrix, weight_array_ones)
    assert len(unique_values) == 1
    assert sum(sum(random_bin_matrix)) == int(weights[0])

    unique_values, weights = wwc.weighted_word_count(random_bin_matrix * 5, weight_array_ones)
    assert len(unique_values) == 1
    assert sum(sum(random_bin_matrix)) == int(weights[0])

    unique_values, weights = wwc.weighted_word_count(random_bin_matrix, weight_array_ones * 2.5)
    assert (sum(sum(random_bin_matrix)) * 2.5) == weights[0]
