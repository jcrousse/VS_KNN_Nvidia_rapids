import pytest
import cupy as cp
import cudf
from cupyx.time import repeat

kernels = pytest.importorskip("vs_knn.custom_kernels")

sessions_per_item, items_per_session = 5000, 10
start_idx = [0, 1000, 5000, 10000, 15000, 20000, 25000, 30000]
len_per_idx = [0, 10, 5, 2, 3, 5, 2, 10]
start_end_idx = [[e, e + l] for e, l in zip(start_idx, len_per_idx)]

values_array = cp.arange(17646935, dtype=cp.intc)
test_sessions_py = [[1, 2, 3, 6, 2, 2, 2], [1, 2, 3]]
test_sessions = cp.vstack([cp.pad(cp.array(q, dtype=cp.intc), (0, items_per_session - len(q)))
                           for q in test_sessions_py])
# test_session = cp.array([1, 2, 3, 6, 2, 2, 2], dtype=cp.intc)

idx_array = cp.vstack([start_end_idx[int(i)] for i in test_sessions.flatten()]).astype(cp.intc)
weight_array = cp.arange(items_per_session, dtype=cp.float32) / items_per_session

buffer_len = sessions_per_item * items_per_session
out_values = cp.random.randint(1, 100, (len(test_sessions_py), buffer_len), dtype=cp.intc)
out_weights = cp.random.random((len(test_sessions_py), buffer_len), dtype=cp.float32)

values_buffer = cp.random.randint(1, 500, (len(test_sessions_py), buffer_len), dtype=cp.intc)
weight_buffer = cp.random.random((len(test_sessions_py), buffer_len), dtype=cp.float32)

groupby_v_buffer = cp.random.randint(1, 500, (len(test_sessions_py), buffer_len), dtype=cp.intc)

arrays_unique = [cp.unique(values_buffer[i, :]) for i in range(len(test_sessions_py))]
target_width = max([len(arr) for arr in arrays_unique])
unique_values = cp.vstack(
    [cp.pad(arr, (0, target_width - len(arr))) for arr in arrays_unique]
)


def test_tiny_copy():
    """
    Simple copy kernel test.

    Input sessions = [[1, 2, 4], [3, 4]]
    Converted to cupy with items_per_session=5:
        [[1, 2, 4, 0, 0],
        [3, 4, 0, 0, 0]]

    sessions_per_items=10
    batch_size = 2.

    idx_array =
        [[1, 3],
        [5, 10],
        [12, 16],
        [20, 22]]

    values_array = cp.arange(25)
    weights = [0.1, 0.2, 0.3, 0.4, 0.5]

    """
    # tiny_idx_array = cp.array([[1, 2, 4, 0, 0],
    #                            [3, 4, 0, 0, 0]], dtype=cp.intc)

    tiny_idx_array = cp.array([[1, 3], [5, 10], [20, 22], [0, 0], [0, 0],
                               [12, 16], [20, 22], [0, 0], [0, 0], [0, 0]], dtype=cp.intc)
    tiny_values = cp.arange(25, dtype=cp.intc)
    tiny_weights = cp.arange(5, dtype=cp.float32) / 10

    out_tv = cp.zeros((2, 50), dtype=cp.intc)
    out_tw = cp.zeros((2, 50), dtype=cp.float32)

    kernels.copy_values_kernel((1,), (100,),
                               (tiny_idx_array, tiny_values, tiny_weights,
                                len(tiny_idx_array), 10, 5, 50, 2,
                                out_tv, out_tw
    ))

    assert out_tv[0, :].sum() == 114
    assert out_tv[1, :].sum() == 133
    assert round(float(out_tw[0, :].sum()) * 10) == 12
    assert round(float(out_tw[1, :].sum()) * 10) == 3


def test_copy_kernel():
    for _ in range(3):
        run_copy_kernel()
        _ = [value_array_check(i, j) for i, j in [(1, 0),
                                                  (100, 0),
                                                  (1000, 0),
                                                  (5000, 0),
                                                  (5004, 0),
                                                  (5005, 0),
                                                  (25001, 0),
                                                  (0, 1),
                                                  (1, 1),
                                                  (5004, 1),
                                                  ]]


def run_copy_kernel():
    kernel_args = (idx_array, values_array, weight_array,
                   len(idx_array), sessions_per_item, items_per_session, buffer_len, len(test_sessions_py),
                   out_values, out_weights)
    t_per_block = 256
    target_threads = len(idx_array) * sessions_per_item
    n_blocks = int(target_threads / t_per_block) + 1
    kernels.copy_values_kernel((n_blocks, ), (t_per_block,), kernel_args)


def value_array_check(out_idx, session_id):
    idx_in_input_session = out_idx // sessions_per_item
    idx_in_hist_sessions = out_idx % sessions_per_item
    item_id = int(test_sessions[session_id, idx_in_input_session])
    value_idx = start_idx[item_id] + idx_in_hist_sessions
    if idx_in_hist_sessions > len_per_idx[item_id]:
        expected_value = 0
        expected_weight = 0.0
    else:
        expected_value = int(values_array[value_idx])
        expected_weight = float(weight_array[idx_in_input_session])

    assert expected_value == int(out_values[session_id, out_idx])
    assert expected_weight == float(out_weights[session_id, out_idx])


def test_copy_kernel_speed():
    print(repeat(run_copy_kernel, n_repeat=10000))


def test_groupby_kernel_speed():
    print(repeat(run_groupby_kernel, n_repeat=1000))


def test_tiny_groupby():
    """
    unique_values =
    [[1, 2, 3, 4, 0, 0, 0],
    [1, 2, 3, 0, 0, 0, 0]]

    values_buffer =
    [[1, 1, 0, 0, 2, 2, 2, 0, 3, 4],
    [1, 0, 0, 0, 2, 0, 0, 3, 0, 0]]

    weigts_buffer =
    [[1., 1., 0, 0, 2., 2., 2., 0, 3., 4],
    [1., 0, 0, 0, 2., 0, 0, 3., 0, 0]]

    result:
    [[2., 6., 3., 4., 0, 0, 0],
    [1., 2., 3., 0, 0, 0, 0]]
    """
    tiny_unique = cp.array(
        [[1, 2, 3, 4, 0, 0, 0],
         [1, 2, 3, 0, 0, 0, 0]], dtype=cp.intc)

    tiny_vb = cp.array(
        [[1, 1, 0, 0, 2, 2, 2, 0, 3, 4],
         [1, 0, 0, 0, 2, 0, 0, 3, 0, 0]], dtype=cp.intc)

    tiny_wb = cp.copy(tiny_vb).astype(cp.float32)
    res = cp.zeros_like(tiny_unique, dtype=cp.float32)
    kernels.groubpy_kernel((1,),
                           (256,),
                           (tiny_vb, tiny_wb, tiny_unique,
                            7, 10, 2,
                            res))
    res = res.astype(cp.intc)
    assert res[0, 0] == 2
    assert res[0, 1] == 6
    assert res[0, 2] == 3
    assert res[0, 3] == 4
    assert res[0, 4] == 0
    assert res[1, 0] == 1
    assert res[1, 1] == 2
    assert res[1, 2] == 3
    assert res[1, 3] == 0


def test_groupby_kernel():
    calc_weights = run_groupby_kernel()
    for batch_id in range(values_buffer.shape[0]):
        weight_array_check(calc_weights, batch_id)


def run_groupby_kernel():
    out_weights_groupby = cp.zeros_like(unique_values, dtype=cp.float32)
    kernel_args = (values_buffer, weight_buffer, unique_values,
                   unique_values.shape[1], values_buffer.shape[1], values_buffer.shape[0],
                   out_weights_groupby)
    t_per_block = 256
    n_items = unique_values.shape[1] * values_buffer.shape[1] * values_buffer.shape[0]
    n_blocks = int(n_items / t_per_block) + 1
    kernels.groubpy_kernel((n_blocks, ), (t_per_block,), kernel_args)
    return out_weights_groupby


def weight_array_check(calculated_weights, batch_id):
    df = cudf.DataFrame(data={
        'group_key': values_buffer[batch_id],
        'agg_val': weight_buffer[batch_id]
    })
    df = df.groupby('group_key').agg({'agg_val': 'sum'}).sort_index()
    expected_weights = df['agg_val'].values

    def f_to_i(x):
        return round(float(x) * 10)
    comparison_key = [int(ek) == int(ck) for ek, ck in zip(unique_values[batch_id], df.index.values)]
    comparison_weights = [f_to_i(ew) == f_to_i(cw) for ew, cw in zip(expected_weights, calculated_weights[batch_id])]

    assert all(comparison_key)
    assert all(comparison_weights)
