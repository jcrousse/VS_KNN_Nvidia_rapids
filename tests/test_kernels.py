import pytest
import cupy as cp
import cudf
from cupyx.time import repeat

kernels = pytest.importorskip("vs_knn.custom_kernels")

sessions_per_item, items_per_session = 5000, 10
start_idx = [0, 1000, 5000, 10000, 15000, 20000, 25000, 30000]
len_per_idx = [1, 10, 5, 2, 3, 5, 2, 10]
start_end_idx = [[e, e + l] for e, l in zip(start_idx, len_per_idx)]

values_array = cp.arange(17646935, dtype=cp.intc)
test_session = cp.array([1, 2, 3, 6, 2, 2, 2], dtype=cp.intc)
idx_array = cp.vstack([start_end_idx[int(i)] for i in test_session]).astype(cp.intc)
weight_array = cp.arange(len(test_session), dtype=cp.float32) / len(test_session)

buffer_shape = sessions_per_item * items_per_session
out_values = cp.random.randint(1, 100, buffer_shape, dtype=cp.intc)
out_weights = cp.random.random(buffer_shape, dtype=cp.float32)

values_buffer = cp.random.randint(0, 500, buffer_shape, dtype=cp.intc)
weight_buffer = cp.random.random(buffer_shape, dtype=cp.float32)

unique_values = cp.unique(values_buffer)


def test_copy_kernel():
    for _ in range(3):
        run_copy_kernel()
        _ = [value_array_check(i) for i in [1, 100, 1000, 5000, 5004, 5005, 25001]]


def run_copy_kernel():
    kernel_args_val = (idx_array, values_array,
                       sessions_per_item, items_per_session, buffer_shape,
                       out_values)
    kernel_args_wei = (idx_array, weight_array,
                       len(idx_array), sessions_per_item, items_per_session, buffer_shape,
                       out_weights)
    t_per_block = 256
    target_threads = len(idx_array) * sessions_per_item
    n_blocks = int(target_threads / t_per_block) + 1
    kernels.copy_values_kernel((n_blocks, ), (t_per_block,), kernel_args_val)
    kernels.copy_weights_kernel((n_blocks,), (t_per_block,), kernel_args_wei)


def value_array_check(out_idx):
    idx_in_input_session = out_idx // sessions_per_item
    idx_in_hist_sessions = out_idx % sessions_per_item
    item_id = int(test_session[idx_in_input_session])
    value_idx = start_idx[item_id] + idx_in_hist_sessions
    if idx_in_hist_sessions > len_per_idx[item_id]:
        expected_value = 0
        expected_weight = 0.0
    else:
        expected_value = int(values_array[value_idx])
        expected_weight = float(weight_array[idx_in_input_session])

    assert expected_value == int(out_values[out_idx])
    assert expected_weight == float(out_weights[out_idx])


def test_copy_kernel_speed():
    print(repeat(run_copy_kernel, n_repeat=10000))


def test_groupby_kernel_speed():
    print(repeat(run_groupby_kernel, n_repeat=1000))


def test_groupby_kernel():
    calc_weights = run_groupby_kernel()
    weight_array_check(calc_weights)


def run_groupby_kernel():
    out_weights_groupby = cp.zeros_like(unique_values, dtype=cp.float32)
    n_items = len(values_buffer) * len(unique_values)
    kernel_args = (values_buffer, weight_buffer, unique_values,
                   len(unique_values), n_items,
                   out_weights_groupby)
    t_per_block = 256
    n_blocks = int( n_items/ t_per_block) + 1
    kernels.groubpy_kernel((n_blocks, ), (t_per_block,), kernel_args)
    return out_weights_groupby


def weight_array_check(calculated_weights):
    df = cudf.DataFrame(data={
        'group_key': values_buffer,
        'agg_val': weight_buffer
    })
    df = df.groupby('group_key').agg({'agg_val': 'sum'}).sort_index()
    expected_weights = df['agg_val'].values

    def f_to_i(x):
        return round(float(x) * 10)
    comparison_key = [int(ek) == int(ck) for ek, ck in zip(unique_values, df.index.values)]
    comparison_weights = [f_to_i(ew) == f_to_i(cw) for ew, cw in zip(expected_weights, calculated_weights)]

    assert all(comparison_key)
    assert all(comparison_weights)
