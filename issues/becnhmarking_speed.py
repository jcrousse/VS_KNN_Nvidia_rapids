import numpy as np
import cupy as cp
from cupyx.time import repeat

n_items = 20000
n_sessions = 500000
items_per_session = 30
sessions_per_item = 500

session_to_item_np = np.random.randint(0, n_items, (n_sessions, items_per_session))

random_rows_select = np.random.randint(0, n_sessions, sessions_per_item)


def transfer_to_gpu():
    return cp.array(session_to_item_np[random_rows_select])


def get_a_row(table, possible_values):
    # idx = cp.random.randint(0, possible_values)
    row = table[123, :]
    return row


if __name__ == '__main__':
    print(repeat(transfer_to_gpu, n_repeat=1000))

    # 1.5ms for transferring 5000 * 100
    # 0.14ms for transferring 500 * 100
    # 0.062ms for transferring 500 * 30
