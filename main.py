from vs_knn.index_handler import IndexHander
from vs_knn.train_test_split import train_test_split

vs_knn_indices = IndexHander()

TRAIN_TEST_SPLIT = False
CALCULATE_INDICES = False

if TRAIN_TEST_SPLIT:
    train_test_split()

if CALCULATE_INDICES:
    vs_knn_indices.create_indices()
else:
    vs_knn_indices.load_indices()


import json
import pandas as pd
import cudf
from vs_knn.col_names import SESSION_ID, ITEM_ID, PI_I
with open('config.json', 'r') as f:
    project_config = json.load(f)

test_series = pd.read_csv(
    project_config['data_sources']['test_data'],
    names=['sess', 'tms', 'item', 'cat'])
processed_series = test_series.sort_values(by=['sess', 'tms'], ascending=[True, False]).drop(columns=['tms', 'cat'])
session_to_item = processed_series.groupby('sess')['item'].apply(list)
test_examples = pd.DataFrame(session_to_item).to_dict(orient='index')

test_examples = {k: v['item'][0:project_config['items_per_session']] for k, v in test_examples.items()}

top_k = project_config['top_k']

examples_100 = list(test_examples.keys())[0:100]


def step1_query_to_cudf(session_items):
    """Convert session data to"""
    n_items = len(session_items)
    pi_i = [e / n_items for e in range(n_items, 0, -1)]
    session_df = cudf.DataFrame({'pi_i': pi_i},
                                index=session_items)
    return session_df


def step2_get_sessions_per_items(query_items, query_df):
    """For each item in query get past sessions containing the item.
    Returns dataframe with item_id (index) corresponding session_id and pi_i value"""
    past_sessions = vs_knn_indices.item_index.loc[query_items]
    items_sessions_pi = past_sessions.join(query_df)
    return items_sessions_pi


def step3_keep_topk(df):
    df = df.groupby(SESSION_ID).agg({PI_I: 'sum'})
    return df.sort_values(by=[PI_I], ascending=False)[0:top_k]


def step4_score_items_in_sessions(top_k_sessions):
    """for the top k sessions with similarity scores, get the items in the sessions.
    Then get total similarity per item"""
    top_k_items = vs_knn_indices.session_index.loc[top_k_sessions.index]
    sessions_with_items = top_k_sessions.join(top_k_items)
    item_scores = sessions_with_items.groupby(ITEM_ID).agg({PI_I: 'sum'})
    return item_scores

import datetime
import numpy as np
timus = np.zeros(len(examples_100))

for idx, user_session in enumerate(examples_100):
    query_items = test_examples[user_session]
    
    start = datetime.datetime.now()
    
    query_df = step1_query_to_cudf(query_items)
    items_sessions_pi = step2_get_sessions_per_items(query_items, query_df)
    top_k_sessions = step3_keep_topk(items_sessions_pi)
    item_scores = step4_score_items_in_sessions(top_k_sessions)

    end = datetime.datetime.now()
    delta = end - start
    timus[idx] = int(delta.total_seconds() * 1000)

print("average duration: ", np.average(timus), " milliseconds")
print("p90 duration: ", np.percentile(timus, 90), " milliseconds")





# we want dict {session_id: {items: []} ordered}


# Todo:
#  -Split train / test: (maybe just use pandas there, it is one-off)
#       select all session IDX in Numpy. Random selection. Join with DF?
#  -Calculate Session Index
#  -Calculate Item Index
#  Then ze algo:
#     for each of the k sessions: Get the items, then group by item where the weight to apply is the session's max p_i
#     score per item = Sum of session similarities for sessions that have the item.
#     session similarities = pi_i values (where match) * (optional) lambda function
#     !! Let's skip the lambda and just use sim
#     !!Shortcut: Not limit to k sessions?
#     ----------------HIGH LEVEL todos:
#     -Finish main algo
#     -Compare perfs pandas - cudf
#     -Box plots of time spent on each step
#     -Low level CUDA approach:
#       -2D array of item to sessions
#       -2D array of sessions to items
#       -Accumulator "output" out_s array of sessions to consider (Size N_Sessions ? Can it be optimized)
#       -Accumulator "output" out_i array of Items to consider (Size N_items ? Can it be optimized)
#       -3D kernel grid: Item x Session x Item = 2 x 500 x 100 (x, y, z)
#       -For each thread: If within bounds, and item z == item x, add weight f(y) to session accumulator out_s
#       -Keep top K sessions in out_s. Sort?
#       -Create accumulator for items in  sessions (size K * z ?)
#       -New kernel y * z: if out_s[y] > threshold(K), increment weight of out_i[z] by out_s[y]
#       -out_s contains items we are looking for. sort it



