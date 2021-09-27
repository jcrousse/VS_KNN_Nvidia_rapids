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
for test_session in test_examples:
    session_items = test_examples[test_session]
    n_items = len(session_items)
    pi_i = [e / n_items for e in range(n_items - 1, -1, -1)]
    session_data = cudf.DataFrame({'pi_i': pi_i},
                                  index=session_items)
    all_sessions = vs_knn_indices.item_index.loc[session_items]
    unique_sessions = all_sessions.drop_duplicates()
    sess_item_pos = unique_sessions.join(session_data)
    sess_item_pos = sess_item_pos.set_index(SESSION_ID)

    sess_item_pos = sess_item_pos.join(vs_knn_indices.session_index)

    if top_k:
        sessions_w_intersection = sess_item_pos[sess_item_pos[ITEM_ID].isin(session_items)]
        sessions_w_intersection = sessions_w_intersection.reset_index()
        session_values = sessions_w_intersection.groupby(SESSION_ID).agg({PI_I: 'sum'})
        top_k_sessions = session_values.sort_values(by=[PI_I], ascending=False)[0:top_k]




# we want dict {session_id: {items: []} ordered}


# Todo:
#  -Split train / test: (maybe just use pandas there, it is one-off)
#       select all session IDX in Numpy. Random selection. Join with DF?
#  -Calculate Session Index
#  -Calculate Item Index
#  Then ze algo:
#       input is a session, and we want to get k-nearest sessions quick:
#           -Try to index the item_to_session df with all items in session.
#               Keep only first occurrence of each item before index, then first of
#               each session (set, then df.loc then .drop_duplicates)
#           -convert results to a Series where index= item_id and value= pi_i
#           -Do a join with the session index?
#           -Left(?) join with pi_i values per item from original session
#           -Keep only items per session in original session (another join):
#             Aggreg on sum of pi_i per session  << Keep this lookup!
#             Sort and keep K
#             -----DONE UNTIL HERE- -------------
#           -At this point we have K nearest sessions, need to calculate item value for each item in K sessions
#                   - For each item in sessions: Sum up all session distances (if item in session n)
#                       This is a join with a DF for original session n
#           -  Session distance x lambda thing (1 - 0.1 * max common val). Keep session distance from before?
#               - Get a Series with all items in K sessions: Join left K sessions to items per sessions
#               - Join with Session weight saved before.
#               - Tricky part: What is the max weight of similar item? Join on rank with initial session,
#               keep same items only, then keep max rank.  Use 0.01 * rank as val.
#               - Sun all ranks * pi across items: That's the SCOORE !
#     !!Shortcut: Not limit to k sessions?


