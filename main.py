import cudf
import pandas as pd
import numpy as np

print(cudf.__version__)

USE_FULL_DF_GPU = True
FULL_DF = 'archive/yoochoose-data/yoochoose-clicks.dat'
SORTED_DF = 'archive/yoochoose-data/yoochoose-clicks-sorted.dat'
SAMPLE_DF = 'archive/yoochoose-data/sample_100k_clicks.dat'
SESS_121_DF = 'archive/yoochoose-data/session_121.csv'
ROWS_424_DF = 'archive/yoochoose-data/424_rows.dat'

if USE_FULL_DF_GPU:
    working_df = cudf.read_csv(FULL_DF,
                               names=['session_id', 'timestamp', 'item_id', 'category'])
else:
    working_df = pd.read_csv(FULL_DF, names=['session_id', 'timestamp', 'item_id', 'category'])

# working_df_sorted = working_df.sort_values(by=['session_id', 'timestamp'], ascending=[True, True])
# working_df_sorted.to_csv('/home/john/dev/NVIDIA/VS_KNN/archive/yoochoose-data/yoochoose-clicks-sorted.dat', index=False, header=False)


def preprocess_df(df, group_key, agg_key):
    return count_and_sort(collect_lists(df, group_key, agg_key))


def collect_lists(df, group_key, agg_key):
    df = df.groupby(group_key).agg({agg_key: list})
    return df


def count_and_sort(df, agg_key, ):
    count_key = 'n_' + agg_key
    df[count_key] = df[agg_key].list.len()
    return df.sort_values(by=count_key, ascending=False)


def top_items_per_sessions(df, num_keep=100, group_key='session_id', sort_col='timestamp'):
    df = df.sort_values(by=[group_key, sort_col], ascending=[True, False]).reset_index().drop(columns='index')
    df['item_pos'] = df.groupby(group_key).cumcount()
    df = df.loc[df.item_pos < num_keep]

    num_items = df.groupby(group_key).agg({'item_pos': 'max'})
    num_items.columns = ['num_items']

    df = df.set_index(group_key)
    df = df.join(num_items)
    df = df.reset_index()
    df = df.sort_values(by=[group_key, sort_col])

    df['item_value'] = df['num_items'] - df['item_pos']
    df = df.drop(columns=['category', 'item_pos', 'timestamp', 'num_items'])
    return df


def top_sessions_per_items(df, num_keep=5000, group_key='item_id', sort_col='session_id'):
    df = df.sort_values(by=[group_key, sort_col, 'timestamp'], ascending=[True, True, False]).reset_index().drop(columns='index')
    df = df.drop_duplicates(subset=[group_key, sort_col])
    df['session_pos'] = df.groupby(group_key).cumcount()
    df = df.loc[df.session_pos < num_keep]
    df = df.drop(columns=['category', 'session_pos', 'timestamp'])
    df = df.set_index(group_key)
    return df


sessions_df = top_items_per_sessions(working_df)
# session_to_items = preprocess_df(working_df, 'session_id', 'item_id')
# top_sessions = session_to_items[0:100].to_pandas()
sessions_df.to_csv('stored_indices/sessions_to_items.csv')


items_df = top_sessions_per_items(working_df)
items_df.to_csv('stored_indices/items_to_sessions.csv')
# item_to_sessions = preprocess_df(working_df, 'item_id', 'session_id')
# top_items = item_to_sessions[0:100].to_pandas()

a = 1

# Todo:
#  -up to 100 items in session, no need to keep more. !! Sort by session -> timestamp (-> item)
#  -up to 5000 sessions per item, no need to keep more. !!! sort by item -> Session -> Timestamp instead of session -> timestamp
#  Then ze algo:
#       input is a session, and we want to get k-nearest sessions quick:
#           -Try to index the item_to_session df with all items in session.
#               Keep only first occurence of each item
#           -convert results to a Series of unique value with pi_i column
#           -try to simply index session_to_items df with a bunch of sessions
#           -Left(?) join with pi_i values per item from original session
#           -Compare session to base one: Aggreg on sum of pi_i per session  << Keep this lookup!
#               Sort and keep K
#           -At this point we have K nearest sessions, need to calculate item value for each item in K sessions
#                   - For each item in sessions: Sum up all session distances (if item in session n)
#           -  Session distance x lambda thing (1 - 0.1 * max common val). Keep session distance from before?
#               - Get a Series with all items in K sessions: Join left K sessions to items per sessions
#               - Join with Session weight saved before.
#               - Tricky part: What is the max wheight of similar item? Join on rank with initial session,
#               keep same items only, then keep max rank.  Use 0.01 * rank as val.
#               - Sun all ranks * pi across items: That's the SCOORE !


