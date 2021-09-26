import cudf
import pandas as pd


FULL_DF = 'archive/yoochoose-data/yoochoose-clicks.dat'
SESSION_INDEX = 'stored_indices/sessions_to_items.csv'
ITEM_INDEX = 'stored_indices/items_to_sessions.csv'

SESSION_ID = 'session_id'
TIMESTAMP = 'timestamp'
ITEM_ID = 'item_id'
CATEGORY = 'category'
ITEM_VALUE = 'item_value'
working_df = cudf.read_csv(FULL_DF, names=[SESSION_ID, TIMESTAMP, ITEM_ID, CATEGORY])


def top_items_per_sessions(df, num_keep=100,):
    df = select_top_rows(df, num_keep, SESSION_ID, TIMESTAMP, ITEM_ID)
    df = calculate_item_val(df)
    df = df.drop(columns=[TIMESTAMP])
    return df


def calculate_item_val(df, ):
    df = df.sort_values(by=[SESSION_ID, TIMESTAMP], ascending=[True, True]).reset_index().drop(columns='index')
    df[ITEM_VALUE] = df.groupby(SESSION_ID).cumcount()
    return df


def top_sessions_per_items(df, num_keep=5000, group_key='item_id'):
    df = select_top_rows(df, num_keep, ITEM_ID, SESSION_ID, TIMESTAMP)
    df = df.drop(columns=[TIMESTAMP])
    df = df.set_index(group_key)
    return df


def select_top_rows(df, num_keep, group_key, sort_col1, sort_col2):
    df = df.sort_values(by=[group_key, sort_col1, sort_col2], ascending=[True, True, False]).reset_index() \
        .drop(columns='index')
    df['cum_count'] = df.groupby(group_key).cumcount()
    df = df.drop_duplicates(subset=[group_key, sort_col1])
    df = df.loc[df.cum_count < num_keep]
    df = df.drop(columns=[CATEGORY, 'cum_count'])
    return df


sessions_df = top_items_per_sessions(working_df)
sessions_df.to_csv(SESSION_INDEX)


items_df = top_sessions_per_items(working_df)
items_df.to_csv(ITEM_INDEX)


a = 1

# Todo:
#  -Calculate Session Index --DONE
#  -Calculate Item Index --DONE
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


