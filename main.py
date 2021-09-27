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



a = 1

# Todo:
#  -Split train / test: (maybe just use pandas there, it is one-off)
#       select all session IDX in Numpy. Random selection. Join with DF?
#  -Calculate Session Index --DONE
#  -Calculate Item Index --DONE
#  Then ze algo:
#       input is a session, and we want to get k-nearest sessions quick:
#           -Try to index the item_to_session df with all items in session.
#               Keep only first occurence of each item before index, then first of
#               each session (set, then df.loc then .drop_duplicates)
#           -convert results to a Series where index= item_id and value= pi_i
#           -Do a join with the session index?
#           -Left(?) join with pi_i values per item from original session
#           -Keep only items per session in original session (another join):
#             Aggreg on sum of pi_i per session  << Keep this lookup!
#             Sort and keep K
#           -At this point we have K nearest sessions, need to calculate item value for each item in K sessions
#                   - For each item in sessions: Sum up all session distances (if item in session n)
#                       This is a join with a DF for original session n
#           -  Session distance x lambda thing (1 - 0.1 * max common val). Keep session distance from before?
#               - Get a Series with all items in K sessions: Join left K sessions to items per sessions
#               - Join with Session weight saved before.
#               - Tricky part: What is the max wheight of similar item? Join on rank with initial session,
#               keep same items only, then keep max rank.  Use 0.01 * rank as val.
#               - Sun all ranks * pi across items: That's the SCOORE !


