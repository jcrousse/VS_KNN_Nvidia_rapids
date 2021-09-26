from vs_knn.index_handler import IndexHander


vs_knn_indices = IndexHander()

CALCULATE_INDICES = True

if CALCULATE_INDICES:
    vs_knn_indices.create_indices()
else:
    vs_knn_indices.load_indices()



a = 1

# Todo:
#  -Split train / test
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


