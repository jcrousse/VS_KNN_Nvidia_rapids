from vs_knn.vs_knn import CupyVsKnnModel
import cudf


def shuffle_df(df: cudf.DataFrame):
    """Test results should be unaffected by row ordering """
    return df.sample(frac=1).reset_index().drop(columns=['index'])


def test_cupymodel(tiny_vsknn_df, tiny_session):
    model = CupyVsKnnModel(top_k=2)
    model.train(tiny_vsknn_df)

    predicted_items, predicted_score = model.predict(tiny_session)

    stop = 1
    # todo: also check the proper filtering of most recent items per sessions and sessions per items
    # todo: Unseen values replaced by 0s instead of skipped?
    #  todo: remove duplicated elements per session: should be binary vectors


def test_cupymodel_str():
    pass


def test_cupymodel_cudfmodel():
    pass

