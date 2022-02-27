from vs_knn.vs_knn import CupyVsKnnModel
import cudf


def shuffle_df(df: cudf.DataFrame):
    """Test results should be unaffected by row ordering """
    return df.sample(frac=1).reset_index().drop(columns=['index'])


def test_cupymodel(tiny_vsknn_df, tiny_session):
    model = CupyVsKnnModel(top_k=2, max_sessions_per_items=20)
    model.train(tiny_vsknn_df)
    model_predict_test(model, tiny_session)
    # todo: also check the proper filtering of most recent items per sessions and sessions per items
    # todo: Unseen values replaced by 0s instead of skipped?


def model_predict_test(model, tiny_session):
    predicted_items, predicted_score = model.predict(tiny_session)
    predicted_score_py = [float(s) for s in predicted_score]
    assert all([pred == expected for pred, expected in zip(predicted_items, [1, 2, 3, 4, 5, 6])])
    assert all([abs(score - expected) < 0.01 for score, expected
                in zip(predicted_score_py, [2.0, 1.666, 2.0, 3.666, 3.666, 1.666])])


def test_save_load(tiny_vsknn_df, tiny_session, tmpdir):
    model = CupyVsKnnModel(top_k=2, max_sessions_per_items=20)
    model.train(tiny_vsknn_df)

    model.save(tmpdir)
    model_new = CupyVsKnnModel()
    model_new.load(tmpdir)

    model_predict_test(model_new, tiny_session)


def test_cupymodel_str():
    pass
